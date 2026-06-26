import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import io
import os
from datetime import datetime

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageStat
from flask import Flask, jsonify, make_response, request, send_file
import torch
import torch.nn.functional as F

from model import Decomposition, LCNet, UNetDenoise


ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASIC_CHECKPOINT_PATH = os.path.join(BASE_DIR, "best_model.pth")
NIGHT_CHECKPOINT_PATH = os.path.join(BASE_DIR, "best_model_v2.pth")
DEFAULT_CHECKPOINT_PATH = BASIC_CHECKPOINT_PATH

PRESET_LIBRARY = {
    "balanced": {
        "label": "均衡增强",
        "description": "适合大多数低照度照片，强调自然提亮和稳定细节。",
        "gamma": 1.15,
        "brightness": 1.06,
        "contrast": 1.08,
        "saturation": 1.04,
        "sharpness": 1.06,
        "warmth": 4.0,
    },
    "night_detail": {
        "label": "夜景细节",
        "description": "增强暗部纹理，适合街景、建筑和灯光复杂场景。",
        "gamma": 1.35,
        "brightness": 1.12,
        "contrast": 1.14,
        "saturation": 1.06,
        "sharpness": 1.16,
        "warmth": 2.0,
    },
    "portrait_soft": {
        "label": "人像柔亮",
        "description": "亮度更柔和，饱和度控制更克制，适合人像和肤色表现。",
        "gamma": 1.18,
        "brightness": 1.10,
        "contrast": 1.02,
        "saturation": 1.01,
        "sharpness": 1.02,
        "warmth": 8.0,
    },
}


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024

BATCH_MAX_WORKERS = max(1, int(os.environ.get("BATCH_MAX_WORKERS", "2" if torch.cuda.is_available() else "4")))

model_bundles = {
    "night": {
        "decomposition": None,
        "l2h_net": None,
        "denoise_net": None,
        "checkpoint_path": NIGHT_CHECKPOINT_PATH,
    },
    "basic": {
        "decomposition": None,
        "l2h_net": None,
        "denoise_net": None,
        "checkpoint_path": BASIC_CHECKPOINT_PATH,
    },
}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Http跨域安全头
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


@app.after_request
def after_request(response):
    return add_cors_headers(response)

@app.route("/<path:path>", methods=["OPTIONS"])
@app.route("/", methods=["OPTIONS"])
def options(path=None):
    response = make_response()
    return add_cors_headers(response)


def allowed_file(filename):
    if not filename or "." not in filename:
        return False
    return filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))


def parse_bool(raw_value, default=False):
    if raw_value is None:
        return default
    return str(raw_value).strip().lower() in {"1", "true", "yes", "on"}


def parse_float(raw_value, default, minimum=None, maximum=None):
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        value = default
    if minimum is not None:
        value = max(minimum, value)
    if maximum is not None:
        value = min(maximum, value)
    return value


def image_from_bytes(image_bytes):
    try:
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise ValueError(f"图像读取失败: {exc}")

#图像尺寸向下取整（奇数）
def even_floor(value):
    return value if value % 2 == 0 else value - 1


def ensure_even_image_size(image, context="full_image"):
    original_width, original_height = image.size
    processed_width = even_floor(original_width)
    processed_height = even_floor(original_height)

    if processed_width < 2 or processed_height < 2:
        raise ValueError("图像尺寸过小。")

    adjusted = processed_width != original_width or processed_height != original_height
    if adjusted:
        image = image.crop((0, 0, processed_width, processed_height))

    return image, {
        "context": context,
        "adjusted": adjusted,
        "original_size": {
            "width": int(original_width),
            "height": int(original_height),
        },
        "processed_size": {
            "width": int(processed_width),
            "height": int(processed_height),
        },
        "strategy": "floor_to_even",
    }


def pil_to_tensor(image):
    image_np = np.asarray(image).astype(np.float32) / 255.0
    tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).to(device)
    return tensor

#将图片调整为偶数倍大小
def pad_tensor_to_multiple(tensor, multiple=4, mode="replicate"):
    _, _, height, width = tensor.shape
    pad_height = (multiple - height % multiple) % multiple
    pad_width = (multiple - width % multiple) % multiple

    if pad_height == 0 and pad_width == 0:
        return tensor, {
            "padded": False,
            "original_size": {"width": int(width), "height": int(height)},
            "processed_size": {"width": int(width), "height": int(height)},
            "pad": {"right": 0, "bottom": 0},
            "multiple": int(multiple),
            "strategy": "none",
        }

    padded = F.pad(tensor, (0, pad_width, 0, pad_height), mode=mode)
    return padded, {
        "padded": True,
        "original_size": {"width": int(width), "height": int(height)},
        "processed_size": {"width": int(width + pad_width), "height": int(height + pad_height)},
        "pad": {"right": int(pad_width), "bottom": int(pad_height)},
        "multiple": int(multiple),
        "strategy": f"pad_right_bottom_{mode}",
    }


def crop_tensor_to_original_size(tensor, size_info):
    width = size_info["original_size"]["width"]
    height = size_info["original_size"]["height"]
    return tensor[:, :, :height, :width]


def tensor_to_pil(tensor):
    tensor = torch.clamp(tensor, 0, 1)
    image_np = tensor.squeeze(0).detach().cpu().permute(1, 2, 0).numpy()
    image_np = (image_np * 255).astype(np.uint8)
    return Image.fromarray(image_np)

def image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def image_to_bytes(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    buffered.seek(0)
    return buffered


def apply_warmth(image, warmth_value):
    if abs(warmth_value) < 1e-6:
        return image

    array = np.asarray(image).astype(np.float32)
    scale = clamp(abs(warmth_value) / 30.0, 0.0, 2.0)
    if warmth_value > 0:
        array[..., 0] += 12.0 * scale
        array[..., 1] += 3.0 * scale
        array[..., 2] -= 10.0 * scale
    else:
        array[..., 0] -= 8.0 * scale
        array[..., 1] += 2.0 * scale
        array[..., 2] += 12.0 * scale

    array = np.clip(array, 0, 255).astype(np.uint8)
    return Image.fromarray(array)


def apply_post_adjustments(image, params):
    image = ImageEnhance.Brightness(image).enhance(params["brightness"])
    image = ImageEnhance.Contrast(image).enhance(params["contrast"])
    image = ImageEnhance.Color(image).enhance(params["saturation"])
    image = ImageEnhance.Sharpness(image).enhance(params["sharpness"])
    image = apply_warmth(image, params["warmth"])
    return image


def analyze_image(image):
    rgb = np.asarray(image).astype(np.float32) / 255.0
    luminance = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]#国际照明委员会（CIE）制定的标准权重

    max_channel = rgb.max(axis=2)
    min_channel = rgb.min(axis=2)
    saturation = np.divide(
        max_channel - min_channel,
        np.maximum(max_channel, 1e-6),
        out=np.zeros_like(max_channel),
        where=max_channel > 1e-6,
    )

    stats = ImageStat.Stat(image)
    return {
        "width": image.width,
        "height": image.height,
        "brightness_mean": round(float(luminance.mean()), 4),
        "brightness_std": round(float(luminance.std()), 4),
        "dark_ratio": round(float((luminance < 0.35).mean()), 4),
        "highlight_ratio": round(float((luminance > 0.85).mean()), 4),
        "saturation_mean": round(float(saturation.mean()), 4),
        "channel_mean": [round(value, 2) for value in stats.mean],
    }


def recommend_parameters(analysis):
    brightness = analysis["brightness_mean"]
    contrast = analysis["brightness_std"]
    dark_ratio = analysis["dark_ratio"]
    highlight_ratio = analysis["highlight_ratio"]
    saturation = analysis["saturation_mean"]

    if dark_ratio > 0.72 or brightness < 0.18:
        preset_name = "night_detail"
        reason = "暗部占比很高，优先补足阴影细节。"
    elif saturation < 0.12 and contrast < 0.12:
        preset_name = "document_clean"
        reason = "画面对比和色彩都偏弱，更适合净化式增强。"
    elif saturation < 0.18 and brightness < 0.30:
        preset_name = "portrait_soft"
        reason = "画面低亮且色彩克制，适合更柔和的人像式提亮。"
    else:
        preset_name = "balanced"
        reason = "场景整体均衡，使用通用增强参数更稳妥。"

    preset = PRESET_LIBRARY[preset_name]
    gamma = clamp(1.0 + (0.42 - brightness) * 1.35, 0.95, 1.65)
    brightness_gain = clamp(preset["brightness"] + (0.35 - brightness) * 0.35, 0.96, 1.28)
    contrast_gain = clamp(preset["contrast"] + (0.12 - contrast) * 0.55, 0.96, 1.24)
    saturation_gain = clamp(preset["saturation"] + (0.20 - saturation) * 0.45, 0.90, 1.20)
    sharpness_gain = clamp(preset["sharpness"] + (0.10 - contrast) * 0.55, 1.0, 1.30)
    warmth = preset["warmth"]

    if highlight_ratio > 0.18:
        gamma = clamp(gamma - 0.08, 0.9, 1.55)
        brightness_gain = clamp(brightness_gain - 0.03, 0.94, 1.22)

    return {
        "preset": preset_name,
        "reason": reason,
        "parameters": {
            "gamma": round(gamma, 2),
            "brightness": round(brightness_gain, 2),
            "contrast": round(contrast_gain, 2),
            "saturation": round(saturation_gain, 2),
            "sharpness": round(sharpness_gain, 2),
            "warmth": round(warmth, 2),
        },
    }


def resolve_processing_parameters(form, image=None):
    preset_name = form.get("preset", "balanced").strip().lower()
    if preset_name not in PRESET_LIBRARY:
        preset_name = "balanced"

    params = dict(PRESET_LIBRARY[preset_name])
    recommendation = None

    if parse_bool(form.get("auto_recommend"), default=False) and image is not None:
        analysis = analyze_image(image)
        recommendation = recommend_parameters(analysis)
        params.update(recommendation["parameters"])
        params["preset"] = recommendation["preset"]
    else:
        params["preset"] = preset_name

    params["gamma"] = parse_float(form.get("gamma"), params["gamma"], 0.1, 5.0)
    params["brightness"] = parse_float(form.get("brightness"), params["brightness"], 0.4, 2.2)
    params["contrast"] = parse_float(form.get("contrast"), params["contrast"], 0.4, 2.4)
    params["saturation"] = parse_float(form.get("saturation"), params["saturation"], 0.0, 2.4)
    params["sharpness"] = parse_float(form.get("sharpness"), params["sharpness"], 0.0, 3.0)
    params["warmth"] = parse_float(form.get("warmth"), params["warmth"], -60.0, 60.0)
    params["apply_adjustments"] = parse_bool(form.get("apply_adjustments"), default=False)
    params["denoise"] = True
    params["auto_recommend"] = recommendation is not None

    return params, recommendation


def parse_region(form, image_width, image_height):
    x = parse_float(form.get("region_x"), 0.0)
    y = parse_float(form.get("region_y"), 0.0)
    w = parse_float(form.get("region_w"), 0.0)
    h = parse_float(form.get("region_h"), 0.0)

    if w <= 0 or h <= 0:
        raise ValueError("局部增强需要有效的选区范围。")

    if x <= 1 and y <= 1 and w <= 1 and h <= 1:
        left = int(x * image_width)
        top = int(y * image_height)
        right = int((x + w) * image_width)
        bottom = int((y + h) * image_height)
    else:
        left = int(x)
        top = int(y)
        right = int(x + w)
        bottom = int(y + h)

    left = clamp(left, 0, image_width - 1)
    top = clamp(top, 0, image_height - 1)
    right = clamp(right, left + 1, image_width)
    bottom = clamp(bottom, top + 1, image_height)

    return left, top, right, bottom


def ensure_even_region(region):
    left, top, right, bottom = region
    original_width = right - left
    original_height = bottom - top

    right -= original_width % 2
    bottom -= original_height % 2

    processed_width = right - left
    processed_height = bottom - top

    if processed_width < 2 or processed_height < 2:
        raise ValueError("局部区域过小")

    return (left, top, right, bottom), {
        "context": "region",
        "adjusted": processed_width != original_width or processed_height != original_height,
        "original_region": {
            "left": int(left),
            "top": int(top),
            "width": int(original_width),
            "height": int(original_height),
        },
        "processed_region": {
            "left": int(left),
            "top": int(top),
            "width": int(processed_width),
            "height": int(processed_height),
            "right": int(right),
            "bottom": int(bottom),
        },
        "strategy": "floor_to_even_by_shrinking_right_bottom",
    }


def normalize_region_for_enhancement(region, image_width, image_height):
    left, top, right, bottom = region
    left = clamp(left, 0, image_width - 1)
    top = clamp(top, 0, image_height - 1)
    right = clamp(right, left + 1, image_width)
    bottom = clamp(bottom, top + 1, image_height)

    return ensure_even_region((left, top, right, bottom))


def validate_single_upload():
    if "file" not in request.files:
        raise ValueError("未找到上传的文件")

    file = request.files["file"]
    if file.filename == "":
        raise ValueError("文件名为空")
    if not allowed_file(file.filename):
        raise ValueError(f"不支持的文件格式，仅支持: {sorted(ALLOWED_EXTENSIONS)}")

    return file


def load_model_bundle(checkpoint_path):
    if not os.path.exists(checkpoint_path):
        print(f"Model checkpoint not found: {checkpoint_path}")
        return None

    checkpoint = torch.load(checkpoint_path, map_location=device)
    bundle = {
        "decomposition": Decomposition().to(device),
        "l2h_net": LCNet(mode="brighten").to(device),
        "denoise_net": UNetDenoise().to(device),
        "checkpoint_path": checkpoint_path,
    }

    bundle["decomposition"].load_state_dict(checkpoint["Decom_net_state_dict"])
    bundle["l2h_net"].load_state_dict(checkpoint["L2H_net_state_dict"])
    bundle["denoise_net"].load_state_dict(checkpoint["Denoise_net_state_dict"])

    bundle["decomposition"].eval()
    bundle["l2h_net"].eval()
    bundle["denoise_net"].eval()
    return bundle


def load_models():
    global model_bundles

    for mode, bundle in list(model_bundles.items()):
        loaded_bundle = load_model_bundle(bundle["checkpoint_path"])
        if loaded_bundle is None:
            return False
        model_bundles[mode] = loaded_bundle

    return True


def models_ready():
    return all(
        bundle["decomposition"] is not None
        and bundle["l2h_net"] is not None
        and bundle["denoise_net"] is not None
        for bundle in model_bundles.values()
    )


def resolve_enhancement_mode(form, params=None):
    raw_mode = str(form.get("enhance_mode", "")).strip().lower()
    if raw_mode in {"night", "night_detail", "night-light", "night_light"}:
        return "night"
    if raw_mode in {"basic", "base", "standard", "default"}:
        return "basic"
    if params is not None and params.get("preset") == "night_detail":
        return "night"
    return "basic"


def get_model_bundle(mode):
    bundle = model_bundles.get(mode)
    if bundle is None:
        raise RuntimeError(f"Unsupported enhancement mode: {mode}")
    return bundle

def enhance_low_light(img_tensor, gamma=1.0, denoise=True, enhancement_mode="basic"):
    with torch.no_grad():
        try:
            if not models_ready():
                raise RuntimeError("模型未加载完成")

            img_tensor, tensor_size_adjustment = pad_tensor_to_multiple(img_tensor, multiple=4)
            bundle = get_model_bundle(enhancement_mode)
            reflectance, illumination = bundle["decomposition"](img_tensor)
            enhanced_light = bundle["l2h_net"](illumination)

            reflectance = bundle["denoise_net"](reflectance)
 
            reflectance = F.interpolate(
                reflectance,
                size=enhanced_light.shape[2:],
                mode="bilinear",
                align_corners=True,
            )
            enhanced_img = reflectance * enhanced_light

            if gamma != 1.0:
                enhanced_img = torch.pow(torch.clamp(enhanced_img, min=1e-6), 1.0 / gamma)

            if tensor_size_adjustment["padded"]:
                enhanced_img = crop_tensor_to_original_size(enhanced_img, tensor_size_adjustment)

            return torch.clamp(enhanced_img, 0, 1)
        except Exception as exc:
            raise RuntimeError(f"Image enhancement failed: {exc}")


def enhance_pil_image(image, params, context="full_image", enhancement_mode="basic"):
    image, size_adjustment = ensure_even_image_size(image, context=context)
    tensor = pil_to_tensor(image)
    enhanced_tensor = enhance_low_light(
        tensor,
        gamma=params["gamma"],
        denoise=params["denoise"],
        enhancement_mode=enhancement_mode,
    )
    enhanced_image = tensor_to_pil(enhanced_tensor)

    if params["apply_adjustments"]:
        enhanced_image = apply_post_adjustments(enhanced_image, params)

    return enhanced_image, size_adjustment


def build_json_image_response(image, filename, params, recommendation=None, analysis=None, size_adjustment=None, enhancement_mode="basic"):
    return jsonify(
        {
            "success": True,
            "message": "图像增强成功",
            "enhanced_image": image_to_base64(image),
            "parameters": {
                "preset": params["preset"],
                "gamma": params["gamma"],
                "brightness": params["brightness"],
                "contrast": params["contrast"],
                "saturation": params["saturation"],
                "sharpness": params["sharpness"],
                "warmth": params["warmth"],
                "apply_adjustments": params["apply_adjustments"],
                "denoise": True,
                "original_filename": filename,
                "enhance_mode": enhancement_mode,
            },
            "recommendation": recommendation,
            "analysis": analysis,
            "size_adjustment": size_adjustment,
        }
    )


def apply_region_enhancement(image, params, region, feather_radius=None, enhancement_mode="basic"):
    image, image_adjustment = ensure_even_image_size(image, context="region_full_image")
    region, region_adjustment = normalize_region_for_enhancement(region, image.width, image.height)
    left, top, right, bottom = region

    crop = image.crop((left, top, right, bottom))
    enhanced_crop, crop_size_adjustment = enhance_pil_image(
        crop,
        params,
        context="region_crop",
        enhancement_mode=enhancement_mode,
    )

    canvas = image.copy()
    canvas.paste(enhanced_crop, (left, top))

    if feather_radius is None:
        feather_radius = max(8, int(min(right - left, bottom - top) * 0.08))

    mask = Image.new("L", image.size, 0)
    region_mask = Image.new("L", (right - left, bottom - top), 255)
    mask.paste(region_mask, (left, top))
    mask = mask.filter(ImageFilter.GaussianBlur(radius=feather_radius))

    return Image.composite(canvas, image, mask), {
        "image": image_adjustment,
        "region": region_adjustment,
        "crop": crop_size_adjustment,
    }


@app.route("/")
def index():
    return jsonify(
        {
            "message": "低光照图像增强",
            "status": "running",
            "device": str(device),
            "models_loaded": models_ready(),
            "timestamp": datetime.now().isoformat(),
        }
    )


@app.route("/health")
def health_check():
    status_code = 200 if models_ready() else 503
    return (
        jsonify(
            {
                "status": "healthy" if models_ready() else "degraded",
                "models_loaded": models_ready(),
                "device": str(device),
                "timestamp": datetime.now().isoformat(),
            }
        ),
        status_code,
    )


@app.route("/info")
def get_info():
    return jsonify(
        {
            "api_name": "低光照图像增强",
            "device": str(device),
            "models_loaded": models_ready(),
            "supported_formats": sorted(ALLOWED_EXTENSIONS),
            "default_checkpoint": DEFAULT_CHECKPOINT_PATH,
            "checkpoint_paths": {
                "night": NIGHT_CHECKPOINT_PATH,
                "basic": BASIC_CHECKPOINT_PATH,
            },
            "presets": PRESET_LIBRARY,
        }
    )


@app.route("/presets")
def get_presets():
    return jsonify(
        {
            "success": True,
            "default": "balanced",
            "presets": PRESET_LIBRARY,
        }
    )


@app.route("/recommend_params", methods=["POST"])
def recommend_params():
    try:
        file = validate_single_upload()
        image = image_from_bytes(file.read())
        analysis = analyze_image(image)
        recommendation = recommend_parameters(analysis)

        return jsonify(
            {
                "success": True,
                "analysis": analysis,
                "recommendation": recommendation,
                "preset_detail": PRESET_LIBRARY[recommendation["preset"]],
            }
        )
    except ValueError as exc:
        return jsonify({"success": False, "message": str(exc)}), 400
    except Exception as exc:
        return jsonify({"success": False, "message": f"智能推荐失败: {exc}"}), 500


@app.route("/enhance", methods=["POST"])
def enhance_image():
    try:
        if not models_ready():
            return jsonify({"success": False, "message": "模型未成功加载"}), 503

        file = validate_single_upload()
        image = image_from_bytes(file.read())
        analysis = analyze_image(image)
        params, recommendation = resolve_processing_parameters(request.form, image=image)
        enhancement_mode = resolve_enhancement_mode(request.form, params)

        enhanced_image, size_adjustment = enhance_pil_image(
            image,
            params,
            context="full_image",
            enhancement_mode=enhancement_mode,
        )
        output_format = request.form.get("output_format", "base64").strip().lower()

        if output_format == "file":
            output_bytes = image_to_bytes(enhanced_image)
            response = send_file(
                output_bytes,
                mimetype="image/png",
                as_attachment=True,
                download_name=f"enhanced_{file.filename}"
            )
            return add_cors_headers(response)

        return build_json_image_response(
            enhanced_image,
            file.filename,
            params,
            recommendation=recommendation,
            analysis=analysis,
            size_adjustment=size_adjustment,
            enhancement_mode=enhancement_mode,
        )

    except ValueError as exc:
        return jsonify({"success": False, "message": str(exc)}), 400
    except RuntimeError as exc:
        return jsonify({"success": False, "message": str(exc)}), 500
    except Exception as exc:
        return jsonify({"success": False, "message": f"处理失败: {exc}"}), 500


@app.route("/enhance_region", methods=["POST"])
def enhance_region():
    try:
        if not models_ready():
            return jsonify({"success": False, "message": "模型未成功加载"}), 503

        file = validate_single_upload()
        image = image_from_bytes(file.read())
        analysis = analyze_image(image)
        params, recommendation = resolve_processing_parameters(request.form, image=image)
        enhancement_mode = resolve_enhancement_mode(request.form, params)
        region = parse_region(request.form, image.width, image.height)

        feather_radius = int(parse_float(request.form.get("feather_radius"), 0, 0, 128)) or None
        result, size_adjustment = apply_region_enhancement(
            image,
            params,
            region,
            feather_radius=feather_radius,
            enhancement_mode=enhancement_mode,
        )

        return build_json_image_response(
            result,
            file.filename,
            params,
            recommendation=recommendation,
            analysis={
                **analysis,
                "region": size_adjustment["region"]["processed_region"],
            },
            size_adjustment=size_adjustment,
            enhancement_mode=enhancement_mode,
        )

    except ValueError as exc:
        return jsonify({"success": False, "message": str(exc)}), 400
    except RuntimeError as exc:
        return jsonify({"success": False, "message": str(exc)}), 500
    except Exception as exc:
        return jsonify({"success": False, "message": f"局部增强失败: {exc}"}), 500


def process_batch_image(filename, image_bytes, form_data):
    try:
        image = image_from_bytes(image_bytes)
        analysis = analyze_image(image)
        params, recommendation = resolve_processing_parameters(form_data, image=image)
        enhancement_mode = resolve_enhancement_mode(form_data, params)
        params["apply_adjustments"] = parse_bool(form_data.get("apply_adjustments"), default=True)

        enhanced_image, size_adjustment = enhance_pil_image(
            image,
            params,
            context="batch_full_image",
            enhancement_mode=enhancement_mode,
        )
        return {
            "filename": filename,
            "success": True,
            "enhanced_image": image_to_base64(enhanced_image),
            "parameters": {
                "preset": params["preset"],
                "gamma": params["gamma"],
                "brightness": params["brightness"],
                "contrast": params["contrast"],
                "saturation": params["saturation"],
                "sharpness": params["sharpness"],
                "warmth": params["warmth"],
                "apply_adjustments": params["apply_adjustments"],
                "enhance_mode": enhancement_mode,
            },
            "recommendation": recommendation,
            "analysis": analysis,
            "size_adjustment": size_adjustment,
        }
    except Exception as exc:
        return {
            "filename": filename,
            "success": False,
            "error": str(exc),
        }


@app.route("/enhance_batch", methods=["POST"])
def enhance_batch():
    try:
        if not models_ready():
            return jsonify({"success": False, "message": "模型未成功加载"}), 503

        if "files" not in request.files:
            return jsonify({"success": False, "message": "未找到上传的文件"}), 400

        files = request.files.getlist("files")
        if not files:
            return jsonify({"success": False, "message": "未选择文件"}), 400

        form_data = request.form.copy()
        tasks = []
        results = [None] * len(files)

        for index, file in enumerate(files):
            if not file or file.filename == "":
                continue

            if not allowed_file(file.filename):
                results[index] = {
                    "filename": file.filename,
                    "success": False,
                    "error": f"不支持的文件格式，仅支持: {sorted(ALLOWED_EXTENSIONS)}",
                }
                continue

            try:
                tasks.append((index, file.filename, file.read()))
            except Exception as exc:
                results[index] = {
                    "filename": file.filename,
                    "success": False,
                    "error": str(exc),
                }

        worker_count = min(BATCH_MAX_WORKERS, len(tasks)) if tasks else 0
        if worker_count:
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                future_map = {
                    executor.submit(process_batch_image, filename, image_bytes, form_data): index
                    for index, filename, image_bytes in tasks
                }
                for future in as_completed(future_map):
                    results[future_map[future]] = future.result()

        results = [item for item in results if item is not None]

        success_count = sum(1 for item in results if item.get("success"))
        return jsonify(
            {
                "success": True,
                "message": "批量处理完成",
                "total": len(results),
                "success_count": success_count,
                "worker_count": worker_count,
                "results": results,
            }
        )

    except Exception as exc:
        return jsonify({"success": False, "message": f"批量处理失败: {exc}"}), 500

if load_models():
    print("模型加载成功！")
else:
    print("模型加载失败")
print("=" * 60)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False, threaded=True)
