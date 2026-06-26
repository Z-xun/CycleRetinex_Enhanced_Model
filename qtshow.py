import base64
import io
import os
import sys

import requests
from PIL import Image, ImageEnhance
from PyQt5.QtCore import QEasingCurve, QObject, QPoint, QPropertyAnimation, QRect, QThread, QTimer, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QImage, QLinearGradient, QPainter, QPalette, QPen, QPixmap
from PyQt5.QtWidgets import (QApplication,QAbstractItemView,QCheckBox,QFileDialog,QFrame,QGraphicsDropShadowEffect,QGridLayout,
    QHBoxLayout,QLabel,QListWidget,QListWidgetItem,QMainWindow,QMessageBox,QComboBox,QPushButton,QProgressBar,QScrollArea,
    QSizePolicy,QSlider,QTabWidget,QVBoxLayout,QWidget,)


API_BASE = "http://127.0.0.1:8000"

PRESETS = {
    "balanced": {
        "label": "均衡增强",
        "description": "适合大多数低照度场景，整体观感稳妥自然。",
        "gamma": 1.15,
        "brightness": 1.06,
        "contrast": 1.08,
        "saturation": 1.04,
        "sharpness": 1.06,
        "warmth": 4.0,
    },
    "night_detail": {
        "label": "夜景细节",
        "description": "强调暗部纹理和边缘清晰度，适合街景和建筑。",
        "gamma": 1.35,
        "brightness": 1.12,
        "contrast": 1.14,
        "saturation": 1.06,
        "sharpness": 1.16,
        "warmth": 2.0,
    },
    "portrait_soft": {
        "label": "人像柔亮",
        "description": "柔和提亮并保持肤色自然，适合人物主体。",
        "gamma": 1.18,
        "brightness": 1.10,
        "contrast": 1.02,
        "saturation": 1.01,
        "sharpness": 1.02,
        "warmth": 8.0,
    },
}


def pil_to_qpixmap(image):
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    data = image.tobytes("raw", "RGBA")
    q_image = QImage(data, image.width, image.height, QImage.Format_RGBA8888)
    return QPixmap.fromImage(q_image.copy())


def base64_to_pil(data):
    return Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB")


def format_value(value, scale=100.0, suffix=""):
    return f"{value / scale:.2f}{suffix}"


class ApiWorker(QObject):
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, endpoint, data=None, file_paths=None, field_name="file", timeout=180):
        super().__init__()
        self.endpoint = endpoint
        self.data = data or {}
        self.file_paths = file_paths or []
        self.field_name = field_name
        self.timeout = timeout

    def run(self):
        handles = [] #保存文件句柄
        try:
            files = None
            if self.file_paths:
                files = []
                for path in self.file_paths:
                    handle = open(path, "rb")
                    handles.append(handle)
                    files.append((self.field_name, (os.path.basename(path), handle, "application/octet-stream")))

            response = requests.post(
                f"{API_BASE}{self.endpoint}",
                data=self.data,
                files=files,
                timeout=self.timeout,
            )

            try:
                payload = response.json()
            except ValueError:
                payload = None

            if not response.ok:
                if payload and payload.get("message"):
                    raise RuntimeError(payload["message"])
                response.raise_for_status()#检查网络请求是否成功

            if payload is None:
                raise RuntimeError("返回的数据无法解析。")

            if payload.get("success") is False:
                raise RuntimeError(payload.get("message", "返回了失败状态。"))

            self.finished.emit(payload)
        except Exception as exc:
            self.failed.emit(str(exc))
        finally:
            for handle in handles:
                handle.close()


class ParameterSlider(QWidget):
    valueChanged = pyqtSignal()

    def __init__(self, title, minimum, maximum, value, scale=100.0, hint="", parent=None):
        super().__init__(parent)
        self.scale = scale

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)

        self.title_label = QLabel(title)
        self.title_label.setObjectName("paramTitle")
        header.addWidget(self.title_label)

        header.addStretch(1)

        self.value_label = QLabel()
        self.value_label.setObjectName("paramValue")
        header.addWidget(self.value_label)
        layout.addLayout(header)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(minimum, maximum)
        self.slider.setValue(value)
        self.slider.valueChanged.connect(self.on_change)
        layout.addWidget(self.slider)

        if hint:
            hint_label = QLabel(hint)
            hint_label.setObjectName("paramHint")
            hint_label.setWordWrap(True)
            layout.addWidget(hint_label)

        self.on_change()

    def on_change(self):
        self.value_label.setText(format_value(self.slider.value(), self.scale))
        self.valueChanged.emit()

    def value(self):
        return self.slider.value() / self.scale

    def set_value(self, value):
        self.slider.setValue(int(value))


class ControlCard(QFrame):
    def __init__(self, title, description="", parent=None):
        super().__init__(parent)
        self.setObjectName("controlCard")

        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(28)
        shadow.setOffset(0, 12)
        shadow.setColor(QColor(19, 28, 45, 40))
        self.setGraphicsEffect(shadow)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(14)

        title_label = QLabel(title)
        title_label.setObjectName("sectionTitle")
        layout.addWidget(title_label)

        if description:
            desc_label = QLabel(description)
            desc_label.setObjectName("sectionDesc")
            desc_label.setWordWrap(True)
            layout.addWidget(desc_label)

        self.content_layout = QVBoxLayout()
        self.content_layout.setSpacing(12)
        layout.addLayout(self.content_layout)


class SelectionCanvas(QWidget):
    selectionChanged = pyqtSignal(object)

    def __init__(self, selectable=False, parent=None):
        super().__init__(parent)
        self.setObjectName("imagePreview")
        self.setMinimumHeight(280)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.pixmap = QPixmap()
        self.placeholder = "等待载入图片"
        self.selectable = selectable
        self.display_rect = QRect()
        self.selection_norm = None #归一化坐标
        self.dragging = False
        self.drag_start = QPoint()
        self.drag_end = QPoint()

    def set_pixmap(self, pixmap):
        self.pixmap = pixmap
        self.update()

    def clear(self, text):
        self.pixmap = QPixmap()
        self.placeholder = text
        self.selection_norm = None
        self.update()

    def set_selectable(self, selectable):
        self.selectable = selectable
        if not selectable and self.selection_norm is not None:
            self.selection_norm = None
            self.selectionChanged.emit(None)
        self.update()

    def clear_selection(self):
        self.selection_norm = None
        self.dragging = False
        self.update()
        self.selectionChanged.emit(None)
    #坐标转化函数
    def normalized_to_rect(self):
        if not self.selection_norm or self.display_rect.isNull():
            return QRect()

        x, y, w, h = self.selection_norm
        return QRect(
            self.display_rect.x() + int(x * self.display_rect.width()),
            self.display_rect.y() + int(y * self.display_rect.height()),
            max(1, int(w * self.display_rect.width())),
            max(1, int(h * self.display_rect.height())),
        )

    def current_drag_rect(self):
        start = self.clamp_point(self.drag_start)
        end = self.clamp_point(self.drag_end)
        return QRect(start, end).normalized()#自动处理从左向右拖拽

    def clamp_point(self, point):
        if self.display_rect.isNull():
            return point
        x = max(self.display_rect.left(), min(point.x(), self.display_rect.right()))
        y = max(self.display_rect.top(), min(point.y(), self.display_rect.bottom()))
        return QPoint(x, y)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        card_rect = self.rect().adjusted(0, 0, -1, -1)
        painter.setPen(QPen(QColor(145, 160, 193, 80), 1, Qt.SolidLine))
        painter.setBrush(QColor(243, 247, 255, 180))
        painter.drawRoundedRect(card_rect, 22, 22)

        content_rect = self.rect().adjusted(14, 14, -14, -14)
        self.display_rect = QRect()

        if not self.pixmap.isNull():
            scaled_size = self.pixmap.size()
            scaled_size.scale(content_rect.size(), Qt.KeepAspectRatio)
            x = content_rect.x() + (content_rect.width() - scaled_size.width()) // 2
            y = content_rect.y() + (content_rect.height() - scaled_size.height()) // 2
            self.display_rect = QRect(x, y, scaled_size.width(), scaled_size.height())
            painter.drawPixmap(self.display_rect, self.pixmap)
        else:
            painter.setPen(QColor(123, 137, 170))
            painter.drawText(content_rect, Qt.AlignCenter, self.placeholder)

        if not self.display_rect.isNull() and (self.selection_norm or self.dragging):
            selection_rect = self.current_drag_rect() if self.dragging else self.normalized_to_rect()
            painter.setBrush(QColor(38, 78, 255, 50))
            painter.setPen(QPen(QColor(59, 102, 255), 2, Qt.DashLine))
            painter.drawRoundedRect(selection_rect, 14, 14)

            corner_text = "局部增强区域"
            painter.setPen(Qt.white)
            painter.fillRect(QRect(selection_rect.x(), selection_rect.y() - 24, 106, 22), QColor(59, 102, 255, 210))
            painter.drawText(QRect(selection_rect.x() + 8, selection_rect.y() - 24, 96, 22), Qt.AlignVCenter, corner_text)

    def mousePressEvent(self, event):
        if event.button() == Qt.RightButton and self.selectable:
            self.clear_selection()
            return

        if not self.selectable or self.display_rect.isNull():
            return
        if event.button() == Qt.LeftButton and self.display_rect.contains(event.pos()):
            self.dragging = True
            self.drag_start = self.clamp_point(event.pos())
            self.drag_end = self.drag_start
            self.update()

    def mouseMoveEvent(self, event):
        if not self.dragging:
            return
        self.drag_end = self.clamp_point(event.pos())
        self.update()

    def mouseReleaseEvent(self, event):
        if not self.dragging:
            return

        self.dragging = False
        rect = self.current_drag_rect()
        if rect.width() < 16 or rect.height() < 16:
            self.selection_norm = None
            self.selectionChanged.emit(None)
            self.update()
            return

        self.selection_norm = (
            (rect.x() - self.display_rect.x()) / self.display_rect.width(),
            (rect.y() - self.display_rect.y()) / self.display_rect.height(),
            rect.width() / self.display_rect.width(),
            rect.height() / self.display_rect.height(),
        )
        self.selectionChanged.emit(self.selection_norm)
        self.update()


class ImageCard(QFrame):
    def __init__(self, title, subtitle, selectable=False, parent=None):
        super().__init__(parent)
        self.setObjectName("imageCard")
        self.setMinimumSize(420, 320)

        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(34)
        shadow.setOffset(0, 14)
        shadow.setColor(QColor(22, 30, 48, 50))
        self.setGraphicsEffect(shadow)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 20)
        layout.setSpacing(14)

        title_label = QLabel(title)
        title_label.setObjectName("cardTitle")
        layout.addWidget(title_label)

        subtitle_label = QLabel(subtitle)
        subtitle_label.setObjectName("cardSubtitle")
        subtitle_label.setWordWrap(True)
        layout.addWidget(subtitle_label)

        self.canvas = SelectionCanvas(selectable=selectable)
        layout.addWidget(self.canvas)

    def set_pil_image(self, image):
        self.canvas.set_pixmap(pil_to_qpixmap(image))

    def clear(self, text):
        self.canvas.clear(text)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.presets = PRESETS
        self.current_preset = "balanced"
        self.enhancement_mode = "basic"
        self.original_image = None
        self.base_image = None
        self.preview_image = None
        self.current_image_path = None
        self.selection_norm = None
        self.recommendation_payload = None
        self.batch_file_paths = []
        self.batch_results = []
        self.worker_refs = {}

        self.preview_timer = QTimer(self)
        self.preview_timer.setInterval(60)
        self.preview_timer.setSingleShot(True)
        self.preview_timer.timeout.connect(self.refresh_preview)

        self.setWindowTitle("弱光图像增强系统")
        self.resize(1600, 980)
        self.setMinimumSize(1320, 840)

        self.build_ui()
        self.apply_styles()
        self.apply_preset("balanced", schedule=False)
        self.animate_window()
        self.update_meta_summary()

    def build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)

        main_layout = QVBoxLayout(root)
        main_layout.setContentsMargins(24, 22, 24, 24)
        main_layout.setSpacing(18)

        hero = QFrame()
        hero.setObjectName("heroPanel")
        hero_layout = QHBoxLayout(hero)
        hero_layout.setContentsMargins(28, 24, 28, 24)
        hero_layout.setSpacing(24)

        hero_left = QVBoxLayout()
        hero_left.setSpacing(10)

        badge = QLabel(" ")
        badge.setObjectName("heroBadge")
        hero_left.addWidget(badge)

        title = QLabel("弱光增强系统")
        title.setObjectName("heroTitle")
        title.setWordWrap(True)
        hero_left.addWidget(title)


        tag_row = QHBoxLayout()
        tag_row.setSpacing(10)
        for text in ["批量工作", "智能推荐", "局部增强"]:
            tag = QLabel(text)
            tag.setObjectName("heroPill")
            tag_row.addWidget(tag)
        tag_row.addStretch(1)
        hero_left.addLayout(tag_row)

        hero_layout.addLayout(hero_left, 1)

        hero_right = QVBoxLayout()
        hero_right.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        self.status_chip = QLabel("就绪")
        self.status_chip.setObjectName("statusChip")
        hero_right.addWidget(self.status_chip, alignment=Qt.AlignRight)

        self.file_chip = QLabel("未选择图片")
        self.file_chip.setObjectName("fileChip")
        hero_right.addWidget(self.file_chip, alignment=Qt.AlignRight)

        self.meta_chip = QLabel("均衡增强 | Gamma 1.15")
        self.meta_chip.setObjectName("fileChip")
        hero_right.addWidget(self.meta_chip, alignment=Qt.AlignRight)

        hero_layout.addLayout(hero_right)
        main_layout.addWidget(hero)

        self.tabs = QTabWidget()
        self.tabs.setObjectName("mainTabs")
        main_layout.addWidget(self.tabs, 2)

        self.single_tab = QWidget()
        self.batch_tab = QWidget()
        self.tabs.addTab(self.single_tab, "单图工作")
        self.tabs.addTab(self.batch_tab, "批量增强工作")

        self.build_single_tab()
        self.build_batch_tab()

    def build_single_tab(self):
        layout = QHBoxLayout(self.single_tab)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(18)

        left_column = QVBoxLayout()
        left_column.setSpacing(18)

        preview_grid = QGridLayout()
        preview_grid.setSpacing(18)
        self.original_card = ImageCard("原始图像", "在原图上拖拽框选后，可以只增强选中区域。右键可清空选区。", selectable=True)
        self.original_card.canvas.selectionChanged.connect(self.on_selection_changed)
        self.result_card = ImageCard("结果预览", "整图增强或局部增强完成后，这里会显示当前带微调的最终效果。")
        preview_grid.addWidget(self.original_card, 0, 0)
        preview_grid.addWidget(self.result_card, 0, 1)
        left_column.addLayout(preview_grid)

        toolbar = QFrame()
        toolbar.setObjectName("actionBar")
        tool_layout = QHBoxLayout(toolbar)
        tool_layout.setContentsMargins(18, 16, 18, 16)
        tool_layout.setSpacing(12)

        self.import_btn = self.make_button("导入图片", "primary", self.open_image)
        self.recommend_btn = self.make_button("智能推荐", "ghost", self.request_recommendation)
        self.enhance_btn = self.make_button("整图增强", "accent", self.request_enhance)
        self.region_btn = self.make_button("局部增强", "ghost", self.request_region_enhance)
        self.export_btn = self.make_button("导出结果", "ghost", self.export_image)
        self.reset_btn = self.make_button("重置参数", "ghost", self.reset_parameters)

        self.enhance_btn.setEnabled(False)
        self.region_btn.setEnabled(False)
        self.export_btn.setEnabled(False)
        self.reset_btn.setEnabled(False)
        self.recommend_btn.setEnabled(False)

        for button in [
            self.import_btn,
            self.recommend_btn,
            self.enhance_btn,
            self.region_btn,
            self.export_btn,
            self.reset_btn,
        ]:
            tool_layout.addWidget(button)
        tool_layout.addStretch(1)
        left_column.addWidget(toolbar)

        self.region_status = QLabel("当前未选择局部增强区域。")
        self.region_status.setObjectName("regionLabel")
        left_column.addWidget(self.region_status)

        self.single_progress_label = QLabel("当前暂无增强任务。")
        self.single_progress_label.setObjectName("sectionDesc")
        self.single_progress_label.setWordWrap(True)
        left_column.addWidget(self.single_progress_label)

        self.single_progress = QProgressBar()
        self.single_progress.setRange(0, 100)
        self.single_progress.setValue(0)
        self.single_progress.setFormat("等待开始")
        left_column.addWidget(self.single_progress)

        layout.addLayout(left_column, 7)

        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        right_scroll.setObjectName("controlScroll")

        right_body = QWidget()
        right_layout = QVBoxLayout(right_body)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(18)

        preset_card = ControlCard("增强预设", "预设会同步改写模型 gamma 与前端微调参数，适合快速切换不同风格。")
        preset_grid = QGridLayout()
        preset_grid.setSpacing(10)
        self.preset_buttons = {}
        for index, key in enumerate(self.presets):
            button = QPushButton(self.presets[key]["label"])
            button.setObjectName("presetButton")
            button.setCheckable(True)
            button.clicked.connect(lambda checked, preset_key=key: self.apply_preset(preset_key))
            self.preset_buttons[key] = button
            preset_grid.addWidget(button, index // 2, index % 2)
        preset_card.content_layout.addLayout(preset_grid)

        self.preset_desc = QLabel()
        self.preset_desc.setObjectName("sectionDesc")
        self.preset_desc.setWordWrap(True)
        preset_card.content_layout.addWidget(self.preset_desc)
        right_layout.addWidget(preset_card)

        analysis_card = ControlCard("智能参数推荐", "  ")
        self.analysis_summary = QLabel("导入图片后点击“智能推荐”，系统会自动分析当前场景。")
        self.analysis_summary.setObjectName("sectionDesc")
        self.analysis_summary.setWordWrap(True)
        analysis_card.content_layout.addWidget(self.analysis_summary)

        self.metric_labels = {}
        for metric_name in ["平均亮度", "亮度方差", "暗部占比", "平均饱和度", "推荐结果"]:
            label = QLabel(f"{metric_name}：-")
            label.setObjectName("metricText")
            analysis_card.content_layout.addWidget(label)
            self.metric_labels[metric_name] = label
        right_layout.addWidget(analysis_card)

        model_card = ControlCard("模型参数", " ")
        self.gamma_slider = ParameterSlider("Gamma", 10, 300, 115, hint="范围 0.10 - 3.00，影响模型增强阶段的亮度映射。")
        self.gamma_slider.valueChanged.connect(self.on_parameter_changed)
        model_card.content_layout.addWidget(self.gamma_slider)
        mode_row = QHBoxLayout()
        mode_row.setSpacing(10)
        mode_label = QLabel("增亮模式")
        mode_label.setObjectName("sectionDesc")
        mode_row.addWidget(mode_label)
        self.enhancement_mode_combo = QComboBox()
        self.enhancement_mode_combo.addItem("基础增亮", "basic")
        self.enhancement_mode_combo.addItem("夜光增亮", "night")
        self.enhancement_mode_combo.currentIndexChanged.connect(self.on_mode_changed)
        mode_row.addWidget(self.enhancement_mode_combo, 1)
        model_card.content_layout.addLayout(mode_row)
        right_layout.addWidget(model_card)

        refine_card = ControlCard("实时精修", " ")
        self.brightness_slider = ParameterSlider("亮度", 40, 180, 106)
        self.contrast_slider = ParameterSlider("对比度", 40, 220, 108)
        self.saturation_slider = ParameterSlider("饱和度", 0, 220, 104)
        self.sharpness_slider = ParameterSlider("锐化", 0, 300, 106)
        self.warmth_slider = ParameterSlider("色温", -60, 60, 4, scale=1.0, hint="正值偏暖，负值偏冷。")

        for slider in [
            self.brightness_slider,
            self.contrast_slider,
            self.saturation_slider,
            self.sharpness_slider,
            self.warmth_slider,
        ]:
            slider.valueChanged.connect(self.on_parameter_changed)
            refine_card.content_layout.addWidget(slider)
        right_layout.addWidget(refine_card)

        region_card = ControlCard("局部增强说明", "在左侧原图区域拖拽框选，点击“局部增强”即可只增强选中的区域，边缘会自动羽化过渡。")
        self.region_detail = QLabel("尚未选择局部区域。")
        self.region_detail.setObjectName("sectionDesc")
        self.region_detail.setWordWrap(True)
        region_card.content_layout.addWidget(self.region_detail)

        clear_region_btn = self.make_button("清空选区", "ghost", self.clear_region_selection)
        region_card.content_layout.addWidget(clear_region_btn)
        right_layout.addWidget(region_card)

        tips_card = ControlCard("工作流建议", "先用智能推荐快速获得合适参数，再做整图增强，最后用滑块和局部增强处理重点区域，会更高效。")
        for text in [
            "批量模式默认沿用当前单图面板参数。",
            "右键点击原图可以快速清除局部增强选区。",
            "导出的单图结果为当前预览效果，包含实时微调。",
        ]:
            tip = QLabel(f"• {text}")
            tip.setObjectName("tipsText")
            tip.setWordWrap(True)
            tips_card.content_layout.addWidget(tip)
        right_layout.addWidget(tips_card)
        right_layout.addStretch(1)

        right_scroll.setWidget(right_body)
        layout.addWidget(right_scroll, 3)

    def build_batch_tab(self):
        layout = QHBoxLayout(self.batch_tab)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(18)

        left_panel = QVBoxLayout()
        left_panel.setSpacing(18)

        batch_toolbar = QFrame()
        batch_toolbar.setObjectName("actionBar")
        batch_toolbar_layout = QHBoxLayout(batch_toolbar)
        batch_toolbar_layout.setContentsMargins(18, 16, 18, 16)
        batch_toolbar_layout.setSpacing(12)

        self.batch_import_btn = self.make_button("导入多张图片", "primary", self.select_batch_files)
        self.batch_run_btn = self.make_button("开始批量增强", "accent", self.run_batch_enhance)
        self.batch_export_btn = self.make_button("导出全部结果", "ghost", self.export_batch_results)
        self.batch_clear_btn = self.make_button("清空批量列表", "ghost", self.clear_batch_files)
        self.batch_run_btn.setEnabled(False)
        self.batch_export_btn.setEnabled(False)

        for button in [self.batch_import_btn, self.batch_run_btn, self.batch_export_btn, self.batch_clear_btn]:
            batch_toolbar_layout.addWidget(button)
        batch_toolbar_layout.addStretch(1)

        self.batch_auto_checkbox = QCheckBox("批量时为每张图自动推荐参数")
        self.batch_auto_checkbox.setChecked(True)
        batch_toolbar_layout.addWidget(self.batch_auto_checkbox)
        left_panel.addWidget(batch_toolbar)

        queue_card = ControlCard("批量任务队列", "这里展示待处理图片和处理结果摘要。批量模式会直接使用当前面板参数或自动推荐结果输出最终图像。")
        self.batch_progress = QProgressBar()
        self.batch_progress.setRange(0, 100)
        self.batch_progress.setValue(0)
        queue_card.content_layout.addWidget(self.batch_progress)

        self.batch_summary = QLabel("尚未导入批量图片。")
        self.batch_summary.setObjectName("sectionDesc")
        self.batch_summary.setWordWrap(True)
        queue_card.content_layout.addWidget(self.batch_summary)

        self.batch_list = QListWidget()
        self.batch_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.batch_list.currentRowChanged.connect(self.on_batch_item_selected)
        queue_card.content_layout.addWidget(self.batch_list)
        left_panel.addWidget(queue_card, 1)

        layout.addLayout(left_panel, 4)

        right_panel = QVBoxLayout()
        right_panel.setSpacing(18)

        self.batch_preview_card = ImageCard("批量结果预览", "点击左侧列表项可预览该图片的增强结果。")
        right_panel.addWidget(self.batch_preview_card, 1)

        detail_card = ControlCard("批量输出信息", "这里显示当前选中批量结果的参数和推荐信息。")
        self.batch_detail = QLabel("暂无批量结果。")
        self.batch_detail.setObjectName("sectionDesc")
        self.batch_detail.setWordWrap(True)
        detail_card.content_layout.addWidget(self.batch_detail)
        right_panel.addWidget(detail_card)

        layout.addLayout(right_panel, 3)

    def make_button(self, text, kind, handler):
        button = QPushButton(text)
        button.setObjectName(kind)
        button.setCursor(Qt.PointingHandCursor)
        button.clicked.connect(handler)
        return button

    def apply_styles(self):
        self.setStyleSheet(
            """
            QWidget {
                color: #172033;
                font-family: "Segoe UI", "Microsoft YaHei";
                font-size: 14px;
            }
            #heroPanel {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(255,255,255,242),
                    stop:0.55 rgba(242,247,255,232),
                    stop:1 rgba(229,238,255,240));
                border: 1px solid rgba(255,255,255,210);
                border-radius: 28px;
            }
            #heroBadge {
                color: #4662ff;
                background: rgba(70, 98, 255, 26);
                border: 1px solid rgba(70, 98, 255, 46);
                border-radius: 14px;
                padding: 7px 12px;
                font-weight: 600;
                max-width: 250px;
            }
            #heroTitle {
                font-size: 30px;
                font-weight: 700;
                color: #10182a;
            }
            #heroSubtitle, #sectionDesc, #tipsText, #paramHint, #cardSubtitle, #regionLabel, #metricText {
                color: #61708c;
                line-height: 1.5;
            }
            #heroPill {
                background: rgba(255,255,255,150);
                border: 1px solid rgba(255,255,255,205);
                border-radius: 15px;
                padding: 7px 12px;
                color: #33415c;
                font-weight: 600;
            }
            #statusChip {
                background: rgba(17,190,126,18);
                color: #0f8b59;
                border: 1px solid rgba(15,139,89,45);
                border-radius: 16px;
                padding: 8px 14px;
                font-weight: 700;
            }
            #fileChip {
                background: rgba(255,255,255,158);
                color: #53627f;
                border: 1px solid rgba(255,255,255,190);
                border-radius: 16px;
                padding: 8px 14px;
                max-width: 300px;
            }
            QTabWidget::pane {
                border: none;
                background: transparent;
            }
            QTabBar::tab {
                background: rgba(255,255,255,165);
                border: 1px solid rgba(255,255,255,200);
                padding: 12px 24px;
                margin-right: 8px;
                border-top-left-radius: 16px;
                border-top-right-radius: 16px;
                color: #4b5c7b;
                font-weight: 700;
            }
            QTabBar::tab:selected {
                background: rgba(255,255,255,236);
                color: #21365d;
            }
            #imageCard, #controlCard, #actionBar {
                background: rgba(255,255,255,228);
                border-radius: 26px;
                border: 1px solid rgba(255,255,255,206);
            }
            #cardTitle, #sectionTitle {
                font-size: 18px;
                font-weight: 700;
                color: #152136;
            }
            #presetButton {
                min-height: 44px;
                padding: 0 16px;
                border-radius: 16px;
                background: rgba(243,247,255,210);
                color: #31405f;
                border: 1px solid rgba(134,149,183,70);
                font-weight: 700;
            }
            #presetButton:checked {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #4662ff, stop:1 #6b8cff);
                color: white;
                border: none;
            }
            #imagePreview {
                background: transparent;
            }
            #paramTitle {
                color: #20304d;
                font-weight: 600;
            }
            #paramValue {
                color: #4662ff;
                font-weight: 700;
            }
            QSlider::groove:horizontal {
                height: 8px;
                border-radius: 4px;
                background: rgba(205,214,233,180);
            }
            QSlider::sub-page:horizontal {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #5c7cff, stop:1 #11b3c8);
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: white;
                border: 3px solid #4b68ff;
                width: 18px;
                height: 18px;
                margin: -7px 0;
                border-radius: 12px;
            }
            QPushButton {
                min-height: 46px;
                padding: 0 18px;
                border-radius: 18px;
                font-weight: 700;
                font-size: 14px;
                border: none;
            }
            QPushButton#primary {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #4662ff, stop:1 #6388ff);
                color: white;
            }
            QPushButton#accent {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #ff8d57, stop:1 #ffb067);
                color: white;
            }
            QPushButton#ghost {
                background: rgba(244,247,255,190);
                color: #31405f;
                border: 1px solid rgba(134,149,183,70);
            }
            QPushButton:disabled {
                background: rgba(205,211,224,180);
                color: rgba(72,81,104,160);
            }
            QListWidget {
                background: rgba(243,247,255,170);
                border: 1px solid rgba(132, 149, 181, 70);
                border-radius: 18px;
                padding: 8px;
            }
            QListWidget::item {
                border-radius: 12px;
                padding: 10px 12px;
                margin: 4px 0;
            }
            QListWidget::item:selected {
                background: rgba(70, 98, 255, 22);
                color: #20304d;
            }
            QProgressBar {
                background: rgba(236,240,248,205);
                border-radius: 10px;
                text-align: center;
                height: 22px;
                color: #31405f;
            }
            QProgressBar::chunk {
                border-radius: 10px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #4662ff, stop:1 #11b3c8);
            }
            QCheckBox {
                spacing: 10px;
                color: #30415f;
                font-weight: 600;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
            }
            QCheckBox::indicator:unchecked {
                background: rgba(255,255,255,180);
                border: 2px solid rgba(87,103,141,140);
                border-radius: 6px;
            }
            QCheckBox::indicator:checked {
                background: #4565ff;
                border: 2px solid #4565ff;
                border-radius: 6px;
            }
            QScrollArea {
                background: transparent;
                border: none;
            }
            QScrollBar:vertical {
                width: 10px;
                background: transparent;
                margin: 8px 0;
            }
            QScrollBar::handle:vertical {
                background: rgba(137,153,191,120);
                border-radius: 5px;
                min-height: 28px;
            }
            """
        )

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        gradient = QLinearGradient(0, 0, self.width(), self.height())
        gradient.setColorAt(0.0, QColor(243, 247, 255))
        gradient.setColorAt(0.45, QColor(234, 241, 255))
        gradient.setColorAt(1.0, QColor(248, 242, 237))
        painter.fillRect(self.rect(), gradient)

        painter.setBrush(QColor(112, 146, 255, 22))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(QRect(self.width() - 370, 54, 290, 290))
        painter.setBrush(QColor(255, 166, 108, 24))
        painter.drawEllipse(QRect(-70, self.height() - 280, 290, 290))
        super().paintEvent(event)

    def animate_window(self):
        self.fade_animation = QPropertyAnimation(self, b"windowOpacity", self)
        self.fade_animation.setDuration(520)
        self.fade_animation.setStartValue(0.92)
        self.fade_animation.setEndValue(1.0)
        self.fade_animation.setEasingCurve(QEasingCurve.OutCubic)
        self.fade_animation.start()

    def start_worker(self, name, endpoint, file_paths, data, field_name, on_success):
        if name in self.worker_refs:
            return

        worker = ApiWorker(endpoint=endpoint, data=data, file_paths=file_paths, field_name=field_name)
        thread = QThread(self)
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.finished.connect(on_success)
        worker.failed.connect(lambda message, op=name: self.on_worker_error(op, message))
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(lambda op=name: self.cleanup_worker(op))

        self.worker_refs[name] = (thread, worker)
        thread.start()

    def cleanup_worker(self, name):
        if name in self.worker_refs:
            del self.worker_refs[name]

    def on_worker_error(self, operation_name, message):
        self.cleanup_worker(operation_name)

        if operation_name == "recommend":
            self.recommend_btn.setEnabled(self.current_image_path is not None)
        elif operation_name in {"enhance", "region"}:
            self.enhance_btn.setEnabled(self.current_image_path is not None)
            self.region_btn.setEnabled(self.current_image_path is not None and self.selection_norm is not None)
            self.finish_single_progress(success=False)
        elif operation_name == "batch":
            self.batch_run_btn.setEnabled(bool(self.batch_file_paths))
            self.batch_progress.setRange(0, 100)
            self.batch_progress.setValue(0)

        self.set_status("请求失败", success=False)
        QMessageBox.critical(self, "接口调用失败", message)

    def set_status(self, text, success=True):
        if success is True:
            style = (
                "background: rgba(17,190,126,18); color: #0f8b59; "
                "border: 1px solid rgba(15,139,89,45); border-radius: 16px; padding: 8px 14px; font-weight: 700;"
            )
        elif success is False:
            style = (
                "background: rgba(238,83,92,18); color: #c63c42; "
                "border: 1px solid rgba(198,60,66,45); border-radius: 16px; padding: 8px 14px; font-weight: 700;"
            )
        else:
            style = (
                "background: rgba(70,98,255,16); color: #3652dd; "
                "border: 1px solid rgba(70,98,255,36); border-radius: 16px; padding: 8px 14px; font-weight: 700;"
            )
        self.status_chip.setStyleSheet(style)
        self.status_chip.setText(text)

    def start_single_progress(self, message):
        self.single_progress.setRange(0, 0)
        self.single_progress.setFormat("处理中...")
        self.single_progress_label.setText(message)

    def finish_single_progress(self, success=True):
        self.single_progress.setRange(0, 100)
        self.single_progress.setValue(100 if success else 0)
        self.single_progress.setFormat("已完成" if success else "未完成")
        if success:
            self.single_progress_label.setText("当前增强任务已完成，可以继续微调或导出结果。")
        else:
            self.single_progress_label.setText("增强任务失败，请检查服务器状态或重试。")

    def reset_single_progress(self):
        self.single_progress.setRange(0, 100)
        self.single_progress.setValue(0)
        self.single_progress.setFormat("等待开始")
        self.single_progress_label.setText("当前暂无增强任务。")

    def update_meta_summary(self):
        preset_label = self.presets[self.current_preset]["label"]
        mode_label = "夜光增亮" if self.enhancement_mode == "night" else "基础增亮"
        self.meta_chip.setText(f"Preset {preset_label} | Mode {mode_label} | Gamma {self.gamma_slider.value():.2f}")

    def on_mode_changed(self, *_):
        self.enhancement_mode = self.enhancement_mode_combo.currentData() or "basic"
        self.update_meta_summary()

    def collect_processing_payload(self, apply_adjustments=False, auto_recommend=False):
        return {
            "preset": self.current_preset,
            "enhance_mode": self.enhancement_mode,
            "gamma": f"{self.gamma_slider.value():.2f}",
            "brightness": f"{self.brightness_slider.value():.2f}",
            "contrast": f"{self.contrast_slider.value():.2f}",
            "saturation": f"{self.saturation_slider.value():.2f}",
            "sharpness": f"{self.sharpness_slider.value():.2f}",
            "warmth": f"{self.warmth_slider.value():.2f}",
            "denoise": "true",
            "apply_adjustments": "true" if apply_adjustments else "false",
            "auto_recommend": "true" if auto_recommend else "false",
            "output_format": "base64",
        }

    def apply_preset(self, preset_name, schedule=True):
        if preset_name not in self.presets:
            return

        preset = self.presets[preset_name]
        self.current_preset = preset_name

        if preset_name == "night_detail":
            self.enhancement_mode_combo.setCurrentIndex(1)
            self.enhancement_mode = "night"

        self.gamma_slider.set_value(int(round(preset["gamma"] * 100)))
        self.brightness_slider.set_value(int(round(preset["brightness"] * 100)))
        self.contrast_slider.set_value(int(round(preset["contrast"] * 100)))
        self.saturation_slider.set_value(int(round(preset["saturation"] * 100)))
        self.sharpness_slider.set_value(int(round(preset["sharpness"] * 100)))
        self.warmth_slider.set_value(int(round(preset["warmth"])))

        for key, button in self.preset_buttons.items():
            button.setChecked(key == preset_name)

        self.preset_desc.setText(preset["description"])
        self.update_meta_summary()
        if schedule:
            self.schedule_preview_refresh()

    def open_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择待增强图片",
            "",
            "Image Files (*.png *.jpg *.jpeg *.bmp)",
        )
        if not file_path:
            return

        try:
            image = Image.open(file_path).convert("RGB")
        except Exception as exc:
            QMessageBox.critical(self, "读取失败", f"无法打开图片：{exc}")
            return

        self.current_image_path = file_path
        self.original_image = image
        self.base_image = image.copy()
        self.preview_image = image.copy()
        self.selection_norm = None
        self.recommendation_payload = None

        self.original_card.set_pil_image(image)
        self.result_card.set_pil_image(image)
        self.original_card.canvas.clear_selection()

        self.file_chip.setText(os.path.basename(file_path))
        self.enhance_btn.setEnabled(True)
        self.recommend_btn.setEnabled(True)
        self.export_btn.setEnabled(True)
        self.reset_btn.setEnabled(True)
        self.region_btn.setEnabled(False)
        self.set_status("图片已载入", success=True)
        self.analysis_summary.setText(" ")
        for label in self.metric_labels.values():
            label.setText(label.text().split("：")[0] + "：-")
        self.region_status.setText("当前未选择局部增强区域。")
        self.region_detail.setText("尚未选择局部区域。")
        self.reset_single_progress()
        self.schedule_preview_refresh()

    def request_recommendation(self):
        if not self.current_image_path:
            QMessageBox.information(self, "提示", "请先导入一张图片。")
            return

        self.recommend_btn.setEnabled(False)
        self.set_status("分析场景中...", success=None)
        self.start_worker(
            name="recommend",
            endpoint="/recommend_params",
            file_paths=[self.current_image_path],
            data={},
            field_name="file",
            on_success=self.on_recommendation_ready,
        )

    def on_recommendation_ready(self, payload):
        recommendation = payload.get("recommendation", {})
        analysis = payload.get("analysis", {})
        parameters = recommendation.get("parameters", {})
        preset_name = recommendation.get("preset", "balanced")

        self.recommendation_payload = payload
        if preset_name in self.presets:
            self.apply_preset(preset_name, schedule=False)

        self.gamma_slider.set_value(int(round(parameters.get("gamma", self.gamma_slider.value()) * 100)))
        self.brightness_slider.set_value(int(round(parameters.get("brightness", self.brightness_slider.value()) * 100)))
        self.contrast_slider.set_value(int(round(parameters.get("contrast", self.contrast_slider.value()) * 100)))
        self.saturation_slider.set_value(int(round(parameters.get("saturation", self.saturation_slider.value()) * 100)))
        self.sharpness_slider.set_value(int(round(parameters.get("sharpness", self.sharpness_slider.value()) * 100)))
        self.warmth_slider.set_value(int(round(parameters.get("warmth", self.warmth_slider.value()))))

        self.analysis_summary.setText(recommendation.get("reason", "已完成场景分析。"))
        self.metric_labels["平均亮度"].setText(f"平均亮度：{analysis.get('brightness_mean', '-')}")
        self.metric_labels["亮度方差"].setText(f"亮度方差：{analysis.get('brightness_std', '-')}")
        self.metric_labels["暗部占比"].setText(f"暗部占比：{analysis.get('dark_ratio', '-')}")
        self.metric_labels["平均饱和度"].setText(f"平均饱和度：{analysis.get('saturation_mean', '-')}")
        preset_label = self.presets.get(preset_name, {}).get("label", preset_name)
        self.metric_labels["推荐结果"].setText(f"推荐结果：{preset_label} / Gamma {parameters.get('gamma', '-')}")

        self.recommend_btn.setEnabled(True)
        self.set_status("推荐完成", success=True)
        self.schedule_preview_refresh()

    def request_enhance(self):
        if not self.current_image_path:
            QMessageBox.information(self, "提示", "请先导入一张图片。")
            return

        self.enhance_btn.setEnabled(False)
        self.region_btn.setEnabled(False)
        self.start_single_progress("整图增强已开始，正在上传图片并等待结果...")
        self.set_status("整图增强中...", success=None)
        self.start_worker(
            name="enhance",
            endpoint="/enhance",
            file_paths=[self.current_image_path],
            data=self.collect_processing_payload(apply_adjustments=False, auto_recommend=False),
            field_name="file",
            on_success=self.on_single_enhance_ready,
        )

    def on_single_enhance_ready(self, payload):
        image_data = payload.get("enhanced_image")
        if not image_data:
            self.on_worker_error("enhance", "缺少图像数据。")
            return

        image = base64_to_pil(image_data)
        self.base_image = image
        self.preview_image = image
        self.recommendation_payload = payload
        self.enhance_btn.setEnabled(True)
        self.region_btn.setEnabled(self.selection_norm is not None)
        self.export_btn.setEnabled(True)
        self.finish_single_progress(success=True)
        self.set_status("整图增强完成", success=True)
        self.schedule_preview_refresh()

    def on_selection_changed(self, selection):
        self.selection_norm = selection
        self.region_btn.setEnabled(self.current_image_path is not None and selection is not None)

        if not selection:
            self.region_status.setText("当前未选择局部增强区域。")
            self.region_detail.setText("尚未选择局部区域。")
            return

        x, y, w, h = selection
        self.region_status.setText(f"已选择区域：x={x:.2f}, y={y:.2f}, w={w:.2f}, h={h:.2f}")
        self.region_detail.setText("局部增强会只处理这个区域，并自动做羽化过渡，避免边缘生硬。")

    def clear_region_selection(self):
        self.original_card.canvas.clear_selection()

    def request_region_enhance(self):
        if not self.current_image_path:
            QMessageBox.information(self, "提示", "请先导入一张图片。")
            return
        if self.selection_norm is None:
            QMessageBox.information(self, "提示", "请先在原图上拖拽选择一个区域。")
            return

        x, y, w, h = self.selection_norm
        payload = self.collect_processing_payload(apply_adjustments=False, auto_recommend=False)
        payload.update(
            {
                "region_x": f"{x:.6f}",
                "region_y": f"{y:.6f}",
                "region_w": f"{w:.6f}",
                "region_h": f"{h:.6f}",
                "feather_radius": "18",
            }
        )

        self.enhance_btn.setEnabled(False)
        self.region_btn.setEnabled(False)
        self.start_single_progress("局部增强已开始，正在处理所选区域并合成结果...")
        self.set_status("局部增强中...", success=None)
        self.start_worker(
            name="region",
            endpoint="/enhance_region",
            file_paths=[self.current_image_path],
            data=payload,
            field_name="file",
            on_success=self.on_region_enhance_ready,
        )

    def on_region_enhance_ready(self, payload):
        image_data = payload.get("enhanced_image")
        if not image_data:
            self.on_worker_error("region", "缺少图像数据。")
            return

        image = base64_to_pil(image_data)
        self.base_image = image
        self.preview_image = image
        self.enhance_btn.setEnabled(True)
        self.region_btn.setEnabled(self.selection_norm is not None)
        self.export_btn.setEnabled(True)
        self.finish_single_progress(success=True)
        self.set_status("局部增强完成", success=True)
        self.schedule_preview_refresh()

    def on_parameter_changed(self):
        self.update_meta_summary()
        self.schedule_preview_refresh()

    def schedule_preview_refresh(self):
        if self.base_image is None:
            return
        self.preview_timer.start()

    def refresh_preview(self):
        if self.base_image is None:
            return

        image = self.base_image.copy()
        image = ImageEnhance.Brightness(image).enhance(self.brightness_slider.value())
        image = ImageEnhance.Contrast(image).enhance(self.contrast_slider.value())
        image = ImageEnhance.Color(image).enhance(self.saturation_slider.value())
        image = ImageEnhance.Sharpness(image).enhance(self.sharpness_slider.value())
        image = self.apply_warmth(image, self.warmth_slider.value())

        self.preview_image = image
        self.result_card.set_pil_image(image)

    def apply_warmth(self, image, warmth_value):
        if abs(warmth_value) < 1e-6:
            return image

        array = Image.Image.convert(image, "RGB")
        rgb = list(array.getdata())
        adjusted = []
        scale = min(abs(warmth_value) / 30.0, 2.0)

        for red, green, blue in rgb:
            if warmth_value > 0:
                red += int(12 * scale)
                green += int(3 * scale)
                blue -= int(10 * scale)
            else:
                red -= int(8 * scale)
                green += int(2 * scale)
                blue += int(12 * scale)
            adjusted.append((max(0, min(255, red)), max(0, min(255, green)), max(0, min(255, blue))))

        warmed = Image.new("RGB", image.size)
        warmed.putdata(adjusted)
        return warmed

    def export_image(self):
        if self.preview_image is None:
            QMessageBox.information(self, "提示", "当前没有可导出的图片。")
            return

        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "保存增强结果",
            "enhanced_result.png",
            "PNG Image (*.png);;JPEG Image (*.jpg *.jpeg)",
        )
        if not save_path:
            return

        try:
            self.preview_image.save(save_path)
            self.set_status("结果已导出", success=True)
        except Exception as exc:
            QMessageBox.critical(self, "保存失败", f"无法保存图片：{exc}")

    def reset_parameters(self):
        self.apply_preset(self.current_preset, schedule=True)
        self.set_status("参数已重置", success=True)

    def select_batch_files(self):
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "选择批量增强图片",
            "",
            "Image Files (*.png *.jpg *.jpeg *.bmp)",
        )
        if not file_paths:
            return

        self.batch_file_paths = file_paths
        self.batch_list.clear()
        self.batch_results = []

        for path in file_paths:
            item = QListWidgetItem(f"{os.path.basename(path)}  ·  等待处理")
            self.batch_list.addItem(item)

        self.batch_summary.setText(f"已导入 {len(file_paths)} 张图片，当前将使用“{self.presets[self.current_preset]['label']}”预设。")
        self.batch_run_btn.setEnabled(True)
        self.batch_export_btn.setEnabled(False)
        self.batch_progress.setValue(0)
        self.batch_preview_card.clear("批量结果将在这里预览")
        self.batch_detail.setText("暂无批量结果。")
        self.tabs.setCurrentWidget(self.batch_tab)

    def clear_batch_files(self):
        self.batch_file_paths = []
        self.batch_results = []
        self.batch_list.clear()
        self.batch_preview_card.clear("批量结果将在这里预览")
        self.batch_detail.setText("暂无批量结果。")
        self.batch_summary.setText("尚未导入批量图片。")
        self.batch_progress.setValue(0)
        self.batch_run_btn.setEnabled(False)
        self.batch_export_btn.setEnabled(False)

    def run_batch_enhance(self):
        if not self.batch_file_paths:
            QMessageBox.information(self, "提示", "请先导入批量图片。")
            return

        self.batch_run_btn.setEnabled(False)
        self.batch_progress.setRange(0, 0)
        self.batch_summary.setText("正在批量处理中，请稍候...")
        self.set_status("批量处理中...", success=None)

        payload = self.collect_processing_payload(
            apply_adjustments=True,
            auto_recommend=self.batch_auto_checkbox.isChecked(),
        )
        self.start_worker(
            name="batch",
            endpoint="/enhance_batch",
            file_paths=self.batch_file_paths,
            data=payload,
            field_name="files",
            on_success=self.on_batch_ready,
        )

    def on_batch_ready(self, payload):
        self.batch_progress.setRange(0, 100)
        self.batch_progress.setValue(100)
        self.batch_run_btn.setEnabled(True)

        self.batch_results = payload.get("results", [])
        success_count = payload.get("success_count", 0)
        total = payload.get("total", len(self.batch_results))
        self.batch_summary.setText(f"批量处理完成：成功 {success_count} / {total}。")
        self.set_status("批量处理完成", success=True)

        self.batch_list.clear()
        for result in self.batch_results:
            if result.get("success"):
                preset_name = result.get("parameters", {}).get("preset", self.current_preset)
                preset_label = self.presets.get(preset_name, {}).get("label", preset_name)
                item = QListWidgetItem(f"{result['filename']}  ·  成功  ·  {preset_label}")
            else:
                item = QListWidgetItem(f"{result['filename']}  ·  失败  ·  {result.get('error', '未知错误')}")
            self.batch_list.addItem(item)

        self.batch_export_btn.setEnabled(success_count > 0)
        if self.batch_results:
            self.batch_list.setCurrentRow(0)

    def on_batch_item_selected(self, row):
        if row < 0 or row >= len(self.batch_results):
            return

        result = self.batch_results[row]
        if not result.get("success"):
            self.batch_preview_card.clear("该任务执行失败，无法预览结果。")
            self.batch_detail.setText(result.get("error", "未知错误"))
            return

        image = base64_to_pil(result["enhanced_image"])
        self.batch_preview_card.set_pil_image(image)

        params = result.get("parameters", {})
        analysis = result.get("analysis", {})
        recommendation = result.get("recommendation")
        details = [
            f"文件名：{result.get('filename', '-')}",
            f"预设：{self.presets.get(params.get('preset', ''), {}).get('label', params.get('preset', '-'))}",
            f"Gamma：{params.get('gamma', '-')}",
            f"亮度 / 对比度：{params.get('brightness', '-')} / {params.get('contrast', '-')}",
            f"饱和度 / 锐化：{params.get('saturation', '-')} / {params.get('sharpness', '-')}",
            f"平均亮度 / 暗部占比：{analysis.get('brightness_mean', '-')} / {analysis.get('dark_ratio', '-')}",
        ]
        if recommendation:
            details.append(f"推荐原因：{recommendation.get('reason', '-')}")
        self.batch_detail.setText("\n".join(details))

    def export_batch_results(self):
        if not self.batch_results:
            QMessageBox.information(self, "提示", "当前没有可导出的批量结果。")
            return

        target_dir = QFileDialog.getExistingDirectory(self, "选择批量导出目录")
        if not target_dir:
            return

        exported = 0
        for result in self.batch_results:
            if not result.get("success"):
                continue
            try:
                image = base64_to_pil(result["enhanced_image"])
                filename = os.path.splitext(result["filename"])[0] + "_enhanced.png"
                image.save(os.path.join(target_dir, filename))
                exported += 1
            except Exception:
                continue

        self.set_status("批量结果已导出", success=True)
        QMessageBox.information(self, "导出完成", f"成功导出 {exported} 张图片。")


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(244, 247, 255))
    palette.setColor(QPalette.Base, QColor(255, 255, 255))
    palette.setColor(QPalette.Button, QColor(255, 255, 255))
    app.setPalette(palette)

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
