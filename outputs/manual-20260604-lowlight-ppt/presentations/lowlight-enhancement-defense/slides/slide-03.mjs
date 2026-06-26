import { C, bg, title, footer, text, card, hbar, node } from "./common.mjs";

export async function slide03(presentation, ctx) {
  const slide = presentation.slides.add();
  bg(slide, ctx);
  title(slide, ctx, "02 功能框架", "系统采用前后端分离，把模型能力封装成可交互工具", "用户只接触导入、调参、预览和导出，后端统一处理模型推理与结果封装。");

  const laneY = 188;
  const laneH = 390;
  const lanes = [
    ["界面层 PyQt5", "单图增强\n局部框选\n批量任务\n结果预览 / 导出", 70, C.blue, C.paleBlue],
    ["服务层 Flask", "/health\n/presets\n/recommend_params\n/enhance\n/enhance_region\n/enhance_batch", 350, C.teal, C.paleTeal],
    ["模型层 PyTorch", "模型加载\n尺寸规整\nRetinex 推理\nGamma / 后处理", 630, C.orange, C.paleOrange],
    ["数据与输出", "LOL-v2 / Night_data\n图片 / 视频帧\nBase64 / 文件保存\nPSNR / SSIM 曲线", 910, C.navy, "#EEF2F6"],
  ];

  for (const [head, body, x, color, fill] of lanes) {
    card(slide, ctx, x, laneY, 230, laneH, { fill: "#FFFFFF" });
    text(slide, ctx, head, x + 22, laneY + 24, 190, 28, { size: 20, bold: true, color });
    hbar(slide, ctx, x + 22, laneY + 64, 70, color);
    text(slide, ctx, body, x + 26, laneY + 102, 180, 188, { size: 18, color: C.ink });
    card(slide, ctx, x + 26, laneY + 314, 178, 38, { fill, stroke: color, strokeWidth: 1 });
    text(slide, ctx, "高内聚，低耦合", x + 36, laneY + 324, 158, 20, { size: 13, color, bold: true, align: "center" });
  }

  text(slide, ctx, "HTTP", 308, 345, 34, 20, { size: 12, color: C.mute, align: "center", bold: true });
  text(slide, ctx, "调用", 588, 345, 34, 20, { size: 12, color: C.mute, align: "center", bold: true });
  text(slide, ctx, "返回", 868, 345, 34, 20, { size: 12, color: C.mute, align: "center", bold: true });
  text(slide, ctx, "→", 314, 370, 22, 40, { size: 26, color: C.mute, align: "center", bold: true });
  text(slide, ctx, "→", 594, 370, 22, 40, { size: 26, color: C.mute, align: "center", bold: true });
  text(slide, ctx, "→", 874, 370, 22, 40, { size: 26, color: C.mute, align: "center", bold: true });

  card(slide, ctx, 250, 590, 780, 64, { fill: "#FFFFFF", stroke: C.teal, strokeWidth: 1.5 });
  text(slide, ctx, "局部增强补充流程", 272, 608, 170, 24, { size: 17, color: C.teal, bold: true });
  text(slide, ctx, "前端记录归一化选区坐标，后端换算像素区域、裁剪增强，再用羽化掩膜贴回原图。", 455, 608, 520, 32, {
    size: 13,
    color: C.ink,
  });

  footer(slide, ctx, 3);
  return slide;
}
