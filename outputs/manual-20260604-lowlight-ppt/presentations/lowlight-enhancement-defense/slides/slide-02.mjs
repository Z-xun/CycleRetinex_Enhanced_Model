import { C, bg, title, footer, text, card, node, pill, hbar } from "./common.mjs";

export async function slide02(presentation, ctx) {
  const slide = presentation.slides.add();
  bg(slide, ctx);
  title(slide, ctx, "01 主要技术", "技术路线围绕“可解释分解 + 学习式增强”展开", "模型负责恢复光照与细节，工程层负责把推理能力变成可操作的应用。");

  card(slide, ctx, 70, 180, 330, 420, { fill: "#FFFFFF" });
  text(slide, ctx, "算法基础", 96, 206, 180, 28, { size: 22, bold: true, color: C.navy });
  hbar(slide, ctx, 96, 244, 80, C.teal);
  node(slide, ctx, "Retinex 分解", "图像 S 被建模为反射率 R 与光照 L 的乘积：S = R × L。", 96, 274, 250, 78, C.teal, C.paleTeal);
  node(slide, ctx, "曲线光照增强", "LCNet 学习非线性增强强度，优先提升暗部并控制亮部过曝。", 96, 368, 250, 78, C.orange, C.paleOrange);
  node(slide, ctx, "反射率去噪", "U-Net 通过编码器、解码器和跳连保留纹理，同时抑制噪声。", 96, 462, 250, 78, C.blue, C.paleBlue);

  card(slide, ctx, 430, 180, 420, 420, { fill: "#FFFFFF" });
  text(slide, ctx, "核心模型", 456, 206, 180, 28, { size: 22, bold: true, color: C.navy });
  hbar(slide, ctx, 456, 244, 80, C.orange);
  const xs = [470, 598, 726];
  const labels = ["Decomposition", "LCNet", "UNetDenoise"];
  const details = ["R / L", "L → L'", "R → R'"];
  const colors = [C.teal, C.orange, C.blue];
  for (let i = 0; i < 3; i += 1) {
    card(slide, ctx, xs[i], 304, 108, 108, { fill: i === 1 ? C.paleOrange : i === 0 ? C.paleTeal : C.paleBlue, stroke: colors[i], strokeWidth: 1.5 });
    text(slide, ctx, labels[i], xs[i] + 8, 322, 92, 24, { size: 12, color: colors[i], bold: true, align: "center" });
    text(slide, ctx, details[i], xs[i] + 8, 362, 92, 24, { size: 18, color: C.ink, bold: true, align: "center" });
    if (i < 2) text(slide, ctx, "→", xs[i] + 112, 340, 22, 34, { size: 24, color: C.mute, bold: true, align: "center" });
  }
  text(slide, ctx, "PatchGAN 判别器", 492, 466, 300, 28, { size: 18, color: C.red, bold: true, align: "center" });
  text(slide, ctx, "约束局部亮度与纹理分布，让增强结果更接近真实图像。", 492, 500, 300, 44, { size: 14, color: C.mute, align: "center" });

  card(slide, ctx, 880, 180, 330, 420, { fill: "#FFFFFF" });
  text(slide, ctx, "工程实现", 906, 206, 180, 28, { size: 22, bold: true, color: C.navy });
  hbar(slide, ctx, 906, 244, 80, C.blue);
  pill(slide, ctx, "Python", 908, 285, 102, C.navy, "#F8FAFC");
  pill(slide, ctx, "PyTorch", 1024, 285, 112, C.teal, "#F8FAFC");
  pill(slide, ctx, "Flask", 908, 330, 102, C.orange, "#F8FAFC");
  pill(slide, ctx, "PyQt5", 1024, 330, 112, C.blue, "#F8FAFC");
  pill(slide, ctx, "PIL / OpenCV", 908, 375, 228, C.mute, "#F8FAFC");
  text(slide, ctx, "训练：LOL-v2、Night_data\n推理：best_model.pth / best_model_v2.pth\n部署：本地 HTTP 服务 + 桌面端交互", 908, 442, 250, 100, {
    size: 15,
    color: C.ink,
  });

  footer(slide, ctx, 2);
  return slide;
}
