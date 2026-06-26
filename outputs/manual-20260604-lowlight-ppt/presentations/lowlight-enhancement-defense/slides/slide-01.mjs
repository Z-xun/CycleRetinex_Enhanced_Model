import { C, bg, text, pill, image } from "./common.mjs";

export async function slide01(presentation, ctx) {
  const slide = presentation.slides.add();
  bg(slide, ctx, C.dark);

  ctx.addShape(slide, { x: 0, y: 0, width: 1280, height: 720, fill: "#0D1B2A" });
  ctx.addShape(slide, { x: 0, y: 0, width: 1280, height: 720, fill: "linear(35deg, #0D1B2A 0%, #133B4A 55%, #1F2937 100%)" });
  ctx.addShape(slide, { x: 70, y: 56, width: 4, height: 88, fill: C.orange });

  text(slide, ctx, "低照度图像增强系统", 90, 54, 560, 72, {
    size: 44,
    color: "#FFFFFF",
    bold: true,
  });
  text(slide, ctx, "基于 Retinex 分解与深度学习的设计与实现", 92, 126, 540, 34, {
    size: 20,
    color: "#D7E5F0",
  });
  text(slide, ctx, "5分钟答辩版 | 主要技术、功能框架、关键算法流程", 92, 178, 540, 28, {
    size: 15,
    color: "#9FB7C8",
  });

  pill(slide, ctx, "PyTorch", 92, 236, 108, C.teal, "#0F2A36");
  pill(slide, ctx, "Retinex", 214, 236, 112, C.orange, "#2C2518");
  pill(slide, ctx, "Flask + PyQt5", 340, 236, 150, "#9FC8FF", "#102640");

  text(slide, ctx, "项目目标", 92, 312, 160, 28, { size: 20, color: "#FFFFFF", bold: true });
  text(
    slide,
    ctx,
    "把夜间、弱光和逆光图像中的暗部信息恢复出来，同时抑制亮度增强带来的噪声放大，并封装成可交互的桌面应用。",
    92,
    352,
    520,
    90,
    { size: 19, color: "#DCE8F2" },
  );

  await image(slide, ctx, "img_low_max.jpg", 690, 92, 230, 300, "cover");
  await image(slide, ctx, "high_enh.jpg", 940, 92, 230, 300, "cover");
  text(slide, ctx, "输入：低照度图像", 690, 410, 230, 24, { size: 14, color: "#C9D8E5", align: "center" });
  text(slide, ctx, "输出：增强图像", 940, 410, 230, 24, { size: 14, color: "#C9D8E5", align: "center" });
  text(slide, ctx, "→", 918, 212, 18, 42, { size: 28, color: C.orange, bold: true, align: "center" });

  ctx.addShape(slide, { x: 92, y: 596, width: 1080, height: 1, fill: "#355267" });
  text(slide, ctx, "汇报路径：问题背景  →  技术路线  →  系统框架  →  算法流程  →  训练优化  →  总结展望", 92, 620, 980, 24, {
    size: 15,
    color: "#C7D7E6",
  });
  return slide;
}
