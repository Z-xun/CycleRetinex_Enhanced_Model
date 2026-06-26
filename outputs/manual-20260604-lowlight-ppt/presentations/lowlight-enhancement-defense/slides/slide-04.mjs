import { C, bg, title, footer, text, card, hbar, node, image } from "./common.mjs";

export async function slide04(presentation, ctx) {
  const slide = presentation.slides.add();
  bg(slide, ctx);
  title(slide, ctx, "03 关键算法流程", "Retinex 分解让增强过程更可解释", "光照分量负责亮度提升，反射率分量负责颜色与纹理，再重建为最终图像。");

  await image(slide, ctx, "L_low.jpg", 78, 204, 140, 105, "cover");
  await image(slide, ctx, "R_low.jpg", 78, 422, 140, 105, "cover");
  text(slide, ctx, "光照 L", 78, 316, 140, 20, { size: 12, color: C.orange, bold: true, align: "center" });
  text(slide, ctx, "反射率 R", 78, 534, 140, 20, { size: 12, color: C.teal, bold: true, align: "center" });

  node(slide, ctx, "输入 S", "低照度 RGB 图像", 260, 322, 118, 80, C.navy, "#FFFFFF");
  text(slide, ctx, "→", 390, 338, 34, 34, { size: 26, color: C.mute, bold: true, align: "center" });
  node(slide, ctx, "Decomposition", "拼接 RGB 与最大通道图，输出 R 和 L", 430, 300, 172, 124, C.teal, C.paleTeal);

  text(slide, ctx, "↗", 612, 274, 44, 44, { size: 32, color: C.mute, align: "center", bold: true });
  text(slide, ctx, "↘", 612, 426, 44, 44, { size: 32, color: C.mute, align: "center", bold: true });
  node(slide, ctx, "LCNet", "L' = L + αL(1-L)\n迭代 3 次，提升暗部亮度", 670, 210, 190, 108, C.orange, C.paleOrange);
  node(slide, ctx, "U-Net 去噪", "R' = R + 0.1F(R)\n保留纹理，抑制增强噪声", 670, 430, 190, 108, C.blue, C.paleBlue);

  text(slide, ctx, "↘", 874, 312, 44, 44, { size: 32, color: C.mute, align: "center", bold: true });
  text(slide, ctx, "↗", 874, 404, 44, 44, { size: 32, color: C.mute, align: "center", bold: true });
  node(slide, ctx, "重建 S'", "S' = R' × L'\n再执行 Gamma 与色彩后处理", 930, 322, 180, 112, C.navy, "#FFFFFF");

  card(slide, ctx, 72, 578, 1080, 64, { fill: "#FFFFFF" });
  hbar(slide, ctx, 92, 602, 72, C.orange);
  text(slide, ctx, "讲解抓手", 182, 594, 100, 26, { size: 17, color: C.navy, bold: true });
  text(slide, ctx, "先说明 S=R×L 的物理含义，再讲两条分支：LCNet 管光照，U-Net 管噪声，最后相乘重建。", 292, 596, 760, 28, {
    size: 17,
    color: C.ink,
  });

  footer(slide, ctx, 4);
  return slide;
}
