import { C, bg, title, footer, text, card, hbar, image, bullet } from "./common.mjs";

export async function slide05(presentation, ctx) {
  const slide = presentation.slides.add();
  bg(slide, ctx);
  title(slide, ctx, "04 训练与优化", "联合损失把亮度、结构、真实感同时约束住", "训练脚本记录损失曲线与 PSNR / SSIM，用于选择最佳 checkpoint。");

  card(slide, ctx, 70, 182, 480, 430, { fill: "#FFFFFF" });
  text(slide, ctx, "训练配置", 98, 210, 160, 28, { size: 22, color: C.navy, bold: true });
  hbar(slide, ctx, 98, 248, 80, C.teal);
  bullet(slide, ctx, "数据：LOL-v2 Real_captured 与 Night_data", 100, 288, 390, C.teal);
  bullet(slide, ctx, "网络：Decomposition、L2H、H2L、UNetDenoise、3个判别器", 100, 332, 390, C.orange);
  bullet(slide, ctx, "优化器：Adam，生成器学习率 1e-4，判别器学习率 5e-5", 100, 388, 390, C.blue);
  bullet(slide, ctx, "评价：PSNR、SSIM，以及二者组合 score", 100, 444, 390, C.navy);
  bullet(slide, ctx, "输出：checkpoint、对比图、损失曲线、指标曲线", 100, 488, 390, C.red);

  card(slide, ctx, 590, 182, 590, 430, { fill: "#FFFFFF" });
  text(slide, ctx, "损失函数组合", 618, 210, 190, 28, { size: 22, color: C.navy, bold: true });
  hbar(slide, ctx, 618, 248, 80, C.orange);
  const losses = [
    ["重建损失", "保证 R × L 可还原输入图像", C.teal],
    ["循环一致性", "L2H 与 H2L 双向约束亮度转换", C.orange],
    ["光照先验 / 平滑", "让光照图符合自然照明变化", C.blue],
    ["反射率约束", "保持纹理与颜色结构稳定", C.navy],
    ["对抗损失", "提升局部亮度和纹理真实感", C.red],
  ];
  for (let i = 0; i < losses.length; i += 1) {
    const y = 282 + i * 48;
    ctx.addShape(slide, { geometry: "ellipse", x: 620, y: y + 7, width: 16, height: 16, fill: losses[i][2] });
    text(slide, ctx, losses[i][0], 650, y, 140, 24, { size: 17, color: C.ink, bold: true });
    text(slide, ctx, losses[i][1], 790, y + 2, 330, 24, { size: 15, color: C.mute });
  }

  await image(slide, ctx, "night_loss_img/generator_loss.png", 792, 526, 160, 58, "contain");
  await image(slide, ctx, "night_loss_img/score.png", 970, 526, 160, 58, "contain");
  text(slide, ctx, "训练过程可视化", 792, 586, 340, 18, { size: 11, color: C.mute, align: "center" });

  footer(slide, ctx, 5);
  return slide;
}
