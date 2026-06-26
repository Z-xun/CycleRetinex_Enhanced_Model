import { C, bg, title, footer, text, card, hbar } from "./common.mjs";

export async function slide06(presentation, ctx) {
  const slide = presentation.slides.add();
  bg(slide, ctx);
  title(slide, ctx, "05 展示节奏与总结", "5分钟内按“问题、方法、系统、算法、结果”收束", "答辩时突出核心贡献，细节问题再展开代码与公式。");

  card(slide, ctx, 70, 180, 510, 430, { fill: "#FFFFFF" });
  text(slide, ctx, "展示时间分配", 98, 208, 190, 28, { size: 22, color: C.navy, bold: true });
  hbar(slide, ctx, 98, 246, 80, C.teal);
  const timeline = [
    ["0:00-0:35", "背景与目标", C.navy],
    ["0:35-1:20", "主要技术路线", C.teal],
    ["1:20-2:05", "系统功能框架", C.blue],
    ["2:05-3:15", "关键算法流程", C.orange],
    ["3:15-4:10", "训练与优化", C.red],
    ["4:10-4:50", "效果、问题与展望", C.navy],
  ];
  for (let i = 0; i < timeline.length; i += 1) {
    const y = 286 + i * 48;
    ctx.addShape(slide, { x: 104, y: y + 10, width: 14, height: 14, fill: timeline[i][2] });
    text(slide, ctx, timeline[i][0], 134, y, 104, 24, { size: 15, color: timeline[i][2], bold: true });
    text(slide, ctx, timeline[i][1], 254, y, 240, 24, { size: 17, color: C.ink });
  }

  card(slide, ctx, 620, 180, 560, 430, { fill: "#FFFFFF" });
  text(slide, ctx, "总结陈述", 648, 208, 160, 28, { size: 22, color: C.navy, bold: true });
  hbar(slide, ctx, 648, 246, 80, C.orange);
  text(slide, ctx, "已完成", 650, 292, 100, 24, { size: 18, color: C.teal, bold: true });
  text(slide, ctx, "图像增强模型、Flask 后端、PyQt5 桌面端、单图/局部/批量/视频增强流程。", 760, 292, 340, 54, { size: 17, color: C.ink });
  text(slide, ctx, "核心价值", 650, 374, 100, 24, { size: 18, color: C.orange, bold: true });
  text(slide, ctx, "通过 Retinex 分解提高可解释性，通过 LCNet 与 U-Net 分工处理亮度和噪声。", 760, 374, 340, 54, { size: 17, color: C.ink });
  text(slide, ctx, "后续方向", 650, 456, 100, 24, { size: 18, color: C.blue, bold: true });
  text(slide, ctx, "扩充复杂夜间数据，引入时序一致性网络，进一步压缩模型以适配移动端或嵌入式部署。", 760, 456, 340, 70, { size: 17, color: C.ink });

  card(slide, ctx, 648, 548, 456, 36, { fill: C.paleOrange, stroke: C.orange, strokeWidth: 1 });
  text(slide, ctx, "结束语：系统已经形成从训练、推理到交互展示的完整闭环。", 664, 557, 424, 18, {
    size: 14,
    color: C.navy,
    bold: true,
    align: "center",
  });

  footer(slide, ctx, 6);
  return slide;
}
