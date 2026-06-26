export const C = {
  ink: "#17202A",
  mute: "#637083",
  bg: "#F7F9FC",
  panel: "#FFFFFF",
  navy: "#12324A",
  teal: "#2AAE9F",
  orange: "#F28C28",
  blue: "#4C7CF3",
  red: "#E25B5B",
  line: "#D8E0EA",
  paleTeal: "#E8F6F4",
  paleOrange: "#FFF1E1",
  paleBlue: "#EEF3FF",
  dark: "#0D1B2A",
};

export const projectRoot = "D:/Code/code_conda/Undergraduate Thesis";

export function bg(slide, ctx, color = C.bg) {
  ctx.addShape(slide, { x: 0, y: 0, width: ctx.W, height: ctx.H, fill: color });
}

export function text(slide, ctx, value, x, y, width, height, opts = {}) {
  return ctx.addText(slide, {
    text: value,
    x,
    y,
    width,
    height,
    fontSize: opts.size ?? 24,
    color: opts.color ?? C.ink,
    bold: opts.bold ?? false,
    typeface: opts.face ?? "Microsoft YaHei UI",
    align: opts.align ?? "left",
    valign: opts.valign ?? "top",
    fill: opts.fill ?? "#00000000",
    line: opts.line ?? ctx.line("#00000000", 0),
    insets: opts.insets ?? { left: 0, right: 0, top: 0, bottom: 0 },
  });
}

export function title(slide, ctx, kicker, headline, sub = "") {
  text(slide, ctx, kicker, 70, 38, 560, 26, { size: 15, color: C.teal, bold: true });
  text(slide, ctx, headline, 70, 70, 900, 46, { size: 34, color: C.ink, bold: true });
  if (sub) text(slide, ctx, sub, 72, 134, 980, 28, { size: 16, color: C.mute });
}

export function footer(slide, ctx, page, note = "基于 Retinex 分解与深度学习的低照度图像增强系统") {
  ctx.addShape(slide, { x: 70, y: 675, width: 1050, height: 1, fill: C.line });
  text(slide, ctx, note, 70, 684, 650, 18, { size: 10, color: "#8792A1" });
  text(slide, ctx, String(page).padStart(2, "0"), 1150, 680, 60, 24, {
    size: 12,
    color: "#8792A1",
    align: "right",
  });
}

export function card(slide, ctx, x, y, width, height, opts = {}) {
  return ctx.addShape(slide, {
    x,
    y,
    width,
    height,
    fill: opts.fill ?? C.panel,
    line: opts.line ?? ctx.line(opts.stroke ?? "#E3E9F2", opts.strokeWidth ?? 1),
  });
}

export function pill(slide, ctx, label, x, y, width, color, fill = "#FFFFFF") {
  ctx.addShape(slide, {
    x,
    y,
    width,
    height: 28,
    fill,
    line: ctx.line(color, 1.4),
  });
  text(slide, ctx, label, x + 12, y + 5, width - 24, 18, {
    size: 11,
    color,
    bold: true,
    align: "center",
  });
}

export function node(slide, ctx, label, detail, x, y, width, height, color, fill) {
  card(slide, ctx, x, y, width, height, { fill, stroke: color, strokeWidth: 1.5 });
  text(slide, ctx, label, x + 16, y + 14, width - 32, 26, { size: 17, color, bold: true });
  text(slide, ctx, detail, x + 16, y + 46, width - 32, height - 60, { size: 12, color: C.mute });
}

export function hbar(slide, ctx, x, y, width, color) {
  ctx.addShape(slide, { x, y, width, height: 3, fill: color });
}

export async function image(slide, ctx, relPath, x, y, width, height, fit = "cover") {
  return ctx.addImage(slide, {
    path: `${projectRoot}/${relPath}`,
    x,
    y,
    width,
    height,
    fit,
  });
}

export function bullet(slide, ctx, label, x, y, width, color = C.teal) {
  ctx.addShape(slide, { geometry: "ellipse", x, y: y + 7, width: 8, height: 8, fill: color });
  text(slide, ctx, label, x + 18, y, width - 18, 34, { size: 15, color: C.ink });
}

export function tinyLabel(slide, ctx, label, x, y, width, color = C.mute) {
  text(slide, ctx, label, x, y, width, 18, { size: 10.5, color, align: "center", bold: true });
}
