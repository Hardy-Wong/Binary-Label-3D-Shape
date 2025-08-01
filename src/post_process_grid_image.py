import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
from matplotlib.patches import FancyArrowPatch
from matplotlib.collections import LineCollection
import matplotlib as mpl
mpl.rcParams['mathtext.fontset'] = 'stix'


def draw_gradient_arrow(ax, start, end, color, n=100, label=None, label_offset=(0,0), label_color=None):
    # start, end: (x, y)
    # color: (r,g,b) 终点色，起点为黑
    x = np.linspace(start[0], end[0], n)
    y = np.linspace(start[1], end[1], n)
    # 渐变色
    colors = np.linspace([0,0,0], color, n)
    points = np.array([x, y]).T.reshape(-1,1,2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, colors=colors, linewidths=4, zorder=10)
    ax.add_collection(lc)
    # 箭头
    ax.annotate('', xy=end, xytext=(x[-2], y[-2]), arrowprops=dict(arrowstyle='->', color=color, lw=4), zorder=11)
    # 标签
    if label:
        ax.text(end[0]+label_offset[0], end[1]+label_offset[1], label, color=label_color or color, fontsize=30, ha='center', va='center', zorder=12)

def add_labels_and_xyz_colorbar(
    grid_img_path,
    out_path=None,
    case_left="Case132",
    case_right="Case049",
    z1_label="Latent Variables $z_1$",
    z2_label="Latent Variables $z_2$",
    xyz_tick=20,
    pad=120,
    xyz_legend_size=180
):
    # 读取grid图像
    grid_im = Image.open(grid_img_path)
    grid_w, grid_h = grid_im.size
    # 新建大画布
    fig_w = grid_w + pad * 2 + xyz_legend_size
    fig_h = grid_h + pad * 2 + xyz_legend_size
    dpi = 100
    fig = plt.figure(figsize=(fig_w/dpi, fig_h/dpi), dpi=dpi)
    ax = plt.gca()
    ax.axis('off')
    # 画底图
    ax.imshow(np.ones((fig_h, fig_w, 3)), extent=[0, fig_w, fig_h, 0])  # 白底
    ax.imshow(grid_im, extent=[pad, pad+grid_w, pad, pad+grid_h])
    # 上方双向箭头和z1标签（修正位置）
    arrow_y = pad+grid_h+50  # 距离画布顶部40像素
    label_y = pad+grid_h+80  # 距离画布顶部10像素
    ax.annotate('', xy=(pad+grid_w, arrow_y), xytext=(pad, arrow_y),
                arrowprops=dict(arrowstyle="<->", lw=2), zorder=20)
    ax.text(pad-10, arrow_y, case_left, fontsize=50, ha='right', va='center')
    #ax.text(pad+grid_w+10, arrow_y, case_right, fontsize=18, ha='left', va='center')
    ax.text(pad+grid_w/2, label_y, z1_label, fontsize=50, ha='center', va='bottom')
    # 右侧双向箭头和z2标签
    arrow_x = pad+grid_w+50
    ax.annotate('', xy=(arrow_x, pad), xytext=(arrow_x, pad+grid_h),
                arrowprops=dict(arrowstyle="<->", lw=2), zorder=20)
    ax.text(arrow_x, pad-10, case_right, fontsize=50, ha='center', va='top')
    #ax.text(arrow_x, pad-10, case_left, fontsize=18, ha='center', va='bottom')
    ax.text(arrow_x+35, pad+grid_h/2, z2_label, fontsize=50, ha='left', va='center', rotation=90)
    # 右下角渐变xyz colorbar
    # 坐标系原点
    cb_x0 = 30
    cb_y0 = 30
    L = 80
    # x轴：红
    draw_gradient_arrow(ax, (cb_x0, cb_y0), (cb_x0+L, cb_y0), color=(1,0,0), label=f'x\n{xyz_tick}mm', label_offset=(90,0), label_color='red')
    # y轴：绿
    draw_gradient_arrow(ax, (cb_x0, cb_y0), (cb_x0-L*0.7, cb_y0-L*0.7), color=(0,1,0), label=f'y\n{xyz_tick}mm', label_offset=(-100,20), label_color='green')
    # z轴：蓝
    draw_gradient_arrow(ax, (cb_x0, cb_y0), (cb_x0, cb_y0+L), color=(0,0,1), label=f'z\n{xyz_tick}mm', label_offset=(0,60), label_color='blue')
    # 原点黑点
    ax.scatter([cb_x0], [cb_y0], color='k', s=30, zorder=15)
    # xyz标签
    # 已在箭头终点加
    # 保存
    if out_path is None:
        base, ext = os.path.splitext(grid_img_path)
        out_path = base + '_final.png'
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0.1, dpi=dpi)
    plt.close(fig)
    print(f"Saved: {out_path}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('grid_img_path', help='Path to grid image (png)')
    parser.add_argument('--out', default=None, help='Output path')
    parser.add_argument('--case_left', default='Case132')
    parser.add_argument('--case_right', default='Case049')
    parser.add_argument('--z1_label', default='Latent Variables $z_1$')
    parser.add_argument('--z2_label', default='Latent Variables $z_2$')
    args = parser.parse_args()
    add_labels_and_xyz_colorbar(
        args.grid_img_path,
        out_path=args.out,
        case_left=args.case_left,
        case_right=args.case_right,
        z1_label=args.z1_label,
        z2_label=args.z2_label
    ) 