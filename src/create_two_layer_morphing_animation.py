from create_morphing_animation import draw_morphing
from visualize.plotly_vis.camera import PredefinedCameraParameters

from pathlib import Path

import hydra
from omegaconf import DictConfig
from tqdm import tqdm

from mesh.mesh import OrganMesh
from visualize.organ import draw_organ_3d, draw_organ_diff_3d, draw_organ_diff_rgb_3d
from visualize.plotly_vis.camera import PredefinedCameraParameters
from visualize.plotly_vis.figure2PILImage import figure2PILImage
from visualize.color import ColorPalette
import plotly.graph_objects as go
import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageFont
from logging import getLogger
import matplotlib.pyplot as plt
from io import BytesIO

log = getLogger(__name__)

BBOX = tuple[int, int, int, int] # (left, top, right, bottom)

@hydra.main(
    version_base="1.1",
    config_path="conf",
    config_name="create_two_layer_morphing_animation",
)
def main(cfg: DictConfig):
    title = "morphing"
    data_dir = Path(cfg.data_dir)
    frame_skip_rate_num = cfg.frame_skip_rate_num
    save_file_name = cfg.save_file_name
    camera_param = PredefinedCameraParameters[cfg.camera_param]
    if cfg.mode == "separate":
        draw_morphing(
            title=title,
            data_dir=data_dir,
            frame_skip_rate_num=frame_skip_rate_num,
            save_file_name=f"layer_1=half_{save_file_name}.gif",
            camera_param=camera_param,
            file_prefix="layer_1_half_layer2",
            separate=cfg.separate,
            axis_off=cfg.axis_off,
            color_difference=cfg.color_difference,
            enhance_difference=cfg.enhance_difference,
            display_mean_organ=cfg.display_mean_organ,
            show_axis_rgb_color=cfg.show_axis_rgb_color,
        )
        draw_morphing(
            title=title,
            data_dir=data_dir,
            frame_skip_rate_num=frame_skip_rate_num,
            save_file_name=f"layer_2=half_{save_file_name}.gif",
            camera_param=camera_param,
            file_prefix="layer_2_half_layer1",
            separate=cfg.separate,
            axis_off=cfg.axis_off,
            color_difference=cfg.color_difference,
            enhance_difference=cfg.enhance_difference,
            display_mean_organ=cfg.display_mean_organ,
            show_axis_rgb_color=cfg.show_axis_rgb_color,
        )
        # draw_morphing(
        #     title=title,
        #     data_dir=data_dir,
        #     frame_skip_rate_num=frame_skip_rate_num,
        #     save_file_name=f"both_{save_file_name}.gif",
        #     camera_param=camera_param,
        #     file_prefix="layer_1_and_layer_2",
        #     separate=cfg.separate,
        #     axis_off=cfg.axis_off,
        #     color_difference=cfg.color_difference,
        #     enhance_difference=cfg.enhance_difference,
        #                 display_mean_organ=cfg.display_mean_organ,
        # )
    elif cfg.mode == "grid":
        draw_grid_morphing(
            data_dir=data_dir,
            camera_param=camera_param,
            axis_off=cfg.axis_off,
            color_difference=cfg.color_difference,
            enhance_difference=cfg.enhance_difference,
            display_mean_organ=cfg.display_mean_organ,
            show_axis_rgb_color=cfg.show_axis_rgb_color,
            draw_text=cfg.draw_text,
            adjust_center=cfg.adjust_center,
        )
        draw_grid_morphing(
            data_dir=data_dir,
            camera_param=PredefinedCameraParameters.coronal,
            axis_off=cfg.axis_off,
            color_difference=cfg.color_difference,
            enhance_difference=cfg.enhance_difference,
            display_mean_organ=cfg.display_mean_organ,
            show_axis_rgb_color=cfg.show_axis_rgb_color,
            draw_text=cfg.draw_text,
            adjust_center=cfg.adjust_center,
        )
        draw_grid_morphing(
            data_dir=data_dir,
            camera_param=PredefinedCameraParameters.sagittal,
            axis_off=cfg.axis_off,
            color_difference=cfg.color_difference,
            enhance_difference=cfg.enhance_difference,
            display_mean_organ=cfg.display_mean_organ,
            show_axis_rgb_color=cfg.show_axis_rgb_color,
            draw_text=cfg.draw_text,
            adjust_center=cfg.adjust_center,
        )


def draw_grid_morphing(
    data_dir: Path,
    camera_param: PredefinedCameraParameters,
    axis_off: bool = True,
    color_difference: bool = False,
    enhance_difference: bool = False,
    crop_buffer_size: int = 50,
    display_mean_organ: bool = False,
    show_axis_rgb_color: bool = False,
    draw_text: bool = False,
    adjust_center: bool = False,
):
    """grid morphing を描画する関数.
    plyファイルのファイル名は、"{i}_{j}.ply"とすること.

    Args:
        title (str): 描画されるタイトル
        data_dir (Path): .plyファイルが保存されているディレクトリ
        frame_skip_rate_num (int): フレーム間隔
        save_file_name (str): 保存するファイルネーム
        camera_param (PredefinedCameraParameter): 画像のカメラパラメーター
        separate (bool): モーフィング結果を別々に保存するか、1枚のgifにするか
        axis_off (bool): 軸を表示しないかどうか
        color_difference (bool): モーフィングにおいて中間メッシュとの差分を色付けするかどうか
    """
    files = list(data_dir.glob("*.ply"))
    file_idxs = [int(f.stem.split("_")[-1]) for f in files]
    mesh_num = max(file_idxs) + 1
    mean_mesh_idx = mesh_num // 2  # 中間メッシュのインデックス
    mean_mesh = OrganMesh(mesh_path=data_dir / f"{mean_mesh_idx}_{mean_mesh_idx}.ply")
    
    grid_ims = [[None for _ in range(max(file_idxs) + 1)] for _ in range(max(file_idxs) + 1)]

    for i in tqdm(range(mesh_num)):
        # z_2: i
        for j in tqdm(range(mesh_num)):
            # z_1: j
            mesh_path = data_dir / f"{i}_{j}.ply"
            mesh = OrganMesh(mesh_path=mesh_path)
            plotly_fig = go.Figure()
            if color_difference:
                if show_axis_rgb_color and not (i == mean_mesh_idx and j == mean_mesh_idx):
                    draw_organ_diff_rgb_3d(
                        fig=plotly_fig,
                        source_organ=mean_mesh,
                        target_organ=mesh,
                        enhance_differece=enhance_difference,
                        display_source_organ=display_mean_organ,
                        adjust_center=adjust_center
                    )
                else:
                    draw_organ_diff_3d(
                        fig=plotly_fig,
                        source_organ=mean_mesh,
                        target_organ=mesh,
                        cmax=30,
                        cmin=0,
                        cmax_color=ColorPalette.red.value.rgb_plotly_str,
                        cmin_color=ColorPalette.gray.value.rgb_plotly_str,
                        enhance_differece=enhance_difference,
                        display_source_organ=display_mean_organ
                    )
            else:
                draw_organ_3d(
                    fig=plotly_fig, organ=mesh, color=ColorPalette.cyan.value.rgb_plotly_str
                )
            plotly_fig.update_layout(scene_camera=camera_param.value)
            if axis_off:
                plotly_fig.update_layout(scene=dict(xaxis=dict(visible=False)))
                plotly_fig.update_layout(scene=dict(yaxis=dict(visible=False)))
                plotly_fig.update_layout(scene=dict(zaxis=dict(visible=False)))
            if camera_param == PredefinedCameraParameters.coronal:
                plotly_fig.update_layout(
                    scene=dict(
                        xaxis=dict(autorange="reversed"),
                    )
                )
            plotly_fig.update_layout(showlegend=False)
            im = figure2PILImage(figure=plotly_fig)
            
            grid_ims[i][j] = im
    
    # 1. 计算所有图片的 tight bbox
    bboxes = []
    for row in grid_ims:
        for im in row:
            inv_img = ImageOps.invert(im.convert("RGB"))
            bbox = inv_img.getbbox()
            bboxes.append(bbox)
    # 2. 取最大范围
    left = min(b[0] for b in bboxes)
    top = min(b[1] for b in bboxes)
    right = max(b[2] for b in bboxes)
    bottom = max(b[3] for b in bboxes)
    max_bbox = (left, top, right, bottom)
    # 3. 用最大 bbox crop 所有图片
    for i, row in enumerate(grid_ims):
        for j, im in enumerate(row):
            grid_ims[i][j] = im.crop(max_bbox)
    grid_im = np.concatenate([np.concatenate(row, axis=1) for row in grid_ims], axis=0)
    grid_im = Image.fromarray(grid_im)
    grid_im.save(f"{camera_param.name}_gird.png")   
            

def left_bbox_is_larger(left_bbox: BBOX, right_bbox: BBOX):
    left_width = left_bbox[2] - left_bbox[0]
    left_height = left_bbox[3] - left_bbox[1]
    right_width = right_bbox[2] - right_bbox[0]
    right_height = right_bbox[3] - right_bbox[1]
    return left_width > right_width and left_height > right_height

def buffer_bbox(bbox: BBOX, buffer_size: int, max_size: int = 256): 
    return (
        max(bbox[0] - buffer_size, 0),
        max(bbox[1] - buffer_size, 0),
        min(bbox[2] + buffer_size, max_size),
        min(bbox[3] + buffer_size, max_size),
    )

def adjust_target_bbox(target_bbox: BBOX, source_bbox: BBOX, max_size: int = 256):
    target_center = (
        (target_bbox[0] + target_bbox[2]) // 2,
        (target_bbox[1] + target_bbox[3]) // 2,
    )
    source_center = (
        (source_bbox[0] + source_bbox[2]) // 2,
        (source_bbox[1] + source_bbox[3]) // 2,
    )
    dx, dy = target_center[0] - source_center[0], target_center[1] - source_center[1]
    return (
        max(target_bbox[0] + dx, 0),
        max(target_bbox[1] + dy, 0),
        min(target_bbox[2] + dx, max_size),
        min(target_bbox[3] + dy, max_size),
    )

    
        
    
        




if __name__ == "__main__":
    main()
