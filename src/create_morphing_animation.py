from pathlib import Path

import hydra
from omegaconf import DictConfig
from tqdm import tqdm
from typing import Optional

from mesh.mesh import OrganMesh
from visualize.organ import draw_organ_3d, draw_organ_diff_3d, draw_organ_diff_rgb_3d
from visualize.plotly_vis.camera import PredefinedCameraParameters
from visualize.plotly_vis.figure2PILImage import figure2PILImage
from visualize.color import ColorPalette
import plotly.graph_objects as go
import numpy as np
from PIL import Image, ImageOps


@hydra.main(
    version_base="1.1",
    config_path="conf",
    config_name="create_morphing_animation",
)
def main(cfg: DictConfig):
    title = cfg.title
    data_dir = Path(cfg.data_dir)
    frame_skip_rate_num = cfg.frame_skip_rate_num
    save_file_name = cfg.save_file_name
    draw_morphing(
        title=title,
        data_dir=data_dir,
        frame_skip_rate_num=frame_skip_rate_num,
        save_file_name=f"{save_file_name}.gif",
        camera_param=PredefinedCameraParameters.axial,
        separate=cfg.separate,
        axis_off=cfg.axis_off,
        color_difference=cfg.color_difference,
        enhance_difference=cfg.enhance_difference,
    )


def draw_morphing(
    title: str,
    data_dir: Path,
    frame_skip_rate_num: int,
    save_file_name: str,
    camera_param: PredefinedCameraParameters,
    file_prefix: Optional[str] = None,
    extension: str = "ply",
    separate: bool = False,
    axis_off: bool = True,
    color_difference: bool = False,
    enhance_difference: bool = False,
    display_mean_organ: bool = False,
    show_axis_rgb_color: bool = False,
):
    """morphing を描画する関数

    Args:
        title (str): 描画されるタイトル
        data_dir (Path): .plyファイルが保存されているディレクトリ
        frame_skip_rate_num (int): フレーム間隔
        save_file_name (str): 保存するファイルネーム
        camera_param (PredefinedCameraParameter): 画像のカメラパラメーター
        separate (bool): モーフィング結果を別々に保存するか、1枚のgifにするか
        axis_off (bool): 軸を表示しないかどうか
        color_difference (bool): モーフィングにおいて中間メッシュとの差分を色付けするかどうか
        show_axis_rgb_color (bool): モーフィングにおいて中間メッシュとの差分を色付けする際に、軸の色を表示するかどうか
    """
    # Figureを追加
    ims = []
    file_exp = f"{file_prefix}_*" if file_prefix is not None else "*"
    file_exp = f"{file_exp}.{extension}"
    files = list(data_dir.glob(file_exp))
    file_idxs = [int(f.stem.split("_")[-1]) for f in files]
    sorted_files = [x for _, x in sorted(zip(file_idxs, files))]
    mean_mesh_idx = len(sorted_files) // 2  # 中間メッシュのインデックス
    mean_mesh = OrganMesh(mesh_path=sorted_files[mean_mesh_idx])

    for i, mesh_path in tqdm(enumerate(sorted_files)):
        if i % frame_skip_rate_num != 0:
            continue
        mesh = OrganMesh(mesh_path=mesh_path)
        plotly_fig = go.Figure()
        if color_difference:
            if show_axis_rgb_color:
                draw_organ_diff_rgb_3d(
                    fig=plotly_fig,
                    source_organ=mean_mesh,
                    target_organ=mesh,
                    enhance_differece=enhance_difference,
                    display_source_organ=display_mean_organ,
                    adjust_center=True
                )
            else:
                draw_organ_diff_3d(
                    fig=plotly_fig,
                    source_organ=mean_mesh,
                    target_organ=mesh,
                    cmax=0.5,
                    cmin=0,
                    cmax_color=ColorPalette.red.value.rgb_plotly_str,
                    cmin_color=ColorPalette.blue.value.rgb_plotly_str,
                    enhance_differece=enhance_difference,
                    display_source_organ=display_mean_organ,
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
        im = ImageOps.crop(im, border=150)
        ims.append(np.asarray(im))
    if separate:
        for i, im in enumerate(ims):
            img = Image.fromarray(im)
            inv_img = ImageOps.invert(img.convert("RGB"))
            img = img.crop(inv_img.getbbox())
            img.save(f"{save_file_name}_{i}.png")
        return
    ims = np.stack(ims)
    ims = [Image.fromarray(im) for im in ims]
    # Converting it to RGB to ensure that it has 3 dimensions as requested
    ims[0].save(save_file_name, save_all=True, append_images=ims[1:])


if __name__ == "__main__":
    main()
