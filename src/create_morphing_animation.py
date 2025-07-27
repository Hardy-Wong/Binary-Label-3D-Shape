from pathlib import Path

import hydra
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from omegaconf import DictConfig
from tqdm import tqdm

from src.mesh.mesh import OrganMesh
from src.visualize.organ import draw_organ


@hydra.main(
    version_base="1.1",
    config_path="/work/src/conf",
    config_name="create_morphing_animation",
)
def main(cfg: DictConfig):
    title = cfg.title
    data_dir = Path(cfg.data_dir)
    frame_skip_rate_num = cfg.frame_skip_rate_num
    save_file_name = cfg.save_file_name
    # elev_azim_lists = [(0, 0), (45, 45), (90, 0), (90, 180), (0, 90), (0, 180), (0, -90), (0, 270)]
    elev_azim_lists = [(0, 0)]
    for elev, azim in elev_azim_lists:
        draw_morphing(
            title=title,
            data_dir=data_dir,
            frame_skip_rate_num=frame_skip_rate_num,
            save_file_name=f"elev={elev},azim={azim},{save_file_name}.gif",
            elev=elev,
            azim=azim,
        )


def draw_morphing(
    title: str,
    data_dir: Path,
    frame_skip_rate_num: int,
    save_file_name: str,
    elev: int,
    azim: int,
):
    """morphing を描画する関数

    Args:
        title (str): 描画されるタイトル
        data_dir (Path): .plyファイルが保存されているディレクトリ
        frame_skip_rate_num (int): フレーム間隔
        save_file_name (str): 保存するファイルネーム
        elev (int): 3dで確認する角度(緯度(deg))
        azim (int): 3dで確認する角度(経度(deg))
    """
    # Figureを追加
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title, size=20)
    ax.set_xlabel("x", size=14)
    ax.set_ylabel("y", size=14)
    ax.set_zlabel("z", size=14)
    ax.set_ylim(100, -80)
    ax.set_xlim(75, -150)
    ax.set_zlim(130, -50)
    if ax.xaxis_inverted:
        ax.invert_xaxis()
    if ax.yaxis_inverted:
        ax.invert_yaxis()
    if ax.zaxis_inverted:
        ax.invert_zaxis()
    ims = []
    for i, mesh_path in tqdm(
        enumerate(sorted(data_dir.glob("*.ply"), key=lambda f: int(f.stem)))
    ):
        if i % frame_skip_rate_num != 0:
            continue
        mesh = OrganMesh(mesh_path=mesh_path)
        title = ax.text(
            x=0.5,
            y=1.01,
            z=0.0,
            s=f"{i}th organ",
            ha="center",
            va="bottom",
            transform=ax.transAxes,
            fontsize="large",
        )
        im = draw_organ(fig, ax, mesh, color="blue", linestyle="", marker="o")
        ims.append(im + [title])
    ani = animation.ArtistAnimation(
        fig, ims, interval=100, blit=True, repeat_delay=1000
    )
    frame_num = len(ims)
    ani.save(
        save_file_name,
        writer="pillow",
        progress_callback=lambda i, n: print(f"{str(i/frame_num)[:4]} done"),
    )


if __name__ == "__main__":
    main()
