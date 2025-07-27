from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, TypedDict

import hydra
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig
from sklearn.decomposition import PCA

COLORS = {
    "liver": "blue",
    "stomach": "red",
    "left_kidney": "green",
    "right_kidney": "brown",
}


class LatentInfo(TypedDict):
    organ: str
    case_id: str
    color: str
    marker: str
    decom: np.ndarray


def load_data(
    data_dirs: List[str],
) -> Tuple[Dict[str, np.ndarray], Dict[str, List[str]]]:
    """潜在変数の分布の可視化をするためにデータをロードする関数。

    Args:
        data_dirs (List[str]): 潜在変数の.npyファイルが含まれるフォルダ。

    Returns:
        Tuple[Dict[str, np.ndarray], Dict[str, List[str]]]: concatされた潜在変数とcaseIDの配列が帰ります。
    """
    vecs = defaultdict(list)
    case_ids = defaultdict(list)
    for data_dir in data_dirs:
        data_dir = Path(data_dir)
        for latent_path in list(data_dir.glob("*.npy")):
            case_id, organ, _, _ = latent_path.stem.split("_")  # TODO: 別関数に切り出す
            latent_vec = np.load(str(latent_path))
            vecs[organ].append(latent_vec)
            case_ids[organ].append(case_id)
    return vecs, case_ids


def create_morphing_distribution(
    fig,
    ax,
    start_latent_info: LatentInfo,
    end_latent_info: LatentInfo,
    interpolate_num: int,
    save_file_name: str,
    frame_skip_rate_num: int,
    interpolate_point_color: str = "purple",
):
    ims = []
    start_point, end_point = start_latent_info["decom"], end_latent_info["decom"]
    for i in range(interpolate_num):
        if i % frame_skip_rate_num != 0:
            continue
        point = start_point * (1.0 - i / interpolate_num) + end_point * (
            i / interpolate_num
        )
        im = ax.plot(
            point[0],
            point[1],
            color=interpolate_point_color,
            marker=start_latent_info["marker"],
        )
        ims.append(im)
    ax.legend()

    ani = animation.ArtistAnimation(
        fig, ims, interval=100, blit=True, repeat_delay=1000
    )
    frame_num = len(ims)
    print(frame_num)
    ani.save(
        save_file_name,
        writer="pillow",
        progress_callback=lambda i, n: print(f"{str(i/frame_num)[:4]} done"),
    )


@hydra.main(
    version_base="1.1",
    config_path="/work/src/conf",
    config_name="create_latent_distribution",
)
def main(cfg: DictConfig):

    # Figureを追加
    fig = plt.figure(figsize=(8, 8))
    # 3DAxesを追加
    ax = fig.add_subplot(111)
    # Axesのタイトルを設定
    ax.set_title("distribution", size=20)
    # 軸ラベルを設定
    ax.set_xlabel("1 axis", size=14)
    ax.set_ylabel("2 axis", size=14)

    data, organ_case_ids = load_data(cfg.data_dirs)
    all_data = np.concatenate(
        [np.concatenate(ndarr) for ndarr in data.values()], axis=1
    )
    decomp = PCA(n_components=2)
    decomp.fit(all_data)

    for organ, latent_vecs in data.items():
        decom_latent = decomp.transform(np.concatenate(latent_vecs))
        color = COLORS[organ]
        ax.scatter(decom_latent[:, 0], decom_latent[:, 1], color=color, label=organ)
        if cfg.annotate_with_case_id:
            for i, organ_case_id in enumerate(organ_case_ids[organ]):
                ax.annotate(organ_case_id, decom_latent[i])

    if cfg.morphing_animation:

        start_case_id = cfg.morphing_start_case_id
        end_case_id = cfg.morphing_end_case_id
        case_ids = organ_case_ids[organ]
        start_case_idx = case_ids.index(start_case_id)
        end_case_idx = case_ids.index(end_case_id)
        start_decom_latent = decom_latent[start_case_idx]
        end_decom_latent = decom_latent[end_case_idx]

        start_latent_info = LatentInfo(
            organ=organ,
            case_id=start_case_id,
            color="red",
            marker="o",
            decom=start_decom_latent,
        )

        end_latent_info = LatentInfo(
            organ=organ,
            case_id=end_case_id,
            color="red",
            marker="o",
            decom=end_decom_latent,
        )

        create_morphing_distribution(
            fig,
            ax,
            start_latent_info,
            end_latent_info,
            frame_skip_rate_num=cfg.frame_skip_rate_num,
            interpolate_num=cfg.interpolate_num,
            save_file_name="morphing_latent_distribution.gif",
        )
    else:
        ax.legend()
        fig.savefig("test.pdf")


if __name__ == "__main__":
    main()
