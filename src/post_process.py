import logging
import hydra
from omegaconf import DictConfig
from pathlib import Path
from visualize.pandas_vis.load_data import load_data

from visualize.pandas_vis.mean_vertex_error import (
    cal_mean_vertex_error_groupby_case_id,
)

from train_scripts.post_processing import plot_metrics, plot_samples

# from metrics.agg import ave_distances, Metric
from mesh.mesh import OrganMesh
from common.entity import OrganName,CaseID

log = logging.getLogger(__name__)


@hydra.main(
    version_base="1.1", config_path="conf", config_name="post_process"
)
def main(cfg: DictConfig) -> None:
    post_process(
        log_vertex_position_df_dir=Path(cfg.log_vertex_position_df_dir),
        commit_hash=cfg.commit_hash,
        template_mesh_path=cfg.template_mesh_path,
        axis_off=cfg.axis_off,
        will_plot_samples=cfg.will_plot_samples,
        will_plot_metrics=cfg.will_plot_metrics,
        concatenate=cfg.concatenate,
        cmax=cfg.cmax,
        organ_name=OrganName[cfg.target_organ],
        plot_sample_case_ids=[CaseID(case_id) for case_id in cfg.plot_sample_case_ids] if hasattr(cfg, "plot_sample_case_ids") else None
    )
    pass


def post_process(
    log_vertex_position_df_dir: Path,
    commit_hash: str,
    template_mesh_path: Path,
    organ_name: OrganName,
    axis_off: bool = False,
    will_plot_metrics: bool = True,
    will_plot_samples: bool = True,
    concatenate: bool = False,
    cmax: int = 20, 
    plot_sample_case_ids: list[CaseID] = None,
) -> None:
    """後処理用のスクリプト郡。

    Args:
        log_vertex_position_df_dir (Path): 頂点位置が保存されているデータフレームのパス。
        logged_df_dir (Path): ログされたメッシュのdfのパス。
        commit_hash (str): コミットハッシュ。
        template_mesh_path (Path): テンプレートメッシュのパス。
        axis_off (bool, optional): 画像を出力する際に軸を入れるか入れないか. Defaults to False.
        concatenate (bool, optional): 画像を出力する際に、画像を結合するかどうか. Defaults to False.
        cmax (int, optional): カラーマップで表示する誤差の最大値。 Defaults to 20 
    """
    # visualize data according to the logged dataframe data while training.
    collected_data = load_data(data_path=log_vertex_position_df_dir)
    template_mesh = OrganMesh(template_mesh_path)

    if will_plot_metrics:
        # 後処理1 エポックごとの誤差を可視化。
        hd_md_metrics = plot_metrics(
            collected_data=collected_data,
            experiment_id=commit_hash,
            template_mesh=template_mesh,
        )
        hd_md_metrics.to_csv(f"{commit_hash}_raw_hd_md.csv")
        mean_vertex_error_groupby_case_id = cal_mean_vertex_error_groupby_case_id(
            collected_data
        )
        log.info("experiment detail...")
        log.info(
            f"\n          vertex_distance\n{mean_vertex_error_groupby_case_id.groupby(['epoch', 'mode']).describe()}"
        )
        log.info(
            f"\n          hd_md\n{hd_md_metrics.groupby(['epoch', 'mode']).describe()}"
        )
        hd_md_metrics.groupby(["epoch", "mode"]).describe().to_csv(
            f"{commit_hash}_result_summary_hd_md.csv"
        )
        mean_vertex_error_groupby_case_id.to_csv(
            f"{commit_hash}_raw_vertex_distance.csv"
        )
        mean_vertex_error_groupby_case_id.groupby(["epoch", "mode"]).describe().to_csv(
            f"{commit_hash}_result_summary.csv"
        )

    if will_plot_samples:
        # 後処理2 最終エポックの中で最良、普通、最悪に関してそれぞれ3枚ずつ描画する。
        plot_samples(
            collected_data=collected_data,
            logged_df_dir=log_vertex_position_df_dir,
            experiment_id=commit_hash,
            axis_off=axis_off,
            concat=concatenate,
            template_mesh_path=template_mesh_path,
            cmax=cmax,
            organ_name=organ_name,
            case_ids=plot_sample_case_ids
        )


if __name__ == "__main__":
    main()
