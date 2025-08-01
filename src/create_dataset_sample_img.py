from pathlib import Path

import hydra
from omegaconf import DictConfig

from common.entity import CaseID, OrganName
from visualize.color import VisualizeColorRGB, ColorPalette
from visualize.plotly_vis.camera import PredefinedCameraParameters
from visualize.plotly_vis.figure2PILImage import figure2PILImage
from visualize.plotly_vis.graph_objects import mesh_body , mesh_edge
from visualize.visualize import transparent_background
from mesh.mesh import OrganMesh
import  plotly.graph_objects as go
from PIL import ImageOps


@hydra.main(version_base="1.1", config_path="conf", config_name="create_dataset_sample_img")
def main(cfg: DictConfig):
    dataset_dir = Path(cfg.dataset_dir)
    case_ids = [
        CaseID(case_id) for case_id in cfg.case_ids        
    ]
    show_organs = [
        OrganName[organ_name] for organ_name in cfg.show_organs
    ]
    camera_parameters = [
        PredefinedCameraParameters[parameter_name] for parameter_name in cfg.camera_parameters
    ]
    for  camera_parameter in camera_parameters:
        for  case_id in case_ids:
            fig = go.Figure()
            for organ_name in show_organs:
                fig = create_img(
                    fig=fig,
                    dataset_dir=dataset_dir,
                    case_id=case_id,
                    organ_name=organ_name
                )
            fig.update_layout(
                scene_camera=camera_parameter.value,
                showlegend=False,
                # scene=dict(xaxis=dict(visible=False),yaxis=dict(visible=False),zaxis=dict(visible=False))
            )
            if camera_parameter.name == PredefinedCameraParameters.coronal.name:
                fig.update_layout(
                    scene=dict(
                        xaxis=dict(autorange="reversed"),
                    )
                )
            img = figure2PILImage(fig)
            img = transparent_background(img)
            img = ImageOps.crop(img,border=180)
            
            img.save(f"{case_id.value}_{camera_parameter.name}.png")
                

        

def draw_organ(
    fig: go.Figure, organ_mesh: OrganMesh, color:VisualizeColorRGB 
) -> None:
    """ draw organ mesh
    """
    colored_target_organ = mesh_body(
        mesh=organ_mesh,
        color=color.rgba_plotly_str(0.3) 
    )
    colored_target_organ_edge = mesh_edge(
        visible=True,
        mesh=organ_mesh,
        color=ColorPalette.black.value.rgba_plotly_str(0.3) 
    )
    fig.add_trace(colored_target_organ)
    fig.add_trace(colored_target_organ_edge)
    return fig

OrganColorMap = {
    OrganName.liver.value:ColorPalette.red.value,
    OrganName.stomach.value: ColorPalette.blue.value
}
    

def create_img(fig: go.Figure ,dataset_dir: Path, case_id: CaseID, organ_name: OrganName):
    mesh_path = dataset_dir /case_id.value / "00/output" / f"{organ_name.value}.ply"
    mesh = OrganMesh(mesh_path)
    draw_organ(
        fig=fig,
        organ_mesh=mesh,
        color=OrganColorMap[organ_name.value]
    )
    return fig


if __name__ == "__main__":
    main()
