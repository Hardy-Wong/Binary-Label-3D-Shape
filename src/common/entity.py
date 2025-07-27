from enum import Enum
from typing import Union


class FeatureName(Enum):
    DGandPosition = "DGandPosition"
    Position = "Position"


class OrganName(Enum):
    liver = "liver"
    duodenum = "duodenum"
    left_kidney = "left_kidney"
    right_kidney = "right_kidney"
    stomach = "stomach"


class ModelName(Enum):
    mesh_VAE = "mesh_VAE"
    absolute_position_and_dg_mesh_linear_VAE = (
        "absolute_position_and_dg_mesh_linear_VAE"
    )
    absolute_position_mesh_linear_VAE = "absolute_position_mesh_linear_VAE"


class CaseID:
    def __init__(self, case_id: Union[int, str]) -> None:
        if type(case_id) is int:
            if not (1 <= case_id <= 146):
                raise ValueError(f"invalid case id {case_id}")
            self.case_id = case_id
        elif type(case_id) is str:
            assert len(case_id) == 7
            assert case_id[:4] == "case"
            self.case_id = int(case_id[4:])
        else:
            raise ValueError("case_id type is invalid")

    def __repr__(self) -> str:
        return self.value

    @property
    def value(self):
        case_id = str(self.case_id).rjust(3, "0")
        return f"case{case_id}"

    @property
    def int_only_id(self):
        return str(self.case_id).rjust(3, "0")
