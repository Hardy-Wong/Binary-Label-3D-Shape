import re
from pathlib import Path
from typing import List

from .entity import CaseID


def load_case_ids(mode: str = "train") -> List[CaseID]:
    data_path = Path(__file__).parent / "data_split"
    if mode == "train":
        with open(data_path / "train_case_ids.txt") as f:
            idxs = f.read().split(" ")
    elif mode == "val":
        with open(data_path / "val_case_ids.txt") as f:
            idxs = f.read().split(" ")
    elif mode == "test":
        with open(data_path / "test_case_ids.txt") as f:
            idxs = f.read().split(" ")
    else:
        raise ValueError("mode should be one of the train, val, test")

    case_ids = []
    for idx in idxs:
        if idx == "":
            continue
        case_id = CaseID(int(re.sub(r"\D", "", idx)))
        case_ids.append(case_id)
    return case_ids
