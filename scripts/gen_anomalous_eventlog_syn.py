import os
from pathlib import Path

from tqdm import tqdm

import sys
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.append(str(SCRIPT_DIR))
sys.path.append(str(PROJECT_DIR))

from dcad.generation.anomaly import *
from dcad.generation.utils import generate_for_process_model


def get_process_model_files(path=None):
    # Base
    ROOT_DIR = PROJECT_DIR / "data" / "original"

    if path is None:
        path = os.path.join(ROOT_DIR / 'synthetic')
    return [os.path.join(path,f) for f in os.listdir(path)]


anomalies = [
    SkipSequenceAnomaly(max_sequence_size=2),##最多跳过的子序列个数
    ReworkAnomaly(max_distance=5, max_sequence_size=5),##重做的和原来的最大距离，重做的最多的子序列个数
    EarlyAnomaly(max_distance=5, max_sequence_size=5),
    LateAnomaly(max_distance=5, max_sequence_size=5),
    InsertAnomaly(max_inserts=5)
]


process_models = [m for m in get_process_model_files()]

if __name__ == "__main__":
    process_model = str(PROJECT_DIR / "data" / "original" / "synthetic" / "wide.plg")
    generate_for_process_model(process_model, size=5000, anomalies=anomalies, anomaly_p=0, num_attr=[1], seed=1337)
    # for process_model in tqdm(process_models, desc='Generate'):
    #     generate_for_process_model(process_model, size=5000, anomalies=anomalies, anomaly_p=0, num_attr=[1], seed=1337)
