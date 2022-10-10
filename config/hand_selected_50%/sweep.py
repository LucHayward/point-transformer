import os
print(os.getcwd())

from pathlib import Path
os.chdir(Path(__file__).parent)
print(os.getcwd())

import wandb
import yaml

shell_script = Path("./tool/train.sh")
dataset = "hand_selected_50%"
exp = None

base_config = Path("hand_selected_50%_pt.yaml")

with open(base_config, "r") as f:
    base_config = yaml.safe_load(f)

voxel_size = [0.04, 0.02]
voxel_max = [x * 5000 for x in range(1, 9)]

scheduler = [None, 'warmup']
# If scheduler == warmup
power = [0.3, 0.6, 0.9]
warmup_length = [10, 25, 50]

freeze_body = [True, False]

batch_size = 3

base_lr = [0.3, 0.5, 0.7]
weight_decay = [0.01, 0.001, 0.0001]

weight = [None, "exp/hand_selected_50%/pretrained/model/s3dis_2_class_head.pth"]

sweep_config = {
    "method": "random",
    "name": "sweep",
    "metric": {
        "name": "mIoU_val",
        "goal": "maximize"
    },
    "parameters": {
        "voxel_size": {
            "values": voxel_size
        },
        "voxel_max": {
            "values": voxel_max
        },
        "scheduler": {
            "values": scheduler
        },
        "warmup_length": {
            "values": warmup_length
        },
        "power": {
            "values": power
        },
        "freeze_body": {
            "values": freeze_body
        },
        "base_lr": {
            "values": base_lr
        },
        "weight_decay": {
            "values": weight_decay
        },
        "weight": {
            "values": weight
        }
    }
}

# yaml.safe_dump(sweep_config, open("sweep.yaml", "w"))

# sweep_id = wandb.sweep(sweep_config, project="point-transformer")


