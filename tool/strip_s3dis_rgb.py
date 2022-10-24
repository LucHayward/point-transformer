from pathlib import Path
import numpy as np
from tqdm import tqdm

for path in tqdm(Path("/home/eco02/Luc/point-transformer/dataset/s3dis/trainval_fullarea_xyz").glob("*.npy")):
    pcd = np.load(path)
    np.save(path, pcd[:, [0,1,2,-1]])  # xyzrgbl -> xyzl

print("Done.")
