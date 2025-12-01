import os
import tarfile
from tqdm import tqdm

filelist=[]

root="./checkpoints_all_cls/MNCDV2/"

models=os.listdir(root)

best_models=[root + model + "/BestF1/" for model in models]

with tarfile.open("checkpoints_uni.tar.gz", "w:gz") as tar:
    for source_dir in tqdm(best_models):
        tar.add(source_dir, arcname=source_dir.split("./")[-1])