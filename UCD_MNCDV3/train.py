import torch
import model
import yaml
from utils import argparser
from accelerate import utils
from codecarbon import track_emissions

@track_emissions(allow_multiple_runs=True)
def main(configs):
    CD_framework=model.Change_Detection_Framework(config=configs)
    CD_framework.training_CD()

if __name__=="__main__":
    utils.set_seed(8888)

    args=argparser.get_argparser().parse_args()

    with open(args.config,'r') as f:
        configs=yaml.safe_load(f)
        if args.dataset_path is not None:
            configs["dataset_path"]=args.dataset_path 
        print(configs)
    
    main(configs)
