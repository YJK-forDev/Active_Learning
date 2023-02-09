import os
import sys
import json
import argparse
import importlib
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from datetime import datetime
from torch.utils.data import DataLoader
import random

sys.path.append(os.path.join(os.getcwd())) # HACK add the root folder
from lib.solver import Solver
from lib.dataset import ScannetDataset, ScannetDatasetWholeScene, ScannetDataset_update_New_TrainData,collate_random, collate_wholescene
from lib.loss import WeightedCrossEntropyLoss
from lib.config import CONF
os.environ["PYTHONHASHSEED"] = str(1024)


def get_dataloader(args, scene_list, phase):

    if args.use_wholescene:
        dataset = ScannetDatasetWholeScene(scene_list, is_weighting=not args.no_weighting, use_color=args.use_color, use_normal=args.use_normal, use_multiview=args.use_multiview)
        dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_wholescene, num_workers=args.num_workers, pin_memory=True)
    else: # only use this version in our study
        dataset = ScannetDataset(phase, scene_list, is_weighting=not args.no_weighting, use_color=args.use_color, use_normal=args.use_normal, use_multiview=args.use_multiview)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_random, num_workers=args.num_workers, pin_memory=True)

    return dataset, dataloader

def get_num_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    num_params = int(sum([np.prod(p.size()) for p in model_parameters]))

    return num_params

def get_solver(args, dataset, dataloader, stamp, weight):
    
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../pointnet2/'))
    Pointnet = importlib.import_module("pointnet2_semseg")
    input_channels = int(args.use_color) * 3 + int(args.use_normal) * 3 + int(args.use_multiview) * 128
    
    model = Pointnet.get_model(num_classes=CONF.NUM_CLASSES, is_msg=args.use_msg, input_channels=input_channels, use_xyz=not args.no_xyz, bn=not args.no_bn).cuda()
    
    num_params = get_num_params(model)
    criterion = WeightedCrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    solver = Solver(model, dataset, dataloader, criterion, optimizer, args.batch_size, stamp, args.use_wholescene, args.ds, args.df, epoch=args.epoch)

    return solver, num_params

def get_scene_list(path):
    scene_list = []
    with open(path) as f:
        for scene_id in f.readlines():
            scene_list.append(scene_id.strip())
    
    return scene_list

    

def save_info(args, root, train_examples, val_examples, num_params):
    info = {}
    for key, value in vars(args).items():
        info[key] = value
    
    info["num_train"] = train_examples
    info["num_val"] = val_examples
    info["num_params"] = num_params

    with open(os.path.join(root, "info.json"), "w") as f:
        json.dump(info, f, indent=4)

def train(args): # 초기 학습 + active learning 수행함
    # init training dataset
    
    print("preparing data...")
    if args.debug:
        train_scene_list = ["scene0000_00"]
        val_scene_list = ["scene0000_00"]
    else:
        #train_scene_list, query_scene_list = get_scene_list(CONF.SCANNETV2_TRAIN, True) # train.txt에 저장되어 있는 scene들을 모두 가져옴, 2는 숫자 뭐 들어가도 상관없게 코딩해놓음
        # 아래의 val_scene_list는 val 경로에 있는 모든 scene들
        #print(train_scene_list)
        train_scene_list = get_scene_list(CONF.SCANNETV2_TRAIN)
        val_scene_list = get_scene_list(CONF.SCANNETV2_VAL)
        query_scene_list = get_scene_list(CONF.SCANNETV2_QUERY)
        
    torch_random_seed(random_seed=1024)
    # dataloader
    train_dataset, train_dataloader = get_dataloader(args, train_scene_list, "train")
    val_dataset, val_dataloader = get_dataloader(args, val_scene_list, "val")
    query_dataset, query_dataloader = get_dataloader(args, query_scene_list, "query")
    #test_dataset, test_dataloader = get_dataloader(args, query_scene_list, "test")
    dataset = {
        "train": train_dataset,
        "val": val_dataset,
        "query": query_dataset
        
    }
    dataloader = {
        "train": train_dataloader,
        "val": val_dataloader,
        "query": query_dataloader
        
    }
    weight = train_dataset.labelweights
    train_scene_num = len(train_dataset)
    val_scene_num = len(val_dataset)

    print("initializing...")
    stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if args.tag: stamp += "_"+args.tag.upper()+"_"+args.scene_number.upper()+"_"+args.method.upper()
    root = os.path.join(CONF.OUTPUT_ROOT, stamp)
    os.makedirs(root, exist_ok=True)
    solver, num_params = get_solver(args, dataset, dataloader, stamp, weight)
    
    print("\n[info]")
    print("Train examples: {}".format(train_scene_num))
    print("Evaluation examples: {}".format(val_scene_num))
    print("Start training...\n")
    save_info(args, root, train_scene_num, val_scene_num, num_params)
    solver(args, args.epoch, args.verbose, args.occlusion_round)

def torch_random_seed(random_seed = 11):
  torch.manual_seed(random_seed)
  torch.default_generator.manual_seed(random_seed)

  torch.cuda.manual_seed(random_seed)
  torch.cuda.manual_seed_all(random_seed) # if use multi-GPU

  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  torch.use_deterministic_algorithms(True)

  np.random.seed(random_seed)
  random.seed(random_seed)
  


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', type=str, help="tag for the participants", default="p1")
    parser.add_argument('--gpu', type=str, help='gpu', default='0')
    parser.add_argument('--batch_size', type=int, help='batch size', default=16)
    parser.add_argument('--epoch', type=int, help='number of epochs', default=500)
    parser.add_argument('--verbose', type=int, help='iterations of showing verbose', default=10)
    parser.add_argument('--num_workers', type=int, help='number of workers in dataloader', default=0)
    parser.add_argument('--lr', type=float, help='learning rate', default=1e-3)
    parser.add_argument('--wd', type=float, help='weight decay', default=0)
    parser.add_argument('--ds', type=int, help='decay step', default=100)
    parser.add_argument('--df', type=float, help='decay factor', default=0.7)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--no_weighting", action="store_true", help="weight the classes")
    parser.add_argument('--no_bn', action="store_true", help="do not apply batch normalization in pointnet++")
    parser.add_argument('--no_xyz', action="store_true", help="do not apply coordinates as features in pointnet++")
    parser.add_argument("--use_wholescene", action="store_true", help="on the whole scene or on a random chunk")
    parser.add_argument("--use_msg", action="store_true", help="apply multiscale grouping or not")
    parser.add_argument("--use_color", action="store_true", help="use color values or not")
    parser.add_argument("--use_normal", action="store_true", help="use normals or not")
    parser.add_argument("--use_multiview", action="store_true", help="use multiview image features or not")
    parser.add_argument('--occlusion_round', type=int, help='which point to save when round', default=1)
    parser.add_argument('--annotation_label', type=str, help='which label to annotate', default="chair")
    parser.add_argument('--scene_number', type=str, help='which scene to annotate', default="0011_00")
    parser.add_argument('--method', type=str, help='which method', default="method1")
    
    args = parser.parse_args()

    if args.annotation_label!="chair" and args.annotation_label!="table":
      raise Exception('annotation label은 chair 또는 table 이어야합니다.')

    # setting
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    torch_random_seed(random_seed=1024)
    train(args)