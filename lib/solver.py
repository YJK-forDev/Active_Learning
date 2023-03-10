import os
import time

import torch
import numpy as np
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import StepLR
#from google.colab import files
import sys
import random
sys.path.append(".")
from tensorboardX import SummaryWriter
from lib.utils import decode_eta
from lib.config import CONF
from scripts.eval import compute_acc, compute_miou
from torch.utils.data import DataLoader
from lib.dataset import ScannetDataset_update_New_TrainData, ScannetDataset, ScannetDatasetWholeScene, collate_random, collate_wholescene

ITER_REPORT_TEMPLATE = """
----------------------iter: [{global_iter_id}/{total_iter}]----------------------
[loss] train_loss: {train_loss}
[sco.] train_point_acc: {train_point_acc}
[sco.] train_point_acc_per_class: {train_point_acc_per_class}
[sco.] train_voxel_acc: {train_voxel_acc}
[sco.] train_voxel_acc_per_class: {train_voxel_acc_per_class}
[sco.] train_point_miou: {train_point_miou}
[sco.] train_voxel_miou: {train_voxel_miou}
[info] mean_fetch_time: {mean_fetch_time}s
[info] mean_forward_time: {mean_forward_time}s
[info] mean_backward_time: {mean_backward_time}s
[info] mean_iter_time: {mean_iter_time}s
[info] ETA: {eta_h}h {eta_m}m {eta_s}s
"""

EPOCH_REPORT_TEMPLATE = """
------------------------summary------------------------
[train] train_loss: {train_loss}
[train] train_point_acc: {train_point_acc}
[train] train_point_acc_per_class: {train_point_acc_per_class}
[train] train_voxel_acc: {train_voxel_acc}
[train] train_voxel_acc_per_class: {train_voxel_acc_per_class}
[train] train_point_miou: {train_point_miou}
[train] train_voxel_miou: {train_voxel_miou}
[val]   val_loss: {val_loss}
[val]   val_point_acc: {val_point_acc}
[val]   val_point_acc_per_class: {val_point_acc_per_class}
[val]   val_voxel_acc: {val_voxel_acc}
[val]   val_voxel_acc_per_class: {val_voxel_acc_per_class}
[val]   val_point_miou: {val_point_miou}
[val]   val_voxel_miou: {val_voxel_miou}
"""

BEST_REPORT_TEMPLATE = """
-----------------------------best-----------------------------
[best] epoch: {epoch}
[loss] loss: {loss}
[sco.] point_acc: {point_acc}
[sco.] point_acc_per_class: {point_acc_per_class}
[sco.] voxel_acc: {voxel_acc}
[sco.] voxel_acc_per_class: {voxel_acc_per_class}
[sco.] point_miou: {point_miou}
[sco.] voxel_miou: {voxel_miou}
"""

total_list = []
confidence_points = []


def torch_random_seed(random_seed = 11):
  torch.manual_seed(random_seed)

  torch.cuda.manual_seed(random_seed)
  torch.cuda.manual_seed_all(random_seed) # if use multi-GPU

  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  torch.use_deterministic_algorithms(True)

  np.random.seed(random_seed)
  random.seed(random_seed)



class Solver():
    # ??? ??????????????? criterion?????? ???????????? ??? ????????? cross-entropy??? ????????????.
    def __init__(self, model, dataset, dataloader, criterion, optimizer, batch_size, stamp, is_wholescene=True, decay_step=10, decay_factor=0.7, active_epoch = 1, annotation_label = "chair", epoch=17):
        self.epoch = epoch                    # set in __call__
        self.verbose = 0                  # set in __call__
        self.active_epoch = active_epoch
        torch_random_seed(random_seed = 1024)
        self.model = model
        self.dataset = dataset
        self.chair_acc = False
        
        self.dataloader = dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.stamp = stamp
        self.is_wholescene = is_wholescene
        self.scheduler = StepLR(optimizer, step_size=decay_step, gamma=decay_factor)
        self.annotation_label = annotation_label
        self.best = {
            "epoch": 0,
            "loss": float("inf"),
            "point_acc": -float("inf"),
            "point_acc_per_class": -float("inf"),
            "voxel_acc": -float("inf"),
            "voxel_acc_per_class": -float("inf"),
            "point_miou": -float("inf"),
            "voxel_miou": -float("inf"),
        }

        # log
        # contains all necessary info for all phases
        self.log = {phase: {} for phase in ["train", "val","query"]}
        
        # tensorboard
        tb_path = os.path.join(CONF.OUTPUT_ROOT, stamp, "tensorboard")
        os.makedirs(tb_path, exist_ok=True)
        self._log_writer = SummaryWriter(tb_path)

        # private
        # only for internal access and temporary results
        self._running_log = {}
        self._global_iter_id = 0
        self._total_iter = {}             # set in __call__

        # templates
        self.__iter_report_template = ITER_REPORT_TEMPLATE
        self.__epoch_report_template = EPOCH_REPORT_TEMPLATE
        self.__best_report_template = BEST_REPORT_TEMPLATE
        

    def _get_scene_list(self, train_path, query_path):
        scene_list = []
        with open(train_path) as f:
            for scene_id in f.readlines():
                scene_list.append(scene_id.strip())
            train_scene_list = scene_list # train_list??? ?????? ?????? scene??? ?????????
        scene_list = []
        with open(query_path) as f:
            for scene_id in f.readlines():
                scene_list.append(scene_id.strip())
            query_scene_list = scene_list # query_list ?????? ?????? scene??? ?????????
            
        return train_scene_list, query_scene_list
        


    def __call__(self, args, epoch, verbose, occlusion_round):
        # setting
        self.epoch = epoch
        self.verbose = verbose
        self.occlusion_round = occlusion_round
        train_scene_list, query_scene_list = self._get_scene_list(CONF.SCANNETV2_TRAIN, CONF.SCANNETV2_QUERY)
        #query_scene_data = np.load(CONF.SCANNETV2_QUERY_FILE.format(query_scene_list[0]))
        #76510?????? ????????? ?????????
        #new_user_input_list = query_scene_data[25000:32651]
        #self.dataset["train"] = ScannetDataset_update_New_TrainData("train", train_scene_list, query_scene_list, new_user_input_list, is_weighting=not args.no_weighting, use_color=args.use_color, use_normal=args.use_normal, use_multiview=args.use_multiview)
            
        self.dataloader["train"] = DataLoader(self.dataset["train"], batch_size=args.batch_size, collate_fn=collate_random, num_workers=args.num_workers, pin_memory=True)

        self._total_iter["train"] = len(self.dataloader["train"]) * epoch
        self._total_iter["val"] = len(self.dataloader["query"]) * epoch
        self.args = args
        # val ????????? ??? 8192?????? ?????????
        # ????????? ?????? ??????!
        self.dataset["val"].generate_chunks()
        
        
        self.dataset["query"].generate_chunks_for_validation_all()
        # train ????????? ??? 15%??? ?????????
        self.dataset["train"].generate_chunks()

        """
        a = self.dataset["train"].chunk_data[list(self.dataset["train"].chunk_data.keys())[0]][230:250]


        
        self.dataset["train"].generate_chunks()
        b = self.dataset["train"].chunk_data[list(self.dataset["train"].chunk_data.keys())[0]][230:250]

       

        print("??????????")
        print(a==b)
        print("------")


        return NotImplemented 
        """

        #pretrained_root = "/content/drive/MyDrive/ewhain_original/????????????????????????/???????????????/New_Davedrum/Pointnet2.ScanNet/outputs/2021-07-29_11-24-45_MSG"
        #saved_file_root = os.path.join(pretrained_root, "model.pth")
        #saved_parameters = torch.load(saved_file_root, map_location=lambda storage, loc:storage.cuda(0))
        #self.model.load_state_dict(saved_parameters)

        # ?????? train data??? 5??? ?????? ????????? 1/12
        for epoch_id in range(self.epoch):
            print("epoch {} starting...".format(epoch_id + 1))
            
            # generate new chunks
            # ??? epoch?????? train ????????? 8192(dataset.npoints???)??? ?????? ???????????? ?????????
            #self.dataset["train"].generate_chunks() # train data?????? 8192?????? ?????? ???????????? ?????????
            #self.dataset["val"].generate_chunks()
            global total_list
            total_list = []
            for i in range(20): 
              total_list.append([])
            
            
            # train
            self._set_phase("train")
            # train loader??? ???????????? ?????? _train ????????? ?????? ???, iterate ?????? ?????? ????????? ?????? chunk ?????? ???????????? ??? (__call ?????? ?????? >> dataset.py)
            
            # ???????????? ???????????? 1/12
            self._train(self.dataloader["train"], epoch_id)
            #self._train(self.dataloader["train"], epoch_id)

            # val
            self._set_phase("eval")
            self._val(self.dataloader["query"], epoch_id)
            avg_total_list = [0 if len(x)==0 else sum(x)/len(x) for x in total_list]
           
            sorting_array = np.array(avg_total_list)
            confidence_ranking = np.argsort(sorting_array)[::-1]
            print("====================")
            #print("??? confidence ?????? ????????? : ",sorting_array)
            print("?????? ??? ????????? : ",confidence_ranking)
            print([CONF.NYUCLASSES[int(x)] for x in confidence_ranking])
            print("====================")

            # epoch report
            
            self._epoch_report(epoch_id)

            # load tensorboard
            self._dump_log(epoch_id)

            # scheduler
            self.scheduler.step() # ?????????????????? learning rate??? ???????????? scheduler??? ?????????, ??? ????????? ????????? ????????? ????????? ??? ??????. (????????? ?????? ???????????? ?????? ?????????)
            
            
            summary = SummaryWriter()
            if epoch_id % 5 == 0:    # ??? 10 iteration?????? ????????????
              
              summary.add_scalar('D_loss_adv', np.mean([loss for loss in self.log["train"][epoch_id]["loss"]]), epoch_id)
              
              summary.add_scalar('D_loss_cls', np.mean([loss for loss in self.log["val"][epoch_id]["loss"]]), epoch_id)
            
        # print best
        self._best_report()
        
        # save model
         
        print("saving last models...\n")
        model_root = os.path.join(CONF.OUTPUT_ROOT, self.stamp)
       
        torch.save(self.model.state_dict(), os.path.join(model_root, "_initial_model_last.pth"))
        
        
        # export
        self._log_writer.export_scalars_to_json(os.path.join(CONF.OUTPUT_ROOT, self.stamp, "tensorboard", "all_scalars.json"))
              
        # best ?????? ????????? ???????????? ?????????
        saved_file_root = os.path.join(model_root, "_initial_best_model.pth")
        saved_parameters = torch.load(saved_file_root, map_location=lambda storage, loc:storage.cuda(0))
        self.model.load_state_dict(saved_parameters)
        
        # validation ????????? ?????? ????????? chunk??? ??????
        #self.dataset["query"].generate_chunks_for_validation_all()
        self._val_confidence_points(self.dataloader["query"], epoch_id)

        # ????????? query scene??? ????????? random?????? 5%??? ???????????? ?????????
        #self.dataset["query"].generate_chunks()
        #self.dataset["query"]._infront_points(occlusion_round=1)
        

        # ??????????????? confidence??? ?????? active learning ??????
        ######################################################
        # epoch?????? ????????? ????????? ??????, pred??? ?????? user????????? labeled data??? ????????? ?????????. ??? ???????????? ????????? solver??? ?????? ????????????, ????????? ?????? ????????? ???????????? ???. 
        # query??? ????????? ?????? scene 1?????? ????????? class ?????? pred??? confidence??? ????????????. 
        # active_epoch??? ??? ??? active learning ?????? ??? ??? ??? ??? ???????????? ???????????????, default??? 3?????? solver ?????? ????????? ??? parameter ????????????
        # epoch??? ??? ??? ??? active learning ????????????
        
        new_user_input_list = []

        query_scene_data = np.load(CONF.SCANNETV2_QUERY_FILE.format(query_scene_list[0]))
        """
        train_scene_list, query_scene_list = self._get_scene_list(CONF.SCANNETV2_TRAIN, CONF.SCANNETV2_QUERY)
        
        #?????? ??? ?????? ????????? ?????? ?????? ??? ?????? ????????????, , 
        train_scene_data = np.load(CONF.SCANNETV2_QUERY_FILE.format(train_scene_list[0]))
        self.query_scene_data_npy = query_scene_data
        self.train_scene_data_npy = train_scene_data
        if active_epoch_id == 0:
          label = 2 #cabinet
        elif active_epoch_id == 1:
          label = 6 # table
        elif active_epoch_id == 2:
          label = 3 #bed
        else:
          print('please check epoch again.')
        query_label, view = self._confidence_pred_for_AL(self.dataloader["query"],self.query_scene_data_npy,label)
        """
            # user?????? ?????? label??? ?????? query??? ?????????. 
            # window??? annotation ????????????. 
        
        # ?????? for????????? ?????? new_user_input_list??? ????????? ??????.
        
        #new_user_input_list = [[1,2,3,4,5,6,7,8,9,10,11],[11,10,9,8,7,6,5,4,3,2,1]]
        #new_input="a"
        
        
        
        #print("????????? ????????? ???????????? ", view, " ?????? ??? ?????????.")
        #print(query_scene_data[0:6,:])

        
        #print(CONF.NYUCLASSES[label]+" ??? ???????????? ?????? query scene?????? annotation ????????????. (annotation.txt) ")
        #final = "3"
        #print("????????? ???????????? 0??? ???????????????.")
        #while(final!="0"):
        #  final = input()
        user_file_name = "/content/drive/MyDrive/ewhain_original/?????????/??????/New_Davedrum/Pointnet2.ScanNet/user_input/"+args.scene_number+"__"+args.annotation_label+"__"+args.method+"__"+args.tag+".txt"
        f = open(user_file_name,"r")
        line = "start"
        new_user_input_list = []
        while line:
          line = f.readline()
          #print(line)
          new_input = line.split(" ")
          
          new_input = [i.split(",") for i in new_input]
          
          
          
          for i in new_input:
            if len(i)<=1:
              break
            new_float_input = list(map(float, i)) # [1., 2., 3.]
            
            #print("?????? ????????? : ", new_float_input) 
            point_index_query_x = np.where(np.round(query_scene_data[:,0],6) == round(float(new_float_input[0]), 6))
            point_index_query_x = list(point_index_query_x)[0]
            #print(point_index_query_x)
            #print("?????? ??? ????????? : ", query_scene_data[:3,1])
            point_index_query_y = np.where(np.round(query_scene_data[:,1],6) == round(float(new_float_input[1]), 6))
            point_index_query_y = list(point_index_query_y)[0]
            #print(point_index_query_y)
            
            point_index_query_z = np.where(np.round(query_scene_data[:,2],6) == round(float(new_float_input[2]), 6))
            point_index_query_z = list(point_index_query_z)[0]
            #print(point_index_query_z)

            point_index_query = list(set(point_index_query_x.tolist()) & set(point_index_query_y.tolist()) & set(point_index_query_z.tolist()))
            #print(point_index_query)
            new_float_input = query_scene_data[point_index_query[0],:10]
            if self.annotation_label =="chair":
              query_label = 4
            elif self.annotation_label == "table":
              query_label = 6
            

            #print(new_float_input)
            new_float_input = np.hstack((new_float_input,[query_label]))
            #print(new_float_input)
            #print(new_float_input.tolist())
            #print("??????, ", new_user_input_list)
            new_user_input_list = new_user_input_list+[new_float_input.tolist()]
            #print(new_user_input_list)
        #print('????????? input??? : ',new_user_input_list[:3])
        '''
        uploaded = files.upload()
        for name, data in uploaded.items():
            f= open(name, 'r')
            line = "start"
            while line:
              line = f.readline()
              
              new_input = line.split(",")
              
              
              #print(new_input)
              #print(len(new_input))
              if (len(new_input)==1):
                
                break
              new_input = list(map(float, new_input)) # [1., 2., 3.]
              #print(new_input)
        '''
            
          

        
        
        #self.dataset["train"] = ScannetDataset_update_New_TrainData("train", train_scene_list, query_scene_list, new_user_input_list, is_weighting=not args.no_weighting, use_color=args.use_color, use_normal=args.use_normal, use_multiview=args.use_multiview)
        train_scene_list, query_scene_list = self._get_scene_list(CONF.SCANNETV2_TRAIN, CONF.SCANNETV2_QUERY)
        #print("=====")
        #print(self.dataset["query"].chunk_data[query_scene_list[0]])
        #print(self.dataset["query"].chunk_data[query_scene_list[0]].shape)
        #print("=====")
        new_user_input_list = np.array(new_user_input_list)
        
        #print("=====")
        #print(new_user_input_list)
        #print(new_user_input_list.shape)
        #print("=====")
        query_choice = np.random.choice(new_user_input_list.shape[0], 8192, replace=True)
        #query_choice = np.random.choice(new_user_input_list.shape[0], 5130, replace=True)
        self.dataset["train"].chunk_data[query_scene_list[0]] = new_user_input_list[query_choice]
        """
        for i in range(100):
          query_choice = np.random.choice(new_user_input_list.shape[0], 8192, replace=True)
          #query_choice = np.random.choice(new_user_input_list.shape[0], new_user_input_list.shape[0], replace=True)
          self.dataset["train"].chunk_data[i] = new_user_input_list[query_choice]
        """
        
        #self.dataset["train"].chunk_data[query_scene_list[0]] = np.concatenate([self.dataset["train"].chunk_data[query_scene_list[0]], new_user_input_list])
        #print("=====")
        #print(self.dataset["query"].chunk_data[query_scene_list[0]])
        #print(self.dataset["query"].chunk_data[query_scene_list[0]].shape)
        #print("=====")
        self.dataloader["train"] = DataLoader(self.dataset["train"], batch_size=args.batch_size, collate_fn=collate_random, num_workers=args.num_workers, pin_memory=True)
        
      

        for active_epoch_id in range(self.epoch):
          self.model.eval()
          # ?????? ??? confidence??? ????????? ?????? confidence??? ?????? label??? ????????????.  [[label],[confidence]]
          
          
          self._total_iter["train"] = len(self.dataloader["train"]) * epoch
          self._total_iter["val"] = len(self.dataloader["query"]) * epoch
          
          
          print("active learning epoch {} starting...".format(active_epoch_id + 1))
          
          # generate new chunks
          #self.dataset["train"].generate_chunks() 
          #self.dataset["val"].generate_chunks()
          #self.dataset["query"].generate_chunks()
          #print(self.dataset["train"].chunk_data["scene0003_00"])
          #print(len(self.dataset["train"].chunk_data["scene0003_00"]))
          
          
          total_list = []
          for i in range(20): 
            total_list.append([])
          # train
          self._set_phase("train")
          self._train(self.dataloader["train"], active_epoch_id, active = True)

          # val
          self._set_phase("eval")
          
            ##            
            
            
          self._val(self.dataloader["query"], active_epoch_id, active = True)
          

          avg_total_list = [0 if len(x)==0 else sum(x)/len(x) for x in total_list]
           
          sorting_array = np.array(avg_total_list)
          confidence_ranking = np.argsort(sorting_array)[::-1]
          print("====================")
          #print("??? confidence ?????? ????????? : ",sorting_array)
          print("?????? ??? ????????? : ",confidence_ranking)
          print([CONF.NYUCLASSES[int(x)] for x in confidence_ranking])
          print("====================")

          # epoch report
          self._epoch_report(active_epoch_id)

          # load tensorboard
          self._dump_log(active_epoch_id+self.epoch)

          # scheduler
          self.scheduler.step() # ?????????????????? learning rate??? ???????????? scheduler??? ?????????, ??? ????????? ????????? ????????? ????????? ??? ??????. (????????? ?????? ???????????? ?????? ?????????)

          # print best
        self._best_report()

        # save model
        # ?????? ????????? ????????? 2?????? scene??? ???????????? ????????? ????????? ????????? ????????????. 
        print("saving last models...\n")
        model_root = os.path.join(CONF.OUTPUT_ROOT, self.stamp)
        torch.save(self.model.state_dict(), os.path.join(model_root, "_active_model_last.pth"))
        

        # export
        self._log_writer.export_scalars_to_json(os.path.join(CONF.OUTPUT_ROOT, self.stamp, "tensorboard", "all_scalars.json"))
        #saved_file_root = os.path.join(model_root, "_active_model.pth")
        #saved_parameters = torch.load(saved_file_root, map_location=lambda storage, loc:storage.cuda(0))
        #self.model.load_state_dict(saved_parameters)
          
    def _formatting_4(self, n):
      return np.round(n,4)        
            
                

    #forward ????????? ???????????? ??????
    def _confidence_pred_for_AL(self, query_loader,query_scene_data_npy, which_label):
        #confidence ranking??? ?????? ?????? label // ??? label??? ?????? ?????? unoccluded ??? ?????? 3?????? return???
        phase = "query"
        for iter_id, data in enumerate(query_loader): # query scene ????????????, for ??? ??? ?????? ???.
            
            '''
            print("==========================")
            
            print(iter_id)
            print(data)
            print("==========================")
            # unpack the data
            coords, feats, semantic_segs, sample_weights, fetch_time = data
            #coords : torch.float32, torch.Size([1, 50000, 3])
            #feats : torch.float32, torch.Size([1, 50000, 3])
            #segs : torch.int64, torch.Size([1, 50000])
            print("????????? ", coords.dtype)
            print(coords.shape)
            print("????????? ", feats.dtype)
            print(feats.shape)
            print("seg??? ", semantic_segs.dtype)
            print(semantic_segs.shape)
            '''
            

            #coords : torch.float32, torch.Size([1, 76510, 3])
            #feats : torch.float32, torch.Size([1, 76510, 3])
            #segs : torch.int64, torch.Size([1, 76510])
            
            # ?????? 4??? ??????????????? ????????? ?????? ?????? ?????????.
            coords = torch.Tensor([query_scene_data_npy[:,:3]])
            #print(query_scene_data_npy[:10,:3])
            #len(feats_2) = 76510
            #feats[0] == [120.0, 94.0, 68.0, -0.1291525512933731, -0.8436745405197144, -0.5210881233215332]
            feats = np.concatenate([query_scene_data_npy[:,3:6]/ 255,query_scene_data_npy[:,6:9]],axis=1).tolist()
            feats = [list(map(self._formatting_4, x)) for x in feats]
            
            feats = torch.Tensor([feats])
            
            #feats_2 = torch.Tensor()
            
            semantic_segs = torch.Tensor([np.zeros((len(query_scene_data_npy),), dtype=int)]).long()
            sample_weights= torch.Tensor([(np.ones(21))])
            '''
            print("????????? ", coords_2.dtype)
            print(coords_2.shape)
            
            print("????????? ", feats_2.dtype)
            
            print(feats_2.shape)
            print("seg??? ", semantic_segs_2.dtype)
            print(semantic_segs_2.shape)      
            # 1.2000e+02,  9.4000e+01,  6.8000e+01, -1.2915e-01, -8.4367e-01,-5.2109e-01
            
            print(coords[0][:3])
            print(coords_2[0][:3]) 
            print(feats[0][:3])
            print(feats_2[0][:3])  
            print(semantic_segs[0][:3])
            print(semantic_segs_2[0][:3]) 
            
            
            #print("????????? ", coords.dtype)
            
            #print("????????? ", feats.dtype)
            #print("seg??? ", semantic_segs.dtype)
            #print("weight??? ", sample_weights.dtype)
            '''

            coords, feats, semantic_segs, sample_weights = coords.cuda(), feats.cuda(), semantic_segs.cuda(), sample_weights.cuda()
            #self.log[phase][epoch_id]["fetch"].append(fetch_time)
            #print("coords??? ????????? : ", len(coords[0]))
            # forward
            # ????????? ??????, forward ??? ??? ??????
            # query npy ??? 8192?????? ?????? ?????? ?????????
            preds = self._forward(coords, feats, self.is_wholescene)
            
            """
              coords[0] = tensor([[0.9362, 2.1084, 1.1275],
                          [0.3519, 2.2397, 1.9149],
                          [0.4654, 1.8777, 0.0994],
                          ...,
                          [0.6213, 1.5378, 0.9377],
                          [0.1902, 1.9552, 1.9303],
                          [0.5116, 2.1635, 1.9074]], device='cuda:0')
              ==> 8192??? ?????? ??????

              coords[0][0] = tensor([3.0140, 0.9570, 0.4980], device='cuda:0')

              coords[0].cpu().numpy() ==>  coordinate_list
              [[0.6819365  1.797      0.35250002]
              [0.8607943  0.60966295 1.3762187 ]
              [0.2533378  1.2025751  1.4044956 ]
              ...
              [0.10113657 2.0105405  0.23733217]
              [0.28751233 0.4734749  1.9172258 ]
              [0.1228848  1.5997864  1.5938265 ]]

            """
            # numpy.ndarray to list
            # 8192 ?????? ???
            coordinate_list = coords[0].cpu().numpy().tolist()
            
            
            # N?????? ????????? N?????? ???????????? ????????? ??????
            # ==> array([1.09800005, 0.59128624, 0.77000004])



            unOccluded_label = []
            # unOccluded_label ??????
            #[[0, 2, 3, 6, 7, 9, 11, 14, 15,..., ], [8, 46, 71, ...], ..., []]
            """
            for k in range(14):
              print(k,"----> 0 ~ 13 ?????????")
              #unOccluded_label \ [[], [], [], [], [], [], [], [], [], [], [], [], [], []]
              unOccluded_label.append([])
              # coordinate_unOccluded??? ??? ???????????? ?????? ?????? ????????? point?????? ??????
              # occlusion_round = 8??? ?????? ???????????? ?????? ??????????????? ???
              
              coordinate_unOccluded = [(np.round(x[0],1) for x in list(self.dataset["query"].infront_point[k].values()))]
              print(self.dataset["query"].infront_point[k].values())
              
              coordinate_unOccluded = [list(x) for x in coordinate_unOccluded[0]]
              
              print("===")
              print(coordinate_unOccluded)
              print("===")

              # len(coordinate_list) == 76510???
              for n in range(len(coordinate_list)):
                # ????????? ?????? ?????? ???????????? ??????????????? numpy ?????? ????????? 0.1 ????????? ??? cm?????? ????????????
                if (n<=5):
                    print("?????? k?????????")
                    print(np.round(coordinate_list[n],1).tolist())
                if ((np.round(coordinate_list[n],1).tolist()) in coordinate_unOccluded):
                  #print(n)
                  #print(coordinate_list[n])
                  
                  unOccluded_label[-1].append(n)
                  # occlusion ?????? ?????? ????????? ??? ???????????? ?????? ??????????????? ????????? (coords ???????????? ?????????)
              # print(unOccluded_label)
              # unOccluded_label??? ??? ????????? ????????? pred ?????? ?????????, occlusion??? ?????? ?????? ?????? ?????? ???????????? index
              # unOccluded_label ==> [[1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 20, 21, 22, 27, 28, 29, 30, 32, 33, 34, 36, 37, 39, 40, 41, 44, 47, 48, 49, 51, 52, 55, 57, 58, 59, 60, 61, 63, 64, 68, 70, 71, 74, 76, 77, 78, 81, 84, 85, 87, 88, 89, 90, 91, 92, 95, 96, 97, 98, 99, 100, 103, 106, 110, 111, 112, 114, 115, 116, 117, 122, 123, 124, 125, 126, 128, 129, 130, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 144, 145, 146, 147, 150, 153, 154, 156, 157, 158, 159, 162, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 178, 179, 180, 181, 182, 183, 184, 185, 186, 188, 189, 190, 192, 194, 196, 197, 198, 199, 200, 201, 202, 203, 206, 208, 209, 210, 212, 214, 215, 216, 218, 219, 222, 223, 226, 227, 228, 230, 231, 232, 233, 234, 235, 236, 237, 238, 240, 241, 242, 244, 246, 247, 248, 249, 253, 254, 255, 256, 260, 262, 263, 265, 268, 269, 270, 271, 272, 273, 275, 276, 277, 280, 281, 284, 285, 286, 287, 289, 291, 292, 294, 296, 297, 298, 299, 300, 301, 303, 304, 305, 306, 307, 308, 311, 312, 313, 314, 316, 318, 319, 321, 322, 323, 324]]
            print("unocclusded_label??? : (??? 14?????? ???????????? ????????? ?????? ????????? ?????????")
            print(unOccluded_label)
            for i in range(len(unOccluded_label)):
              print('????????? ')
              print(len(unOccluded_label[i]))

            """
            """
              for n in range(len(list(self.dataset["query"].infront_point[k].keys()))):
                # self.dataset["query"].infront_point[0][list(self.dataset["query"].infront_point[0].keys())[n]][0] ==> [1.05492127 2.73373747 1.1574291 ]
                coordinate_unOccluded = self.dataset["query"].infront_point[0][list(self.dataset["query"].infront_point[0].keys())[n]][0]
                index_unOccluded = np.where(coordinate_list == coordinate_unOccluded)
                if (index_unOccluded[0].size!=0) :
                  unOccluded_label[-1].append((coordinate_unOccluded,index_unOccluded[0][0]))
                # index_unOccluded = (array([5730, 5730, 5730, 7193, 7193, 7193]), array([0, 1, 2, 0, 1, 2])) 
                # index_unOccluded = (array([]), array([])) -> ????????? 8192??? ???????????? occluded (?????? ???????????? ?????? ?????????) point ?????? ?????? ??????

                # 5730, 7193 ????????? ????????????
                # ????????? ????????????, ??? ?????? ???????????? ???????????? ????????? (???????????? 5730?????? ?????????)
              print(unOccluded_label)
            """

            #self._compute_loss(preds, semantic_segs, sample_weights)
            #self._eval(coords, preds, semantic_segs, sample_weights, self.is_wholescene)

            # record log
            """
            self.log[phase][epoch_id]["loss"].append(self._running_log["loss"].item())
            self.log[phase][epoch_id]["point_acc"].append(self._running_log["point_acc"])
            self.log[phase][epoch_id]["point_acc_per_class"].append(self._running_log["point_acc_per_class"])
            self.log[phase][epoch_id]["voxel_acc"].append(self._running_log["voxel_acc"])
            self.log[phase][epoch_id]["voxel_acc_per_class"].append(self._running_log["voxel_acc_per_class"])
            self.log[phase][epoch_id]["point_miou"].append(self._running_log["point_miou"])
            self.log[phase][epoch_id]["voxel_miou"].append(self._running_log["voxel_miou"])
            """
            # ??? ????????? ????????? ??????
            # points_list = [[-0.11924917995929718, -0.04613947123289108, -0.20690718293190002, -0.2028222382068634, 0.31167054176330566, -0.08550150692462921, -0.040521204471588135, 0.04196982830762863, 0.029743794351816177, 0.151509091258049, -0.05134827271103859, 0.005890154279768467, 0.22072120010852814, -0.3832728862762451, 0.19867755472660065, -0.13119345903396606, 0.22404149174690247, 0.056744541972875595, 0.09138473868370056, 0.21279717981815338], [0.0028198033105582, 0.0317632257938385, -0.03630537912249565, 0.03719541057944298, -0.0015036762924864888, 0.11752472072839737, 0.23951327800750732, -0.12636196613311768, 0.08242180198431015, -0.023187827318906784, 0.06832046806812286, 0.05200092867016792, -0.03182460740208626, 0.23237772285938263, -0.021574841812253, 0.034832652658224106, 0.04854806140065193, 0.00804646871984005, -0.02367490343749523, 0.24299505352973938], [], ... , []]
            # len(preds[0]) ==> 76510 ??? 
            
            points_list = preds[0].tolist()
            
            
            # ??? point ?????? ????????? ????????? ????????? (ex. 3??? sofa -- ????????? ???????????? ?????? ????????? ??????, 9??? window ...)
            # Index_for_each_label = [3, 5, 8, ... , 11]
            Index_for_each_label = [x.index(max(x)) for x in points_list]
            print("============?????? ?????????============")
            print(len(Index_for_each_label))
            print(Index_for_each_label)
            
            

            # ??? point ?????? ????????? ????????? ?????? (confidence) 
            # Value_for_each_label = [0.31167054176330566, 0.24299505352973938, 0.17710644006729126, 0.4054673910140991, ... , ]
            Value_for_each_label = [max(x) for x in points_list]

            
            total_list = []
            for i in range(20): 
              total_list.append([])
            for i in range(len(Index_for_each_label)):
              total_list[Index_for_each_label[i]].append(Value_for_each_label[i])
            avg_total_list = [0 if len(x)==0 else sum(x)/len(x) for x in total_list]
           
            sorting_array = np.array(avg_total_list)
            confidence_ranking = np.argsort(sorting_array)[::-1]
            print("??? confidence ?????? ????????? : ",sorting_array)
            print("?????? ??? ????????? : ",confidence_ranking)
            """
            count_label = []
            for j in range(14):
              count_label_list = [Index_for_each_label[x] for x in unOccluded_label[j]]
              print("count_label_list??? ????????? ",len(count_label_list))
              print("??? ????????? ??????????????? ???????????? ????????????",count_label_list[:10])
              
              #count_label.append(count_label_list.count(int(confidence_ranking[-1])))
              count_label.append(count_label_list.count(int(which_label)))
            print(count_label)
            count_label_array = np.array(count_label)
            print("????????? : ", count_label_array)
            view_indexes = np.argsort(count_label_array)[::-1]
            print("????????? : ", view_indexes)
            """
            
            #print("????????? ???????????? ?????? ??????", query_scene_data_npy[np.where(query_scene_data_npy[:,10]==confidence_ranking[-1]),:3])
            # sofa??? ????????? ????????? confidence ??????, window, ... ?????? ????????? ???????????? ?????? ??? ???????????? ??????.
            # confidence_ ranking??? ???????????? ?????? ???????????? [15 10 12  9 19 13  8  6 18 16  3 14  7  1 11  5  4  0 17  2]
            return confidence_ranking[0], view_indexes[:3]

        """
        if self.is_wholescene:
            pred = []
            coord_chunk, feat_chunk = torch.split(coord.squeeze(0), self.batch_size, 0), torch.split(feat.squeeze(0), self.batch_size, 0)
            # squeeze ????????? ????????? 1??? ????????? ???????????????. (ex. (3,1,4)==squeeze==>(3,4))
            # ??? ??? ??????????????? 0?????? ?????????????????? 0????????? ?????? ?????? ?????? 1??? ????????? ????????? ??????. (ex. (1,2,4)==squeeze(0)==>(2,4))
            # torch.split(tensor, size, dim)??? ????????? dim??? ?????? ?????? tensor??? size?????? ???????????? ?????????. 
            # ???????????? coord.squeeze(0)??? batch_size ?????? ?????? dim=0??? ?????? tensor??? ???????????? ?????????. 
            assert len(coord_chunk) == len(feat_chunk)
            for coord, feat in zip(coord_chunk, feat_chunk):
                output = self.model(torch.cat([coord, feat], dim=2))
                pred.append(output)

            pred = torch.cat(pred, dim=0).unsqueeze(0) 
            # pred??? ?????? ?????? ???????????? ???????????? ?????? ????????????
            # ????????? torch.cat(pred, dim=0)??? ?????? scene ?????? ?????? ?????? ???????????? ?????? pred ?????? ??? ??? ?????? ??????
            # unsqueeze??? ????????? dim(???????????? 0??????)??? size??? 1??? ??? ????????? ??????????????? ????????? ???????????????. (ex. size??? (3,)??? ?????????????????? unsqueeze(0)??? ??????, size??? (1,3)??? ??????????????? ?????????)

        else:
            output = self.model(torch.cat([coord, feat], dim=2)) # coord ????????? feat ????????? 2????????? ???????????? ?????????
            pred = output

        return pred
        """

    def _set_phase(self, phase):
        if phase == "train":
            self.model.train()
        elif phase == "eval":
            self.model.eval()

        else:
            raise ValueError("invalid phase")

    def _forward(self, coord, feat, is_wholescene):
        if self.is_wholescene:
            pred = []
            coord_chunk, feat_chunk = torch.split(coord.squeeze(0), self.batch_size, 0), torch.split(feat.squeeze(0), self.batch_size, 0)
            # squeeze ????????? ????????? 1??? ????????? ???????????????. (ex. (3,1,4)==squeeze==>(3,4))
            # ??? ??? ??????????????? 0?????? ?????????????????? 0????????? ?????? ?????? ?????? 1??? ????????? ????????? ??????. (ex. (1,2,4)==squeeze(0)==>(2,4))
            # torch.split(tensor, size, dim)??? ????????? dim??? ?????? ?????? tensor??? size?????? ???????????? ?????????. 
            # ???????????? coord.squeeze(0)??? batch_size ?????? ?????? dim=0??? ?????? tensor??? ???????????? ?????????. 
            assert len(coord_chunk) == len(feat_chunk)

            

            for coord, feat in zip(coord_chunk, feat_chunk):
                output = self.model(torch.cat([coord, feat], dim=2))
                pred.append(output)

            pred = torch.cat(pred, dim=0).unsqueeze(0) 
            
            outputs = pred.max(3)[1]
            
          
            # pred??? ?????? ?????? ???????????? ???????????? ?????? ????????????
            # ????????? torch.cat(pred, dim=0)??? ?????? scene ?????? ?????? ?????? ???????????? ?????? pred ?????? ??? ??? ?????? ??????
            # unsqueeze??? ????????? dim(???????????? 0??????)??? size??? 1??? ??? ????????? ??????????????? ????????? ???????????????. (ex. size??? (3,)??? ?????????????????? unsqueeze(0)??? ??????, size??? (1,3)??? ??????????????? ?????????)
        else:
            
            output = self.model(torch.cat([coord, feat], dim=2)) # coord ????????? feat ????????? 2????????? ???????????? ?????????
            pred = output
        return pred

    def _backward(self):
        # optimize
        self.optimizer.zero_grad() # ??? ??????????????? grad ?????? ?????? 0?????? ????????? ??????, ?????? backward??? ????????? ?????? ?????????
        self._running_log["loss"].backward() # ??????????????? grad?????? update???
        # self._clip_grad()
        self.optimizer.step() # ???????????? ??????

    def _compute_loss(self, pred, target, weights):
        num_classes = pred.size(-1)
        loss = self.criterion(pred.contiguous().view(-1, num_classes), target.view(-1), weights.view(-1))
        self._running_log["loss"] = loss

    def _train(self, train_loader, epoch_id, active = False):
        # setting
        phase = "train"
        if active==True:
          epoch_id+=self.epoch
        else:
          epoch_id+=0
        self.log[phase][epoch_id] = {
            # info
            "forward": [],
            "backward": [],
            "fetch": [],
            "iter_time": [],
            # loss (float, not torch.cuda.FloatTensor)
            "loss": [],
            # scores (float, not torch.cuda.FloatTensor)
            "point_acc": [],
            "point_acc_per_class": [],
            "voxel_acc": [],
            "voxel_acc_per_class": [],
            "point_miou": [],
            "voxel_miou": [],
        }
        for iter_id, data in enumerate(train_loader):

            """
            if (iter_id ==0) and (active==False) :
              
              np.savetxt('/content/drive/MyDrive/ewhain_original/????????????????????????/???????????????/New_Davedrum/data_test_second.txt', data[0][1].numpy())
            """
            
            # initialize the running loss
            self._running_log = {
                # loss
                "loss": 0,
                # acc
                "point_acc": 0,
                "point_acc_per_class": 0,
                "voxel_acc": 0,
                "voxel_acc_per_class": 0,
                "point_miou": 0,
                "voxel_miou": 0,
            }
            # unpack the data
            coords, feats, semantic_segs, sample_weights, fetch_time = data
            coords, feats, semantic_segs, sample_weights = coords.cuda(), feats.cuda(), semantic_segs.cuda(), sample_weights.cuda()
            self.log[phase][epoch_id]["fetch"].append(fetch_time)

            # forward
            start_forward = time.time()
            # train ??? ??? ?????? forward ?????? ??????
            preds = self._forward(coords, feats, self.is_wholescene)

            self._compute_loss(preds, semantic_segs, sample_weights)
            
            if (iter_id ==1) and (active==False) :
              
              np.savetxt('/content/drive/MyDrive/ewhain_original/????????????????????????/???????????????/New_Davedrum/data_test.txt', preds.cpu().detach().numpy()[0])
            
            self._eval(coords, preds, semantic_segs, sample_weights, self.is_wholescene, self.chair_acc)
            
            ##
            points_list = preds[0].tolist()
            
            
            # ??? point ?????? ????????? ????????? ????????? (ex. 3??? sofa -- ????????? ???????????? ?????? ????????? ??????, 9??? window ...)
            # Index_for_each_label = [3, 5, 8, ... , 11]
            Index_for_each_label = [x.index(max(x)) for x in points_list]
            #print("============?????? ?????????============")
            #print(len(Index_for_each_label))
            #print(Index_for_each_label)
            
            

            # ??? point ?????? ????????? ????????? ?????? (confidence) 
            # Value_for_each_label = [0.31167054176330566, 0.24299505352973938, 0.17710644006729126, 0.4054673910140991, ... , ]
            Value_for_each_label = [max(x) for x in points_list]

            
            
            
            
            self.log[phase][epoch_id]["forward"].append(time.time() - start_forward)

            # backward
            start = time.time()
            self._backward()
            self.log[phase][epoch_id]["backward"].append(time.time() - start)

            try:
            # record log
              self.log[phase][epoch_id]["loss"].append(self._running_log["loss"].item())
              self.log[phase][epoch_id]["point_acc"].append(self._running_log["point_acc"])
              self.log[phase][epoch_id]["point_acc_per_class"].append(self._running_log["point_acc_per_class"])
              self.log[phase][epoch_id]["voxel_acc"].append(self._running_log["voxel_acc"])
              self.log[phase][epoch_id]["voxel_acc_per_class"].append(self._running_log["voxel_acc_per_class"])
              self.log[phase][epoch_id]["point_miou"].append(self._running_log["point_miou"])
              self.log[phase][epoch_id]["voxel_miou"].append(self._running_log["voxel_miou"])

              # report
              iter_time = self.log[phase][epoch_id]["fetch"][-1]
              iter_time += self.log[phase][epoch_id]["forward"][-1]
              iter_time += self.log[phase][epoch_id]["backward"][-1]
              self.log[phase][epoch_id]["iter_time"].append(iter_time)
              if (iter_id + 1) % self.verbose == 0:
                  self._train_report(epoch_id)


              # update the _global_iter_id
              self._global_iter_id += 1
            except:
              pass

    def _val(self, val_loader, epoch_id, active = False):
        # setting
        phase = "val"

        if active==True:
          epoch_id+=self.epoch
        else:
          epoch_id+=0

        self.log[phase][epoch_id] = {
            # info
            "forward": [],
            "backward": [],
            "fetch": [],
            "iter_time": [],
            # loss (float, not torch.cuda.FloatTensor)
            "loss": [],
            # scores (float, not torch.cuda.FloatTensor)
            "point_acc": [],
            "point_acc_per_class": [],
            "voxel_acc": [],
            "voxel_acc_per_class": [],
            "point_miou": [],
            "voxel_miou": [],
        }
        for iter_id, data in enumerate(val_loader):
            # initialize the running loss
            self._running_log = {
                # loss
                "loss": 0,
                # acc
                "point_acc": 0,
                "point_acc_per_class": 0,
                "voxel_acc": 0,
                "voxel_acc_per_class": 0,
                "point_miou": 0,
                "voxel_miou": 0,
            }

            # unpack the data
            coords, feats, semantic_segs, sample_weights, fetch_time = data
            #print("============ val ???????????? =============")
            #print((semantic_segs[0][:10]))

            coords, feats, semantic_segs, sample_weights = coords.cuda(), feats.cuda(), semantic_segs.cuda(), sample_weights.cuda()
            self.log[phase][epoch_id]["fetch"].append(fetch_time)

            # forward
            # ????????? ??????, forward ??? ??? ??????
            preds = self._forward(coords, feats, self.is_wholescene)
            #print("========val??? ?????? ????????????=========")
            points_list = preds[0].tolist()
            Index_for_each_label = [x.index(max(x)) for x in points_list]

            

            #print(Index_for_each_label[:10])

            Value_for_each_label = [max(x) for x in points_list]
            global total_list
            for i in range(len(Index_for_each_label)):
              total_list[Index_for_each_label[i]].append(Value_for_each_label[i])
            
            
            self._compute_loss(preds, semantic_segs, sample_weights)
            self.chair_acc = True
            self._eval(coords, preds, semantic_segs, sample_weights, self.is_wholescene, self.chair_acc)
            self.chair_acc = False

            # record log
            self.log[phase][epoch_id]["loss"].append(self._running_log["loss"].item())
            self.log[phase][epoch_id]["point_acc"].append(self._running_log["point_acc"])
            self.log[phase][epoch_id]["point_acc_per_class"].append(self._running_log["point_acc_per_class"])
            self.log[phase][epoch_id]["voxel_acc"].append(self._running_log["voxel_acc"])
            self.log[phase][epoch_id]["voxel_acc_per_class"].append(self._running_log["voxel_acc_per_class"])
            self.log[phase][epoch_id]["point_miou"].append(self._running_log["point_miou"])
            self.log[phase][epoch_id]["voxel_miou"].append(self._running_log["voxel_miou"])

            print("loss: ", np.mean(self.log[phase][epoch_id]["loss"]))
            print("point_acc: ",np.mean(self.log[phase][epoch_id]["point_acc"]))
            print("point_acc_per_class: ",np.mean(self.log[phase][epoch_id]["point_acc_per_class"]))
            print("voxel_acc: ",np.mean(self.log[phase][epoch_id]["voxel_acc"]))
            print("voxel_acc_per_class: ",np.mean(self.log[phase][epoch_id]["voxel_acc_per_class"]))
            print("point_miou: ",np.mean(self.log[phase][epoch_id]["point_miou"]))
            print("voxel_miou: ",np.mean(self.log[phase][epoch_id]["voxel_miou"]))



        # check best
        cur_criterion = "voxel_miou"
        cur_best = np.mean(self.log[phase][epoch_id][cur_criterion])
        try:
          if cur_best > self.best[cur_criterion]:
              print("best {} achieved: {}".format(cur_criterion, cur_best))
              print("current train_loss: {}".format(np.mean(self.log["train"][epoch_id]["loss"])))
              print("current val_loss: {}".format(np.mean(self.log["val"][epoch_id]["loss"])))
              self.best["epoch"] = epoch_id + 1
              self.best["loss"] = np.mean(self.log[phase][epoch_id]["loss"])
              self.best["point_acc"] = np.mean(self.log[phase][epoch_id]["point_acc"])
              self.best["point_acc_per_class"] = np.mean(self.log[phase][epoch_id]["point_acc_per_class"])
              self.best["voxel_acc"] = np.mean(self.log[phase][epoch_id]["voxel_acc"])
              self.best["voxel_acc_per_class"] = np.mean(self.log[phase][epoch_id]["voxel_acc_per_class"])
              self.best["point_miou"] = np.mean(self.log[phase][epoch_id]["point_miou"])
              self.best["voxel_miou"] = np.mean(self.log[phase][epoch_id]["voxel_miou"])

              # save model
              print("saving models...\n")
              model_root = os.path.join(CONF.OUTPUT_ROOT, self.stamp)
              if active:
                torch.save(self.model.state_dict(), os.path.join(model_root, "_active_best_model.pth"))
              else:
                torch.save(self.model.state_dict(), os.path.join(model_root, "_initial_best_model.pth"))

        except:
          pass


    def _val_confidence_points (self, val_loader, epoch_id):
        # setting
        print("val ??????")
        global confidence_points
        Index_for_each_label=[]
        for iter_id, data in enumerate(val_loader):
            # initialize the running loss
            
            # unpack the data
            coords, feats, semantic_segs, sample_weights, fetch_time = data
    
            coords, feats, semantic_segs, sample_weights = coords.cuda(), feats.cuda(), semantic_segs.cuda(), sample_weights.cuda()
            
            # ????????? ??????, forward ??? ??? ??????
            preds = self._forward(coords, feats, self.is_wholescene)
            #print("========val??? ?????? ????????????=========")
            points_list = preds[0].tolist()
            scores = [max(x) for x in points_list]
            Index_for_each_label = Index_for_each_label+[x.index(max(x)) for x in points_list]

            confidence_points+=scores
        #print(confidence_points)
        confidence_points = np.array(confidence_points)
        #print(confidence_points)
        print("val ????????????")
        
        
        # assign array and range
        #array_1d = [1, 2, 4, 8, 10, 15]
        
        range_to_normalize = (0, 1)
        normalized_confidence = self.normalize(confidence_points,range_to_normalize[0],range_to_normalize[1])
        
        #print(normalized_confidence)

        sorted_confidence_index = np.argsort(confidence_points)
        #print(sorted_confidence_index)
        
        fw=open('/content/drive/MyDrive/ewhain_original/????????????????????????/???????????????/New_Davedrum/confidence_acc_test.txt','a',encoding='utf-8')
        
        abc = 1
        for i in sorted_confidence_index:
          print(abc)
          abc+=1
          #if (normalized_confidence[i]<=0.3):
          coordinate_confidence_x = np.array2string(self.dataset['query'].chunk_data["scene0041_00"][i][0])
          coordinate_confidence_y = np.array2string(self.dataset['query'].chunk_data["scene0041_00"][i][1])
          coordinate_confidence_z = np.array2string(self.dataset['query'].chunk_data["scene0041_00"][i][2])
          
          fw.write("{},{},{},{},{} \n".format(
            coordinate_confidence_x, 
            coordinate_confidence_y, 
            coordinate_confidence_z,
            CONF.NYUCLASSES[Index_for_each_label[i]], 
            normalized_confidence[i]
          ))

          # f.write(coordinate_confidence_x)
          # f.write(",")
          # f.write(coordinate_confidence_y)
          # f.write(",")
          # f.write(coordinate_confidence_z)
          # f.write(",")
          # f.write(np.array2string(normalized_confidence[i]))
          # f.write(" ")

          # f.write("\n")
          #else:
           # break

        
        fw.close()

    def _eval(self, coords, preds, targets, weights, is_wholescene, chair_acc):
        if is_wholescene:
            coords = coords.squeeze(0).view(-1, 3).cpu().numpy()            # (CK * N, 3)
            preds = preds.max(3)[1].squeeze(0).view(-1).cpu().numpy()       # (CK * N)
            targets = targets.squeeze(0).view(-1).cpu().numpy()             # (CK * N)
            weights = weights.squeeze(0).view(-1).cpu().numpy()             # (CK * N)
        else:
            coords = coords.view(-1, 3).cpu().numpy()            # (B * N, 3)
            preds = preds.max(2)[1].view(-1).cpu().numpy()       # (B * N)
            targets = targets.view(-1).cpu().numpy()             # (B * N)
            weights = weights.view(-1).cpu().numpy()             # (B * N)

        pointacc, pointacc_per_class, voxacc, voxacc_per_class, _, acc_mask = compute_acc(coords, preds, targets, weights)
        if chair_acc:
          print("*************************")
          print("chair??? ?????? ????????????: ")
          print("pointacc_: ",pointacc_per_class[4])
          self.chair_pointacc = pointacc_per_class[4]
          print("voxelacc_: ",voxacc_per_class[4])
          self.chair_voxacc = voxacc_per_class[4]

        pointmiou, voxmiou, miou_mask = compute_miou(coords, preds, targets, weights)
        
        self._running_log["point_acc"] = pointacc
        self._running_log["point_acc_per_class"] = np.sum(pointacc_per_class * acc_mask)/np.sum(acc_mask)
        self._running_log["voxel_acc"] = voxacc
        self._running_log["voxel_acc_per_class"] = np.sum(voxacc_per_class * acc_mask)/np.sum(acc_mask)
        self._running_log["point_miou"] = np.sum(pointmiou * miou_mask)/np.sum(miou_mask)
        self._running_log["voxel_miou"] = np.sum(voxmiou * miou_mask)/np.sum(miou_mask)

    def _dump_log(self, epoch_id):
        # loss
        self._log_writer.add_scalars(
            "log/{}".format("loss"),
            {
                "train": np.mean([loss for loss in self.log["train"][epoch_id]["loss"]]),
                "val": np.mean([loss for loss in self.log["val"][epoch_id]["loss"]])
            },
            epoch_id
        )

        # eval
        self._log_writer.add_scalars(
            "eval/{}".format("point_acc"),
            {
                "train": np.mean([acc for acc in self.log["train"][epoch_id]["point_acc"]]),
                "val": np.mean([acc for acc in self.log["val"][epoch_id]["point_acc"]])
            },
            epoch_id
        )
        self._log_writer.add_scalars(
            "eval/{}".format("point_acc_per_class"),
            {
                "train": np.mean([acc for acc in self.log["train"][epoch_id]["point_acc_per_class"]]),
                "val": np.mean([acc for acc in self.log["val"][epoch_id]["point_acc_per_class"]])
            },
            epoch_id
        )
        self._log_writer.add_scalars(
            "eval/{}".format("voxel_acc"),
            {
                "train": np.mean([acc for acc in self.log["train"][epoch_id]["voxel_acc"]]),
                "val": np.mean([acc for acc in self.log["val"][epoch_id]["voxel_acc"]])
            },
            epoch_id
        )
        self._log_writer.add_scalars(
            "eval/{}".format("voxel_acc_per_class"),
            {
                "train": np.mean([acc for acc in self.log["train"][epoch_id]["voxel_acc_per_class"]]),
                "val": np.mean([acc for acc in self.log["val"][epoch_id]["voxel_acc_per_class"]])
            },
            epoch_id
        )
        self._log_writer.add_scalars(
            "eval/{}".format("point_miou"),
            {
                "train": np.mean([miou for miou in self.log["train"][epoch_id]["point_miou"]]),
                "val": np.mean([miou for miou in self.log["val"][epoch_id]["point_miou"]])
            },
            epoch_id
        )
        self._log_writer.add_scalars(
            "eval/{}".format("voxel_miou"),
            {
                "train": np.mean([miou for miou in self.log["train"][epoch_id]["voxel_miou"]]),
                "val": np.mean([miou for miou in self.log["val"][epoch_id]["voxel_miou"]])
            },
            epoch_id
        )
        self._log_writer.add_scalars(
            "eval/{}".format("chair_point_acc"),
            {
                "val": self.chair_pointacc
            },
            epoch_id
        )
        self._log_writer.add_scalars(
            "eval/{}".format("chair_voxel_acc"),
            {
                "val": self.chair_voxacc
            },
            epoch_id
        )
    def normalize(self,arr, t_min, t_max):
            norm_arr = []
            diff = t_max - t_min
            diff_arr = max(arr) - min(arr)
            normalize_value = 0
            
            for i in arr:
                temp = (((i - min(arr))*diff)/diff_arr) + t_min
                norm_arr.append(temp)
                if normalize_value%1000==0:
                  print(normalize_value)
                
                normalize_value+=1
            return norm_arr

    def _train_report(self, epoch_id):
        # compute ETA
        fetch_time = [time for time in self.log["train"][epoch_id]["fetch"]]
        forward_time = [time for time in self.log["train"][epoch_id]["forward"]]
        backward_time = [time for time in self.log["train"][epoch_id]["backward"]]
        iter_time = [time for time in self.log["train"][epoch_id]["iter_time"]]
        mean_train_time = np.mean(iter_time)
        mean_est_val_time = np.mean([fetch + forward for fetch, forward in zip(fetch_time, forward_time)])
        eta_sec = (self._total_iter["train"] - self._global_iter_id - 1) * mean_train_time
        eta_sec += len(self.dataloader["val"]) * (self.epoch - epoch_id) * mean_est_val_time
        eta = decode_eta(eta_sec)

        # print report
        iter_report = self.__iter_report_template.format(
            global_iter_id=self._global_iter_id + 1,
            total_iter=self._total_iter["train"],
            train_loss=round(np.mean([loss for loss in self.log["train"][epoch_id]["loss"]]), 5),
            train_point_acc=round(np.mean([acc for acc in self.log["train"][epoch_id]["point_acc"]]), 5),
            train_point_acc_per_class=round(np.mean([acc for acc in self.log["train"][epoch_id]["point_acc_per_class"]]), 5),
            train_voxel_acc=round(np.mean([acc for acc in self.log["train"][epoch_id]["voxel_acc"]]), 5),
            train_voxel_acc_per_class=round(np.mean([acc for acc in self.log["train"][epoch_id]["voxel_acc_per_class"]]), 5),
            train_point_miou=round(np.mean([miou for miou in self.log["train"][epoch_id]["point_miou"]]), 5),
            train_voxel_miou=round(np.mean([miou for miou in self.log["train"][epoch_id]["voxel_miou"]]), 5),
            mean_fetch_time=round(np.mean(fetch_time), 5),
            mean_forward_time=round(np.mean(forward_time), 5),
            mean_backward_time=round(np.mean(backward_time), 5),
            mean_iter_time=round(np.mean(iter_time), 5),
            eta_h=eta["h"],
            eta_m=eta["m"],
            eta_s=eta["s"]
        )
        print(iter_report)

    def _epoch_report(self, epoch_id):
        print("epoch [{}/{}] done...".format(epoch_id+1, self.epoch))
        epoch_report = self.__epoch_report_template.format(
            train_loss=round(np.mean([loss for loss in self.log["train"][epoch_id]["loss"]]), 5),
            train_point_acc=round(np.mean([acc for acc in self.log["train"][epoch_id]["point_acc"]]), 5),
            train_point_acc_per_class=round(np.mean([acc for acc in self.log["train"][epoch_id]["point_acc_per_class"]]), 5),
            train_voxel_acc=round(np.mean([acc for acc in self.log["train"][epoch_id]["voxel_acc"]]), 5),
            train_voxel_acc_per_class=round(np.mean([acc for acc in self.log["train"][epoch_id]["voxel_acc_per_class"]]), 5),
            train_point_miou=round(np.mean([miou for miou in self.log["train"][epoch_id]["point_miou"]]), 5),
            train_voxel_miou=round(np.mean([miou for miou in self.log["train"][epoch_id]["voxel_miou"]]), 5),
            val_loss=round(np.mean([loss for loss in self.log["val"][epoch_id]["loss"]]), 5),
            val_point_acc=round(np.mean([acc for acc in self.log["val"][epoch_id]["point_acc"]]), 5),
            val_point_acc_per_class=round(np.mean([acc for acc in self.log["val"][epoch_id]["point_acc_per_class"]]), 5),
            val_voxel_acc=round(np.mean([acc for acc in self.log["val"][epoch_id]["voxel_acc"]]), 5),
            val_voxel_acc_per_class=round(np.mean([acc for acc in self.log["val"][epoch_id]["voxel_acc_per_class"]]), 5),
            val_point_miou=round(np.mean([miou for miou in self.log["val"][epoch_id]["point_miou"]]), 5),
            val_voxel_miou=round(np.mean([miou for miou in self.log["val"][epoch_id]["voxel_miou"]]), 5),
        )
        print(epoch_report)
    
    def _best_report(self):
        print("training completed...")
        best_report = self.__best_report_template.format(
            epoch=self.best["epoch"],
            loss=round(self.best["loss"], 5),
            point_acc=round(self.best["point_acc"], 5),
            point_acc_per_class=round(self.best["point_acc_per_class"], 5),
            voxel_acc=round(self.best["voxel_acc"], 5),
            voxel_acc_per_class=round(self.best["voxel_acc_per_class"], 5),
            point_miou=round(self.best["point_miou"], 5),
            voxel_miou=round(self.best["voxel_miou"], 5),
        )
        print(best_report)
        with open(os.path.join(CONF.OUTPUT_ROOT, self.stamp, "best.txt"), "a") as f:
            f.write(best_report)
    
    