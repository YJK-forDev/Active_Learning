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

class Solver():
    # 본 모델에서는 criterion으로 손실함수 중 하나인 cross-entropy를 사용한다.
    def __init__(self, model, dataset, dataloader, criterion, optimizer, batch_size, stamp, is_wholescene=True, decay_step=10, decay_factor=0.7, active_epoch = 1):
        self.epoch = 0                    # set in __call__
        self.verbose = 0                  # set in __call__
        self.active_epoch = active_epoch
        self.model = model
        self.dataset = dataset
        
        
        self.dataloader = dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.stamp = stamp
        self.is_wholescene = is_wholescene
        self.scheduler = StepLR(optimizer, step_size=decay_step, gamma=decay_factor)
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
        

    def _get_scene_list(self, train_path, query_path, random_num=0):
        scene_list = []
        with open(train_path) as f:
            for scene_id in f.readlines():
                scene_list.append(scene_id.strip())
            train_scene_list = random.sample(scene_list,len(scene_list)) # train_list에 있는 모든 scene을 사용함
        scene_list = []
        with open(query_path) as f:
            for scene_id in f.readlines():
                scene_list.append(scene_id.strip())
            query_scene_list = random.sample(scene_list,len(scene_list)) # query_list 있는 모든 scene을 사용함
            
        return train_scene_list, query_scene_list
        


    def __call__(self, args, epoch, verbose, occlusion_round):
        # setting
        self.epoch = epoch
        self.verbose = verbose
        self.occlusion_round = occlusion_round
        train_scene_list, query_scene_list = self._get_scene_list(CONF.SCANNETV2_TRAIN, CONF.SCANNETV2_QUERY,2)
        query_scene_data = np.load(CONF.SCANNETV2_QUERY_FILE.format(query_scene_list[0]))
        #76510개의 포인트 데이터
        new_user_input_list = query_scene_data[25000:32651]
        self.dataset["train"] = ScannetDataset_update_New_TrainData("train", train_scene_list, query_scene_list, new_user_input_list, is_weighting=not args.no_weighting, use_color=args.use_color, use_normal=args.use_normal, use_multiview=args.use_multiview)
            
        self.dataloader["train"] = DataLoader(self.dataset["train"], batch_size=args.batch_size, collate_fn=collate_random, num_workers=args.num_workers, pin_memory=True)

        self._total_iter["train"] = len(self.dataloader["train"]) * epoch
        self._total_iter["val"] = len(self.dataloader["val"]) * epoch
        self.args = args
        self.dataset["val"].generate_chunks()
        for epoch_id in range(1):
            print("epoch {} starting...".format(epoch_id + 1))
            
            # generate new chunks
            # 매 epoch마다 train 직전에 50000개(dataset.npoints개)의 점을 랜덤하게 뽑아냄
            self.dataset["train"].generate_chunks() # train data에서 8192개의 점을 랜덤하게 추출함
            #self.dataset["val"].generate_chunks()
            
            

            
            # train
            self._set_phase("train")
            # train loader를 입력으로 해서 _train 함수를 돌릴 때, iterate 하게 되어 위에서 만든 chunk 값에 접근하게 됨 (__call 함수 참조 >> dataset.py)
            self._train(self.dataloader["train"], epoch_id)

            # val
            self._set_phase("eval")
            self._val(self.dataloader["val"], epoch_id)

            # epoch report
            
            self._epoch_report(epoch_id)

            # load tensorboard
            self._dump_log(epoch_id)

            # scheduler
            self.scheduler.step() # 학습과정에서 learning rate를 조정하는 scheduler를 통해서, 더 정확한 학습률 조절을 수행할 수 있다. (고정된 값을 사용하는 것이 아니라)
            
            '''
            summary = SummaryWriter()
            if epoch_id % 5 == 0:    # 매 10 iteration마다 업데이트
              print("와아아")
              print(self.log["train"][epoch_id]["loss"])
              print(epoch_id)
              print("-------")
              summary.add_scalar('D_loss_adv', self.log["train"][epoch_id]["loss"], epoch_id)
              summary.add_scalar('D_loss_cls', self.log["val"][epoch_id]["loss"], epoch_id)
            '''
        # print best
        self._best_report()

        # save model
        # 가장 처음에 지정된 2개의 scene을 이용하여 학습이 수행된 모델을 저장한다. 
        print("saving last models...\n")
        model_root = os.path.join(CONF.OUTPUT_ROOT, self.stamp)
        #print("모델 이름은요 : ")
        #print(self.model.__class__.__name__)
        torch.save(self.model.state_dict(), os.path.join(model_root, "model_last.pth"))
        

        # export
        self._log_writer.export_scalars_to_json(os.path.join(CONF.OUTPUT_ROOT, self.stamp, "tensorboard", "all_scalars.json"))
        # 아래 세 줄 주석처리했어염 ㅎㅅㅎ 다시 수정해서 주석 풀어놓으세요
        saved_file_root = os.path.join(model_root, "model_last.pth")
        saved_parameters = torch.load(saved_file_root, map_location=lambda storage, loc:storage.cuda(0))
        self.model.load_state_dict(saved_parameters)

        # 여기서 query scene에 대해서 random하게 50000개의 포인트를 추출함
        self.dataset["query"].generate_chunks()
        self.dataset["query"]._infront_points(occlusion_round=1)
        

        # 여기서부터 confidence에 따른 active learning 수행
        ######################################################
        # epoch만큼 학습이 끝나고 나면, pred를 통해 user에게서 labeled data를 추가로 얻는다. 이 데이터를 가지고 solver를 다시 호출하여, 위에서 다시 학습을 수행하게 함. 
        # query의 대상이 되는 scene 1개에 대해서 class 별로 pred의 confidence를 계산한다. 
        # active_epoch는 한 번 active learning 실행 될 때 몇 번 데이터를 입력받을지, default는 3으로 solver 함수 호출할 때 parameter 전달받음
        # epoch는 총 몇 번 active learning 수행할지
        new_user_input_list = []
 
        

        for active_epoch_id in range(3):
          self.model.eval()
          # 라벨 별 confidence를 뽑아서 가장 confidence가 낮은 label을 알아낸다.  [[label],[confidence]]
          train_scene_list, query_scene_list = self._get_scene_list(CONF.SCANNETV2_TRAIN, CONF.SCANNETV2_QUERY,2)
          query_scene_data = np.load(CONF.SCANNETV2_QUERY_FILE.format(query_scene_list[0]))
          # 사실 상 바로 아래에 있는 코드 한 줄은 필요없음, , 
          train_scene_data = np.load(CONF.SCANNETV2_QUERY_FILE.format(train_scene_list[0]))
          self.query_scene_data_npy = query_scene_data
          self.train_scene_data_npy = train_scene_data
          if active_epoch_id == 0:
            label = 4 #chair
          elif active_epoch_id == 1:
            label = 6 # table
          elif active_epoch_id == 2:
            label = 3 #bed
          else:
            print('please check epoch again.')
          query_label, view = self._confidence_pred_for_AL(self.dataloader["query"],self.query_scene_data_npy,label)
          
              # user에게 특정 label에 대한 query를 받는다. 
              # window를 annotation 해주세요. 
          
          # 위의 for문에서 받은 new_user_input_list를 아래에 넣음.
          
          #new_user_input_list = [[1,2,3,4,5,6,7,8,9,10,11],[11,10,9,8,7,6,5,4,3,2,1]]
          new_input="a"
          
          
          
          print("추천된 뷰들은 순서대로 ", view, " 번째 뷰 입니다.")
          #print(query_scene_data[0:6,:])

          
          print(CONF.NYUCLASSES[label]+" 에 해당하는 점을 query scene에서 annotation 해주세요. (annotation.txt) ")
          final = "3"
          print("입력을 완료하면 0을 입력하세요.")
          while(final!="0"):
            final = input()
          
          f = open("/content/drive/MyDrive/대학원/연구/New_Davedrum/Pointnet2.ScanNet/user_input/letters_new.txt","r")
          line = "start"
          while line:
            line = f.readline()
            print(line)
            new_input = line.split(",")
            print(new_input)
            
            
            #print(new_input)
            #print(len(new_input))
            if (len(new_input)==1):
              
              break
            new_input = list(map(float, new_input)) # [1., 2., 3.]
            
            print("일단 찍어봐 : ", new_input) 
            point_index_query_x = np.where(np.round(query_scene_data[:,0],6) == round(float(new_input[0]), 6))
                
            point_index_query_x = list(point_index_query_x)[0]
            print(point_index_query_x)
            point_index_query_y = np.where(np.round(query_scene_data[:,1],6) == round(float(new_input[1]), 6))
            point_index_query_y = list(point_index_query_y)[0]
            print(point_index_query_y)
            
            point_index_query_z = np.where(np.round(query_scene_data[:,2],6) == round(float(new_input[2]), 6))
            point_index_query_z = list(point_index_query_z)[0]
            print(point_index_query_z)

            point_index_query = list(set(point_index_query_x.tolist()) & set(point_index_query_y.tolist()) & set(point_index_query_z.tolist()))
            print(point_index_query)
            new_input = query_scene_data[point_index_query[0],:10]
            new_input = np.hstack((new_input,[query_label]))
            

            new_user_input_list = new_user_input_list+[new_input.tolist()]
          print('유저의 input은 : ',new_user_input_list)
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
              
            

          
          
          self.dataset["train"] = ScannetDataset_update_New_TrainData("train", train_scene_list, query_scene_list, new_user_input_list, is_weighting=not args.no_weighting, use_color=args.use_color, use_normal=args.use_normal, use_multiview=args.use_multiview)
          
          self.dataloader["train"] = DataLoader(self.dataset["train"], batch_size=args.batch_size, collate_fn=collate_random, num_workers=args.num_workers, pin_memory=True)
          
          
          self._total_iter["train"] = len(self.dataloader["train"]) * epoch
          self._total_iter["val"] = len(self.dataloader["val"]) * epoch
          
          
          print("active learning epoch {} starting...".format(epoch_id + 1))
          
          # generate new chunks
          self.dataset["train"].generate_chunks() 
          self.dataset["val"].generate_chunks()
          self.dataset["query"].generate_chunks()
          #print(self.dataset["train"].chunk_data["scene0003_00"])
          #print(len(self.dataset["train"].chunk_data["scene0003_00"]))
          
          
          # train
          self._set_phase("train")
          self._train(self.dataloader["train"], epoch_id)

          # val
          self._set_phase("eval")
          self._val(self.dataloader["val"], epoch_id)

          # epoch report
          self._epoch_report(epoch_id)

          # load tensorboard
          self._dump_log(epoch_id)

          # scheduler
          self.scheduler.step() # 학습과정에서 learning rate를 조정하는 scheduler를 통해서, 더 정확한 학습률 조절을 수행할 수 있다. (고정된 값을 사용하는 것이 아니라)

          # print best
          self._best_report()

          # save model
          # 가장 처음에 지정된 2개의 scene을 이용하여 학습이 수행된 모델을 저장한다. 
          print("saving last models...\n")
          model_root = os.path.join(CONF.OUTPUT_ROOT, self.stamp)
          torch.save(self.model.state_dict(), os.path.join(model_root, "model_last.pth"))
          

          # export
          self._log_writer.export_scalars_to_json(os.path.join(CONF.OUTPUT_ROOT, self.stamp, "tensorboard", "all_scalars.json"))
          saved_file_root = os.path.join(model_root, "model_last.pth")
          saved_parameters = torch.load(saved_file_root, map_location=lambda storage, loc:storage.cuda(0))
          self.model.load_state_dict(saved_parameters)
          
    def _formatting_4(self, n):
      return np.round(n,4)        
            
                

    #forward 함수와 알고리즘 같음
    def _confidence_pred_for_AL(self, query_loader,query_scene_data_npy, which_label):
        #confidence ranking이 가장 낮은 label // 그 label이 가장 많이 unoccluded 인 평면 3개를 return함
        phase = "query"
        for iter_id, data in enumerate(query_loader): # query scene 하나니까, for 문 한 번만 돎.
            
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
            print("좌표는 ", coords.dtype)
            print(coords.shape)
            print("특징은 ", feats.dtype)
            print(feats.shape)
            print("seg는 ", semantic_segs.dtype)
            print(semantic_segs.shape)
            '''
            

            #coords : torch.float32, torch.Size([1, 76510, 3])
            #feats : torch.float32, torch.Size([1, 76510, 3])
            #segs : torch.int64, torch.Size([1, 76510])
            
            # 아래 4줄 주석처리함 ㅎㅅㅎ 주석 다시 푸세요.
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
            print("좌표는 ", coords_2.dtype)
            print(coords_2.shape)
            
            print("특징은 ", feats_2.dtype)
            
            print(feats_2.shape)
            print("seg는 ", semantic_segs_2.dtype)
            print(semantic_segs_2.shape)      
            # 1.2000e+02,  9.4000e+01,  6.8000e+01, -1.2915e-01, -8.4367e-01,-5.2109e-01
            
            print(coords[0][:3])
            print(coords_2[0][:3]) 
            print(feats[0][:3])
            print(feats_2[0][:3])  
            print(semantic_segs[0][:3])
            print(semantic_segs_2[0][:3]) 
            
            
            #print("좌표는 ", coords.dtype)
            
            #print("특징은 ", feats.dtype)
            #print("seg는 ", semantic_segs.dtype)
            #print("weight는 ", sample_weights.dtype)
            '''

            coords, feats, semantic_segs, sample_weights = coords.cuda(), feats.cuda(), semantic_segs.cuda(), sample_weights.cuda()
            #self.log[phase][epoch_id]["fetch"].append(fetch_time)
            #print("coords의 길이는 : ", len(coords[0]))
            # forward
            # 검증할 때도, forward 한 번 호출
            # query npy 중 8192개의 점에 대한 예측값
            preds = self._forward(coords, feats, self.is_wholescene)
            
            """
              coords[0] = tensor([[0.9362, 2.1084, 1.1275],
                          [0.3519, 2.2397, 1.9149],
                          [0.4654, 1.8777, 0.0994],
                          ...,
                          [0.6213, 1.5378, 0.9377],
                          [0.1902, 1.9552, 1.9303],
                          [0.5116, 2.1635, 1.9074]], device='cuda:0')
              ==> 8192개 점의 좌표

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
            # 8192 개의 점
            coordinate_list = coords[0].cpu().numpy().tolist()
            
            
            # N번째 평면의 N번째 포인트의 첫번째 좌표
            # ==> array([1.09800005, 0.59128624, 0.77000004])



            unOccluded_label = []
            # unOccluded_label 생성
            #[[0, 2, 3, 6, 7, 9, 11, 14, 15,..., ], [8, 46, 71, ...], ..., []]
            
            for k in range(14):
              print(k,"----> 0 ~ 13 중에서")
              #unOccluded_label \ [[], [], [], [], [], [], [], [], [], [], [], [], [], []]
              unOccluded_label.append([])
              # coordinate_unOccluded는 각 평면에서 가장 앞에 보이는 point들의 좌표
              # occlusion_round = 8로 해도 되는지는 일단 확인해봐야 함
              
              coordinate_unOccluded = [(np.round(x[0],1) for x in list(self.dataset["query"].infront_point[k].values()))]
              print(self.dataset["query"].infront_point[k].values())
              
              coordinate_unOccluded = [list(x) for x in coordinate_unOccluded[0]]
              
              print("===")
              print(coordinate_unOccluded)
              print("===")

              # len(coordinate_list) == 76510개
              for n in range(len(coordinate_list)):
                # 소수점 아래 첫째 자리까지 해도되는지 numpy 길이 상으로 0.1 정도가 몇 cm인지 알아보기
                if (n<=5):
                    print("평면 k에서는")
                    print(np.round(coordinate_list[n],1).tolist())
                if ((np.round(coordinate_list[n],1).tolist()) in coordinate_unOccluded):
                  #print(n)
                  #print(coordinate_list[n])
                  
                  unOccluded_label[-1].append(n)
                  # occlusion 되지 않은 포인트 중 예측값을 가진 포인트들의 인덱스 (coords 상에서의 인덱스)
              # print(unOccluded_label)
              # unOccluded_label는 각 평면에 대해서 pred 값이 있으며, occlusion이 없는 가장 앞에 있는 포인트의 index
              # unOccluded_label ==> [[1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 20, 21, 22, 27, 28, 29, 30, 32, 33, 34, 36, 37, 39, 40, 41, 44, 47, 48, 49, 51, 52, 55, 57, 58, 59, 60, 61, 63, 64, 68, 70, 71, 74, 76, 77, 78, 81, 84, 85, 87, 88, 89, 90, 91, 92, 95, 96, 97, 98, 99, 100, 103, 106, 110, 111, 112, 114, 115, 116, 117, 122, 123, 124, 125, 126, 128, 129, 130, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 144, 145, 146, 147, 150, 153, 154, 156, 157, 158, 159, 162, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 178, 179, 180, 181, 182, 183, 184, 185, 186, 188, 189, 190, 192, 194, 196, 197, 198, 199, 200, 201, 202, 203, 206, 208, 209, 210, 212, 214, 215, 216, 218, 219, 222, 223, 226, 227, 228, 230, 231, 232, 233, 234, 235, 236, 237, 238, 240, 241, 242, 244, 246, 247, 248, 249, 253, 254, 255, 256, 260, 262, 263, 265, 268, 269, 270, 271, 272, 273, 275, 276, 277, 280, 281, 284, 285, 286, 287, 289, 291, 292, 294, 296, 297, 298, 299, 300, 301, 303, 304, 305, 306, 307, 308, 311, 312, 313, 314, 316, 318, 319, 321, 322, 323, 324]]
            print("unocclusded_label은 : (각 14개의 시점마다 겹치지 않는 점들의 번호는")
            print(unOccluded_label)
            for i in range(len(unOccluded_label)):
              print('개수는 ')
              print(len(unOccluded_label[i]))

            """
              for n in range(len(list(self.dataset["query"].infront_point[k].keys()))):
                # self.dataset["query"].infront_point[0][list(self.dataset["query"].infront_point[0].keys())[n]][0] ==> [1.05492127 2.73373747 1.1574291 ]
                coordinate_unOccluded = self.dataset["query"].infront_point[0][list(self.dataset["query"].infront_point[0].keys())[n]][0]
                index_unOccluded = np.where(coordinate_list == coordinate_unOccluded)
                if (index_unOccluded[0].size!=0) :
                  unOccluded_label[-1].append((coordinate_unOccluded,index_unOccluded[0][0]))
                # index_unOccluded = (array([5730, 5730, 5730, 7193, 7193, 7193]), array([0, 1, 2, 0, 1, 2])) 
                # index_unOccluded = (array([]), array([])) -> 당연히 8192개 중에서는 occluded (전체 포인트에 대한 계산값) point 없을 수도 있음

                # 5730, 7193 번째의 포인트임
                # 중복을 허용하고, 더 앞쪽 인덱스의 예측값을 가져옴 (위에서는 5730번째 인덱스)
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
            # 각 라벨로 예측된 확률
            # points_list = [[-0.11924917995929718, -0.04613947123289108, -0.20690718293190002, -0.2028222382068634, 0.31167054176330566, -0.08550150692462921, -0.040521204471588135, 0.04196982830762863, 0.029743794351816177, 0.151509091258049, -0.05134827271103859, 0.005890154279768467, 0.22072120010852814, -0.3832728862762451, 0.19867755472660065, -0.13119345903396606, 0.22404149174690247, 0.056744541972875595, 0.09138473868370056, 0.21279717981815338], [0.0028198033105582, 0.0317632257938385, -0.03630537912249565, 0.03719541057944298, -0.0015036762924864888, 0.11752472072839737, 0.23951327800750732, -0.12636196613311768, 0.08242180198431015, -0.023187827318906784, 0.06832046806812286, 0.05200092867016792, -0.03182460740208626, 0.23237772285938263, -0.021574841812253, 0.034832652658224106, 0.04854806140065193, 0.00804646871984005, -0.02367490343749523, 0.24299505352973938], [], ... , []]
            # len(preds[0]) ==> 76510 개 
            
            points_list = preds[0].tolist()
            
            
            # 각 point 별로 예측된 라벨의 인덱스 (ex. 3은 sofa -- 첫번째 포인트에 대해 예측된 라벨, 9는 window ...)
            # Index_for_each_label = [3, 5, 8, ... , 11]
            Index_for_each_label = [x.index(max(x)) for x in points_list]
            print("============예측 결과는============")
            print(len(Index_for_each_label))
            print(Index_for_each_label)
            
            

            # 각 point 별로 예측된 라벨일 확률 (confidence) 
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
            print("총 confidence 평균 점수는 : ",sorting_array)
            print("라벨 별 순위는 : ",confidence_ranking)
            count_label = []
            for j in range(14):
              count_label_list = [Index_for_each_label[x] for x in unOccluded_label[j]]
              print("count_label_list의 개수는 ",len(count_label_list))
              print("이 면에서 겹치지않은 포인트의 라벨들은",count_label_list[:10])
              
              #count_label.append(count_label_list.count(int(confidence_ranking[-1])))
              count_label.append(count_label_list.count(int(which_label)))
            print(count_label)
            count_label_array = np.array(count_label)
            print("개수는 : ", count_label_array)
            view_indexes = np.argsort(count_label_array)[::-1]
            print("순서는 : ", view_indexes)
            
            
            #print("라벨링 해야하는 점의 좌표", query_scene_data_npy[np.where(query_scene_data_npy[:,10]==confidence_ranking[-1]),:3])
            # sofa로 예측된 것들의 confidence 평균, window, ... 모두 구해서 평균값이 높은 것 순서대로 매김.
            # confidence_ ranking은 오른쪽과 같은 리스트임 [15 10 12  9 19 13  8  6 18 16  3 14  7  1 11  5  4  0 17  2]
            return confidence_ranking[0], view_indexes[:3]

        """
        if self.is_wholescene:
            pred = []
            coord_chunk, feat_chunk = torch.split(coord.squeeze(0), self.batch_size, 0), torch.split(feat.squeeze(0), self.batch_size, 0)
            # squeeze 함수는 차원이 1인 차원을 제거해준다. (ex. (3,1,4)==squeeze==>(3,4))
            # 이 때 매개변수를 0으로 지정했으므로 0차원에 있는 차원 값이 1인 차원을 제거해 준다. (ex. (1,2,4)==squeeze(0)==>(2,4))
            # torch.split(tensor, size, dim)은 차원이 dim인 곳에 있는 tensor를 size만큼 잘라주는 것이다. 
            # 여기서는 coord.squeeze(0)을 batch_size 크기 만큼 dim=0에 있는 tensor를 잘라주는 것이다. 
            assert len(coord_chunk) == len(feat_chunk)
            for coord, feat in zip(coord_chunk, feat_chunk):
                output = self.model(torch.cat([coord, feat], dim=2))
                pred.append(output)

            pred = torch.cat(pred, dim=0).unsqueeze(0) 
            # pred는 여러 개의 리스트로 구성되어 있는 리스트임
            # 그래서 torch.cat(pred, dim=0)은 그냥 scene 안에 있는 모든 포인트에 대한 pred 값을 싹 다 붙인 것임
            # unsqueeze는 지정한 dim(여기서는 0차원)에 size가 1인 빈 공간을 채워주면서 차원을 확장시킨다. (ex. size가 (3,)인 매트릭스에다 unsqueeze(0)을 하면, size가 (1,3)인 매트릭스로 변환됨)

        else:
            output = self.model(torch.cat([coord, feat], dim=2)) # coord 텐서와 feat 텐서를 2차원을 기준으로 붙여줌
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
            # squeeze 함수는 차원이 1인 차원을 제거해준다. (ex. (3,1,4)==squeeze==>(3,4))
            # 이 때 매개변수를 0으로 지정했으므로 0차원에 있는 차원 값이 1인 차원을 제거해 준다. (ex. (1,2,4)==squeeze(0)==>(2,4))
            # torch.split(tensor, size, dim)은 차원이 dim인 곳에 있는 tensor를 size만큼 잘라주는 것이다. 
            # 여기서는 coord.squeeze(0)을 batch_size 크기 만큼 dim=0에 있는 tensor를 잘라주는 것이다. 
            assert len(coord_chunk) == len(feat_chunk)
            for coord, feat in zip(coord_chunk, feat_chunk):
                output = self.model(torch.cat([coord, feat], dim=2))
                pred.append(output)

            pred = torch.cat(pred, dim=0).unsqueeze(0) 
            outputs = pred.max(3)[1]
            
            print("=====예측 일단 이거맞나요?=====")
            print(outputs)
            print("=====예측 일단 이거맞나요?=====")
            # pred는 여러 개의 리스트로 구성되어 있는 리스트임
            # 그래서 torch.cat(pred, dim=0)은 그냥 scene 안에 있는 모든 포인트에 대한 pred 값을 싹 다 붙인 것임
            # unsqueeze는 지정한 dim(여기서는 0차원)에 size가 1인 빈 공간을 채워주면서 차원을 확장시킨다. (ex. size가 (3,)인 매트릭스에다 unsqueeze(0)을 하면, size가 (1,3)인 매트릭스로 변환됨)
        else:
            output = self.model(torch.cat([coord, feat], dim=2)) # coord 텐서와 feat 텐서를 2차원을 기준으로 붙여줌
            pred = output
        return pred

    def _backward(self):
        # optimize
        self.optimizer.zero_grad() # 각 매개변수의 grad 값을 모두 0으로 초기화 하여, 이전 backward의 영향을 받지 않도록
        self._running_log["loss"].backward() # 매개변수의 grad값을 update함
        # self._clip_grad()
        self.optimizer.step() # 매개변수 갱신

    def _compute_loss(self, pred, target, weights):
        num_classes = pred.size(-1)
        loss = self.criterion(pred.contiguous().view(-1, num_classes), target.view(-1), weights.view(-1))
        self._running_log["loss"] = loss

    def _train(self, train_loader, epoch_id):
        # setting
        phase = "train"
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
            # train 할 때 마다 forward 한번 호출
            preds = self._forward(coords, feats, self.is_wholescene)
            self._compute_loss(preds, semantic_segs, sample_weights)
            self._eval(coords, preds, semantic_segs, sample_weights, self.is_wholescene)
            self.log[phase][epoch_id]["forward"].append(time.time() - start_forward)

            # backward
            start = time.time()
            self._backward()
            self.log[phase][epoch_id]["backward"].append(time.time() - start)

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

    def _val(self, val_loader, epoch_id):
        # setting
        phase = "val"
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
            print("============ val 정답지는 =============")
            print((semantic_segs[0][:10]))

            coords, feats, semantic_segs, sample_weights = coords.cuda(), feats.cuda(), semantic_segs.cuda(), sample_weights.cuda()
            self.log[phase][epoch_id]["fetch"].append(fetch_time)

            # forward
            # 검증할 때도, forward 한 번 호출
            preds = self._forward(coords, feats, self.is_wholescene)
            print("========val에 대한 예측값은=========")
            points_list = preds[0].tolist()
            Index_for_each_label = [x.index(max(x)) for x in points_list]
            print(Index_for_each_label[:10])

            self._compute_loss(preds, semantic_segs, sample_weights)
            self._eval(coords, preds, semantic_segs, sample_weights, self.is_wholescene)

            # record log
            self.log[phase][epoch_id]["loss"].append(self._running_log["loss"].item())
            self.log[phase][epoch_id]["point_acc"].append(self._running_log["point_acc"])
            self.log[phase][epoch_id]["point_acc_per_class"].append(self._running_log["point_acc_per_class"])
            self.log[phase][epoch_id]["voxel_acc"].append(self._running_log["voxel_acc"])
            self.log[phase][epoch_id]["voxel_acc_per_class"].append(self._running_log["voxel_acc_per_class"])
            self.log[phase][epoch_id]["point_miou"].append(self._running_log["point_miou"])
            self.log[phase][epoch_id]["voxel_miou"].append(self._running_log["voxel_miou"])

        # check best
        cur_criterion = "voxel_miou"
        cur_best = np.mean(self.log[phase][epoch_id][cur_criterion])
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
            torch.save(self.model.state_dict(), os.path.join(model_root, "model.pth"))

    def _eval(self, coords, preds, targets, weights, is_wholescene):
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
        with open(os.path.join(CONF.OUTPUT_ROOT, self.stamp, "best.txt"), "w") as f:
            f.write(best_report)
    
    