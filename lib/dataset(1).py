import os
import sys
import time
import h5py
import torch
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
from prefetch_generator import background
import random
import math

sys.path.append(".")
from lib.config import CONF


class ScannetDataset():
  #npoints 원래 값은 8192개
    def __init__(self, phase, scene_list, num_classes=21, npoints=50000, is_weighting=True, use_multiview=False, use_color=False, use_normal=False):
        self.phase = phase
        assert phase in ["train", "val", "test", "query"]
        self.scene_list = scene_list
        self.num_classes = num_classes
        self.npoints = npoints
        self.is_weighting = is_weighting
        self.use_multiview = use_multiview
        self.use_color = use_color
        self.use_normal = use_normal
        self.chunk_data = {} # init in generate_chunks()
        self.infront_point = [] # 각 평면 별 occlusion 
        self._prepare_weights()
        ## 아래 줄 테스트 용으로 추가함
        

    def _infront_points(self, occlusion_round):
      one_infront_point = {}
      query_scene_list = "scene0003_00"
      self.occlusion_round = occlusion_round
      scene = self.scene_data[query_scene_list]
      # 전체 데이터에서 100%를 샘플링한다.
      scene_sampling = np.random.choice(len(scene), int(len(scene) * 1) , replace=False)

      # self.infront_point = [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}]
      # 순서대로 
      # x + 1.53962 = 0 (left)
      # x - 6.46038 = 0 (right)
      # y + 2.32631 = 0 (left)
      # y - 5.67369 = 0 (right)
      # z + 2.51469 = 0 (left)
      # z - 5.48531 = 0 (right)
      # x - y + 4.87016 = 0 (left)
      # x - y - 6.44354 = 0 (right)
      # x + y - 9.79092 = 0 (right)
      # x + y + 1.52278 = 0 (left)
      # x - z + 4.68178 = 0 (left)
      # x - z - 6.63192 = 0 (right)
      # x + z - 9.60254 = 0 (right)
      # x + z + 1.71116 = 0 (left)
      for i in range(14):
        self.infront_point.append({})
      
      for i in scene[scene_sampling]: # 샘플링한 데이터에서 point 1개씩 돌아가면서 봄
        # P는 i[:3] ==> (x, y, z)
        # P_new 는 정사영된 좌표
        
        # 1번째 평면 : x + 1.53962 = 0 (left)
        #P_new = (-1.53962, i[1], i[2])
        #distance = abs(i[0]-(-1.53962))
        P_new_1 = (round(-1.53962,occlusion_round), round(i[1],occlusion_round), round(i[2],occlusion_round))
        distance = abs(i[0]-round(-1.53962,occlusion_round))
        if ((P_new_1 in self.infront_point[0]) and (self.infront_point[0][P_new_1][1] <= distance)):
          continue 
        else:
          self.infront_point[0][P_new_1] = [i[:3], distance]
         
        # 2번째 평면 : x - 6.46038 = 0 (right)
        #P_new = (6.46038, i[1], i[2])
        #distance = abs(i[0]-(6.46038))
        P_new_2 = (round(6.46038, occlusion_round), round(i[1],occlusion_round), round(i[2],occlusion_round))
        distance = abs(i[0]-round(6.46038, occlusion_round))
        if ((P_new_2 in self.infront_point[1]) and (self.infront_point[1][P_new_2][1] <= distance)):
          continue 
        else:
          self.infront_point[1][P_new_2] = [i[:3], distance]
      
        # 3번째 평면 : y + 2.32631 = 0 (left)
        #P_new = (i[0], -2.32631, i[2])
        #distance = abs(i[1]-(-2.32631))
        P_new_3 = (round(i[0],occlusion_round), round(-2.32631,occlusion_round), round(i[2],occlusion_round))
        distance = abs(i[1]-round(-2.32631,occlusion_round))
        if ((P_new_3 in self.infront_point[2]) and (self.infront_point[2][P_new_3][1] <= distance)):
          continue 
        else:
          self.infront_point[2][P_new_3] = [i[:3], distance]

        # 4번째 평면 : y - 5.67369 = 0 (right)
        #P_new = (i[0], 5.67369, i[2])
        #distance = abs(i[1]-(5.67369))
        P_new_4 = (round(i[0],occlusion_round), round(5.67369,occlusion_round), round(i[2],occlusion_round))
        distance = abs(i[1]-round(5.67369,occlusion_round))
        if ((P_new_4 in self.infront_point[3]) and (self.infront_point[3][P_new_4][1] <= distance)):
          continue 
        else:
          self.infront_point[3][P_new_4] = [i[:3], distance]

        # 5번째 평면 : z + 2.51469 = 0 (left)
        #P_new = (i[0,], i[1], -2.51469)
        #distance = abs(i[2]-(-2.51469))
        P_new_5 = (round(i[0],occlusion_round), round(i[1],occlusion_round), round(-2.51469,occlusion_round))
        distance = abs(i[2]-round(-2.51469,occlusion_round))
        if ((P_new_5 in self.infront_point[4]) and (self.infront_point[4][P_new_5][1] <= distance)):
          continue 
        else:
          self.infront_point[4][P_new_5] = [i[:3], distance]
        
        # 6번째 평면 : z - 5.48531 = 0 (right)
        #P_new = (i[0], i[1], 5.48531)
        #distance = abs(i[2]-(5.48531))
        P_new_6 = (round(i[0],occlusion_round), round(i[1],occlusion_round), round(5.48531, occlusion_round))
        distance = abs(i[2]-round(5.48531, occlusion_round))
        if ((P_new_6 in self.infront_point[5]) and (self.infront_point[5][P_new_6][1] <= distance)):
          continue 
        else:
          self.infront_point[5][P_new_6] = [i[:3], distance]
        
        # 7번째 평면 : x - y + 4.87016 = 0 (left)
        #P_new = (i[0] -0.5 * abs(i[0]-i[1]+4.87016) , i[1] +0.5 * abs(i[0]-i[1]+4.87016) ,i[2])
        #distance = math.dist(P_new, i[:3])
        P_new_7 = (round(round(i[0],occlusion_round) -0.5 * abs(round(i[0],occlusion_round)-round(i[1],occlusion_round)+round(4.87016, occlusion_round)),occlusion_round) , round(round(i[1],occlusion_round) +0.5 * abs(round(i[0],occlusion_round)-round(i[1],occlusion_round)+round(4.87016, occlusion_round)),occlusion_round) ,round(i[2],occlusion_round))
        distance = math.dist(P_new_7, i[:3])
        if ((P_new_7 in self.infront_point[6]) and (self.infront_point[6][P_new_7][1] <= distance)):
          continue 
        else:
          self.infront_point[6][P_new_7] = [i[:3], distance]
        
        # 8번째 평면 : x - y - 6.44354 = 0 (right)
        #P_new = (i[0] +0.5 * abs(i[0]-i[1]- 6.44354) , i[1] -0.5 * abs(i[0]-i[1]- 6.44354) ,i[2])
        #distance = math.dist(P_new, i[:3])
        P_new_8 = (round(round(i[0],occlusion_round) +0.5 * abs(round(i[0],occlusion_round)-round(i[1],occlusion_round)- round(6.44354,occlusion_round)),occlusion_round) , round(round(i[1],occlusion_round) -0.5 * abs(round(i[0],occlusion_round)-round(i[1],occlusion_round)- round(6.44354,occlusion_round)),occlusion_round) ,round(i[2],occlusion_round))
        distance = math.dist(P_new_8, i[:3])
        if ((P_new_8 in self.infront_point[7]) and (self.infront_point[7][P_new_8][1] <= distance)):
          continue 
        else:
          self.infront_point[7][P_new_8] = [i[:3], distance]

        # 9번째 평면 : x + y - 9.79092 = 0 (right)
        #P_new = (i[0] +0.5 * abs(i[0]+i[1]- 9.79092) , i[1] + 0.5 * abs(i[0]+i[1]- 9.79092) ,i[2])
        #distance = math.dist(P_new, i[:3])
        P_new_9 = (round(round(i[0],occlusion_round) +0.5 * abs(round(i[0],occlusion_round)+round(i[1],occlusion_round)- round(9.79092,occlusion_round)),occlusion_round) , round(round(i[1],occlusion_round) + 0.5 * abs(round(i[0],occlusion_round)+round(i[1],occlusion_round)- round(9.79092,occlusion_round)),occlusion_round) ,round(i[2],occlusion_round))
        distance = math.dist(P_new_9, i[:3])
        if ((P_new_9 in self.infront_point[8]) and (self.infront_point[8][P_new_9][1] <= distance)):
          continue 
        else:
          self.infront_point[8][P_new_9] = [i[:3], distance]

        # 10번째 평면 : x + y + 1.52278 = 0 (left)
        #P_new = (i[0] - 0.5 * abs(i[0]+i[1] + 1.52278) , i[1] - 0.5 * abs(i[0]+i[1]+ 1.52278) ,i[2])
        #distance = math.dist(P_new, i[:3])
        P_new_10 = (round(round(i[0],occlusion_round) - 0.5 * abs(round(i[0],occlusion_round)+round(i[1],occlusion_round) + round(1.52278, occlusion_round)),occlusion_round) , round(round(i[1],occlusion_round) - 0.5 * abs(round(i[0],occlusion_round)+round(i[1],occlusion_round)+ round(1.52278, occlusion_round)),occlusion_round) ,round(i[2],occlusion_round))
        distance = math.dist(P_new_10, i[:3])
        if ((P_new_10 in self.infront_point[9]) and (self.infront_point[9][P_new_10][1] <= distance)):
          continue 
        else:
          self.infront_point[9][P_new_10] = [i[:3], distance]

        # 11번째 평면 : x - z + 4.68178 = 0 (left)
        #P_new = (i[0] - 0.5 * abs(i[0]-i[2]+ 4.68178) , i[1], i[2] + 0.5 * abs(i[0]-i[2]+ 4.68178))
        #distance = math.dist(P_new, i[:3])
        P_new_11 = (round(round(i[0],occlusion_round) - 0.5 * abs(round(i[0],occlusion_round)-round(i[2],occlusion_round)+ round(4.68178, occlusion_round)),occlusion_round) , round(i[1],occlusion_round), round(round(i[2],occlusion_round) + 0.5 * abs(round(i[0],occlusion_round)-round(i[2],occlusion_round)+ round(4.68178, occlusion_round)),occlusion_round))
        distance = math.dist(P_new_11, i[:3])
        if ((P_new_11 in self.infront_point[10]) and (self.infront_point[10][P_new_11][1] <= distance)):
          continue 
        else:
          self.infront_point[10][P_new_11] = [i[:3], distance]

        # 12번째 평면 : x - z - 6.63192 = 0 (right)
        #P_new = (i[0] + 0.5 * abs(i[0]-i[2]- 6.63192) , i[1], i[2] - 0.5 * abs(i[0]-i[2]- 6.63192))
        #distance = math.dist(P_new, i[:3])
        P_new_12 = (round(round(i[0],occlusion_round) + 0.5 * abs(round(i[0],occlusion_round)-round(i[2],occlusion_round)- round(6.63192, occlusion_round)), occlusion_round) , round(i[1],occlusion_round), round(round(i[2],occlusion_round) - 0.5 * abs(round(i[0],occlusion_round)-round(i[2],occlusion_round)- round(6.63192, occlusion_round)),occlusion_round))
        distance = math.dist(P_new_12, i[:3])
        if ((P_new_12 in self.infront_point[11]) and (self.infront_point[11][P_new_12][1] <= distance)):
          continue 
        else:
          self.infront_point[11][P_new_12] = [i[:3], distance]
        
        # 13번째 평면 : x + z - 9.60254 = 0 (right)
        #P_new = (i[0] + 0.5 * abs(i[0]+i[2]- 9.60254) , i[1], i[2] + 0.5 * abs(i[0]+i[2]- 9.60254))
        #distance = math.dist(P_new, i[:3])
        P_new_13 = (round(round(i[0],occlusion_round) + 0.5 * abs(round(i[0],occlusion_round)+round(i[2],occlusion_round)- round(9.60254, occlusion_round)),occlusion_round) , round(i[1],occlusion_round), round(round(i[2],occlusion_round) + 0.5 * abs(round(i[0],occlusion_round)+round(i[2],occlusion_round)- round(9.60254, occlusion_round)),occlusion_round))
        distance = math.dist(P_new_13, i[:3])
        if ((P_new_13 in self.infront_point[12]) and (self.infront_point[12][P_new_13][1] <= distance)):
          continue 
        else:
          self.infront_point[12][P_new_13] = [i[:3], distance]
        
        # 14번째 평면 : x + z + 1.71116 = 0 (left)
        #P_new = (i[0] - 0.5 * abs(i[0]+i[2]+ 1.71116) , i[1], i[2] - 0.5 * abs(i[0]+i[2]+ 1.71116))
        #distance = math.dist(P_new, i[:3])
        P_new_14 = (round(round(i[0],occlusion_round) - 0.5 * abs(round(i[0],occlusion_round)+round(i[2],occlusion_round)+ round(1.71116, occlusion_round)),occlusion_round) , round(i[1],occlusion_round), round(round(i[2],occlusion_round) - 0.5 * abs(round(i[0],occlusion_round)+round(i[2],occlusion_round)+ round(1.71116, occlusion_round)),occlusion_round))
        distance = math.dist(P_new_14, i[:3])
        if ((P_new_14 in self.infront_point[13]) and (self.infront_point[13][P_new_14][1] <= distance)):
          continue 
        else:
          self.infront_point[13][P_new_14] = [i[:3], distance]

      
      for i in range(len(self.infront_point)):
        print("평면 ", i, " : ",len(self.infront_point[i]))
        # occlusion이 일어나지 않은 점들의 개수
        
      
      
      # self.infront_point = [{(3,5,7):[array([1.397840, 2.373145, 0.0628]), 2.75401136],[~], ..., [~]},{[~], ... ,[~]}]
      # {각 평면마다 정사영된 좌표 : 해당 평면 기준으로 가장 앞에 있는 점의 실제 좌표, 평면까지의 거리}
      
      #print("일곱번째 평면의 앞에 있는 첫번째 좌표는 ==> ",next(iter(self.infront_point[6])))
      #print("일곱번째 평면의 좌표 전체들은 ==> ",self.infront_point[6])
  

    def _prepare_weights(self):
        self.scene_data = {}
        self.multiview_data = {}
        scene_points_list = []
        semantic_labels_list = []
        if self.use_multiview:
            multiview_database = h5py.File(CONF.MULTIVIEW, "r", libver="latest")
        for scene_id in tqdm(self.scene_list):
            scene_data = np.load(CONF.SCANNETV2_FILE.format(scene_id))
            label = scene_data[:, 10]

            # append
            scene_points_list.append(scene_data)
            semantic_labels_list.append(label)
            self.scene_data[scene_id] = scene_data

            if self.use_multiview:
                feature = multiview_database.get(scene_id)[()]
                self.multiview_data[scene_id] = feature

        if self.is_weighting:
            labelweights = np.zeros(self.num_classes)
            for seg in semantic_labels_list:
                tmp,_ = np.histogram(seg,range(self.num_classes + 1))
                labelweights += tmp
            labelweights = labelweights.astype(np.float32)
            labelweights = labelweights/np.sum(labelweights)
            self.labelweights = 1/np.log(1.2+labelweights)
        else:
            self.labelweights = np.ones(self.num_classes)

    @background()
    def __getitem__(self, index):
        start = time.time()
        
        # load chunks
        scene_id = self.scene_list[index]
        scene_data = self.chunk_data[scene_id]


        # unpack
        point_set = scene_data[:, :3] # include xyz by default
        rgb = scene_data[:, 3:6] / 255. # normalize the rgb values to [0, 1]
        normal = scene_data[:, 6:9]
        label = scene_data[:, 10].astype(np.int32)
        if self.use_multiview:
            feature = scene_data[:, 11:]
            point_set = np.concatenate([point_set, feature], axis=1)
        
        if self.use_color:
            point_set = np.concatenate([point_set, rgb], axis=1)

        if self.use_normal:
            point_set = np.concatenate([point_set, normal], axis=1)

        if self.phase == "train":
            point_set = self._augment(point_set)
        
        # prepare mask
        curmin = np.min(point_set, axis=0)[:3]
        curmax = np.max(point_set, axis=0)[:3]
        mask = np.sum((point_set[:, :3] >= (curmin - 0.01)) * (point_set[:, :3] <= (curmax + 0.01)), axis=1) == 3
        sample_weight = self.labelweights[label]
        sample_weight *= mask

        fetch_time = time.time() - start

        return point_set, label, sample_weight, fetch_time

    def __len__(self):
        return len(self.scene_list)

    def _augment(self, point_set):
        # translate the chunk center to the origin
        center = np.mean(point_set[:, :3], axis=0)
        coords = point_set[:, :3] - center

        p = np.random.choice(np.arange(0.01, 1.01, 0.01), size=1)[0]
        if p < 1 / 8:
            # random translation
            coords = self._translate(coords)
        elif p >= 1 / 8 and p < 2 / 8:
            # random rotation
            coords = self._rotate(coords)
        elif p >= 2 / 8 and p < 3 / 8:
            # random scaling
            coords = self._scale(coords)
        elif p >= 3 / 8 and p < 4 / 8:
            # random translation
            coords = self._translate(coords)
            # random rotation
            coords = self._rotate(coords)
        elif p >= 4 / 8 and p < 5 / 8:
            # random translation
            coords = self._translate(coords)
            # random scaling
            coords = self._scale(coords)
        elif p >= 5 / 8 and p < 6 / 8:
            # random rotation
            coords = self._rotate(coords)
            # random scaling
            coords = self._scale(coords)
        elif p >= 6 / 8 and p < 7 / 8:
            # random translation
            coords = self._translate(coords)
            # random rotation
            coords = self._rotate(coords)
            # random scaling
            coords = self._scale(coords)
        else:
            # no augmentation
            pass

        # translate the chunk center back to the original center
        coords += center
        point_set[:, :3] = coords

        return point_set

    def _translate(self, point_set):
        # translation factors
        x_factor = np.random.choice(np.arange(-0.5, 0.501, 0.001), size=1)[0]
        y_factor = np.random.choice(np.arange(-0.5, 0.501, 0.001), size=1)[0]
        z_factor = np.random.choice(np.arange(-0.5, 0.501, 0.001), size=1)[0]

        coords = point_set[:, :3]
        coords += [x_factor, y_factor, z_factor]
        point_set[:, :3] = coords

        return point_set

    def _rotate(self, point_set):
        coords = point_set[:, :3]

        # x rotation matrix
        theta = np.random.choice(np.arange(-5, 5.001, 0.001), size=1)[0] * 3.14 / 180 # in radians
        Rx = np.array(
            [[1, 0, 0],
             [0, np.cos(theta), -np.sin(theta)],
             [0, np.sin(theta), np.cos(theta)]]
        )

        # y rotation matrix
        theta = np.random.choice(np.arange(-5, 5.001, 0.001), size=1)[0] * 3.14 / 180 # in radians
        Ry = np.array(
            [[np.cos(theta), 0, np.sin(theta)],
             [0, 1, 0],
             [-np.sin(theta), 0, np.cos(theta)]]
        )

        # z rotation matrix
        theta = np.random.choice(np.arange(-5, 5.001, 0.001), size=1)[0] * 3.14 / 180 # in radians
        Rz = np.array(
            [[np.cos(theta), -np.sin(theta), 0],
             [np.sin(theta), np.cos(theta), 0],
             [0, 0, 1]]
        )

        # rotate
        R = np.matmul(np.matmul(Rz, Ry), Rx)
        coords = np.matmul(R, coords.T).T

        # dump
        point_set[:, :3] = coords

        return point_set

    def _scale(self, point_set):
        # scaling factors
        factor = np.random.choice(np.arange(0.95, 1.051, 0.001), size=1)[0]

        coords = point_set[:, :3]
        coords *= [factor, factor, factor]
        point_set[:, :3] = coords

        return point_set

    def generate_chunks(self):
        """
            note: must be called before training
        """

        print("generate new chunks for {}...".format(self.phase))
        for scene_id in tqdm(self.scene_list):
            
            scene = self.scene_data[scene_id]
            ## 아래 줄 테스트 용으로 추가함
            #self._infront_points()
            
            
            semantic = scene[:, 10].astype(np.int32)
            if self.use_multiview:
                feature = self.multiview_data[scene_id]

            coordmax = np.max(scene, axis=0)[:3]
            coordmin = np.min(scene, axis=0)[:3]
            
            for _ in range(5):
                curcenter = scene[np.random.choice(len(semantic), 1)[0],:3]
                curmin = curcenter-[0.75,0.75,1.5]
                curmax = curcenter+[0.75,0.75,1.5]
                curmin[2] = coordmin[2]
                curmax[2] = coordmax[2]
                curchoice = np.sum((scene[:, :3]>=(curmin-0.2))*(scene[:, :3]<=(curmax+0.2)),axis=1)==3
                cur_point_set = scene[curchoice]
                cur_semantic_seg = semantic[curchoice]
                if self.use_multiview:
                    cur_feature = feature[curchoice]

                if len(cur_semantic_seg)==0:
                    continue

                mask = np.sum((cur_point_set[:, :3]>=(curmin-0.01))*(cur_point_set[:, :3]<=(curmax+0.01)),axis=1)==3
                vidx = np.ceil((cur_point_set[mask,:3]-curmin)/(curmax-curmin)*[31.0,31.0,62.0])
                vidx = np.unique(vidx[:,0]*31.0*62.0+vidx[:,1]*62.0+vidx[:,2])
                isvalid = np.sum(cur_semantic_seg>0)/len(cur_semantic_seg)>=0.7 and len(vidx)/31.0/31.0/62.0>=0.02

                if isvalid:
                    break
            
            # store chunk
            if self.use_multiview:
                chunk = np.concatenate([cur_point_set, cur_feature], axis=1)
            else:
                chunk = cur_point_set

            # scene_id 마다 학습되어지는 chunk의 포인트 개수가 모두 같아야 함
            # 그래서 npoints 개 만큼 chunk를 뽑아서 데이터를 학습시킴
            # 다르면 에러남   
            
            #train 
            choices = np.random.choice(chunk.shape[0], chunk.shape[0], replace=True)
            
            #choices = np.random.choice(chunk.shape[0], self.npoints, replace=True)
            
            chunk = chunk[choices]
            self.chunk_data[scene_id] = chunk
        
        print("done!\n")



class ScannetDataset_update_New_TrainData():
    # train data와 query data(2차원 리스트,[[],[]])가 모두 필요함. 
    def __init__(self, phase, train_scene_list, query_scene_list, new_labeled_data, num_classes=21, npoints=50000, is_weighting=True, use_multiview=False, use_color=False, use_normal=False):
        self.phase = phase
        assert phase in ["train", "val", "test", "query"]
        self.train_scene_list = train_scene_list
        self.scene_list = train_scene_list + query_scene_list
        self.num_classes = num_classes
        self.npoints = npoints
        self.is_weighting = is_weighting
        self.use_multiview = use_multiview
        self.use_color = use_color
        self.use_normal = use_normal
        self.chunk_data = {} # init in generate_chunks()
        self.new_labeled_data = new_labeled_data
        self.query_scene_list = query_scene_list
        self._prepare_weights()

    def _prepare_weights(self):
        self.scene_data = {}
        self.multiview_data = {}
        scene_points_list = []
        semantic_labels_list = []
        if self.use_multiview:
            multiview_database = h5py.File(CONF.MULTIVIEW, "r", libver="latest")
        
        # 기존의 train data가 업로드됨
        for scene_id in tqdm(self.train_scene_list):
            scene_data = np.load(CONF.SCANNETV2_FILE.format(scene_id))
            # scene_data는 (100000, 11)의 ndarray 형이다.
            # [[],[],...,[]] 
            label = scene_data[:, 10]

            # append
            scene_points_list.append(scene_data)
            semantic_labels_list.append(label)
            self.scene_data[scene_id] = scene_data

            if self.use_multiview:
                feature = multiview_database.get(scene_id)[()]
                self.multiview_data[scene_id] = feature
        
        # 새로운 new_labeled_data가 업로드됨
        # self.scene_data에 데이터가 저장됨
        
        # 리스트 형태의 업데이트 데이터를 array로 바꿔줌
        new_labeled_data = np.array(self.new_labeled_data)
        
        scene_points_list.append(new_labeled_data)
        semantic_labels_list.append(new_labeled_data[:, 10])
        
        # 이전에 해당 쿼리 scene에서 라벨링된 데이터가 있었다면 (즉, 라벨링이 처음이 아니라면)
        if self.query_scene_list[0] in self.scene_data:
            self.scene_data[self.query_scene_list[0]] = np.append(scene_data[self.query_scene_list[0]],(new_labeled_data))
        else:
            self.scene_data[self.query_scene_list[0]] = new_labeled_data
        if self.use_multiview:
            feature = multiview_database.get(self.query_scene_list[0])[()]
            self.multiview_data[self.query_scene_list[0]] = feature


        if self.is_weighting:
            labelweights = np.zeros(self.num_classes)
            for seg in semantic_labels_list:
                tmp,_ = np.histogram(seg,range(self.num_classes + 1))
                labelweights += tmp
            labelweights = labelweights.astype(np.float32)
            labelweights = labelweights/np.sum(labelweights)
            self.labelweights = 1/np.log(1.2+labelweights)
        else:
            self.labelweights = np.ones(self.num_classes)

    @background()
    def __getitem__(self, index):
        start = time.time()
        
        # load chunks
        
        scene_id = self.scene_list[index]
        #scene_id = scene_id.append(tqdm(self.query_scene_list[0]))
        # 각 scene에서 뽑힌 8192(중복 허용)개의 point들을 모두 scene_data로 함
        scene_data = self.chunk_data[scene_id]

        # unpack
        point_set = scene_data[:, :3] # include xyz by default
        rgb = scene_data[:, 3:6] / 255. # normalize the rgb values to [0, 1]
        normal = scene_data[:, 6:9]
        label = scene_data[:, 10].astype(np.int32)
        if self.use_multiview:
            feature = scene_data[:, 11:]
            point_set = np.concatenate([point_set, feature], axis=1)
        
        if self.use_color:
            point_set = np.concatenate([point_set, rgb], axis=1)

        if self.use_normal:
            point_set = np.concatenate([point_set, normal], axis=1)

        if self.phase == "train":
            point_set = self._augment(point_set)
        
        # prepare mask
        curmin = np.min(point_set, axis=0)[:3]
        curmax = np.max(point_set, axis=0)[:3]
        mask = np.sum((point_set[:, :3] >= (curmin - 0.01)) * (point_set[:, :3] <= (curmax + 0.01)), axis=1) == 3
        sample_weight = self.labelweights[label]
        sample_weight *= mask

        fetch_time = time.time() - start

        return point_set, label, sample_weight, fetch_time

    def __len__(self):
        # 1 is for query data
        return len(self.scene_list)

    def _augment(self, point_set):
        # translate the chunk center to the origin
        center = np.mean(point_set[:, :3], axis=0)
        coords = point_set[:, :3] - center

        p = np.random.choice(np.arange(0.01, 1.01, 0.01), size=1)[0]
        if p < 1 / 8:
            # random translation
            coords = self._translate(coords)
        elif p >= 1 / 8 and p < 2 / 8:
            # random rotation
            coords = self._rotate(coords)
        elif p >= 2 / 8 and p < 3 / 8:
            # random scaling
            coords = self._scale(coords)
        elif p >= 3 / 8 and p < 4 / 8:
            # random translation
            coords = self._translate(coords)
            # random rotation
            coords = self._rotate(coords)
        elif p >= 4 / 8 and p < 5 / 8:
            # random translation
            coords = self._translate(coords)
            # random scaling
            coords = self._scale(coords)
        elif p >= 5 / 8 and p < 6 / 8:
            # random rotation
            coords = self._rotate(coords)
            # random scaling
            coords = self._scale(coords)
        elif p >= 6 / 8 and p < 7 / 8:
            # random translation
            coords = self._translate(coords)
            # random rotation
            coords = self._rotate(coords)
            # random scaling
            coords = self._scale(coords)
        else:
            # no augmentation
            pass

        # translate the chunk center back to the original center
        coords += center
        point_set[:, :3] = coords

        return point_set

    def _translate(self, point_set):
        # translation factors
        x_factor = np.random.choice(np.arange(-0.5, 0.501, 0.001), size=1)[0]
        y_factor = np.random.choice(np.arange(-0.5, 0.501, 0.001), size=1)[0]
        z_factor = np.random.choice(np.arange(-0.5, 0.501, 0.001), size=1)[0]

        coords = point_set[:, :3]
        coords += [x_factor, y_factor, z_factor]
        point_set[:, :3] = coords

        return point_set

    def _rotate(self, point_set):
        coords = point_set[:, :3]

        # x rotation matrix
        theta = np.random.choice(np.arange(-5, 5.001, 0.001), size=1)[0] * 3.14 / 180 # in radians
        Rx = np.array(
            [[1, 0, 0],
             [0, np.cos(theta), -np.sin(theta)],
             [0, np.sin(theta), np.cos(theta)]]
        )

        # y rotation matrix
        theta = np.random.choice(np.arange(-5, 5.001, 0.001), size=1)[0] * 3.14 / 180 # in radians
        Ry = np.array(
            [[np.cos(theta), 0, np.sin(theta)],
             [0, 1, 0],
             [-np.sin(theta), 0, np.cos(theta)]]
        )

        # z rotation matrix
        theta = np.random.choice(np.arange(-5, 5.001, 0.001), size=1)[0] * 3.14 / 180 # in radians
        Rz = np.array(
            [[np.cos(theta), -np.sin(theta), 0],
             [np.sin(theta), np.cos(theta), 0],
             [0, 0, 1]]
        )

        # rotate
        R = np.matmul(np.matmul(Rz, Ry), Rx)
        coords = np.matmul(R, coords.T).T

        # dump
        point_set[:, :3] = coords

        return point_set

    def _scale(self, point_set):
        # scaling factors
        factor = np.random.choice(np.arange(0.95, 1.051, 0.001), size=1)[0]

        coords = point_set[:, :3]
        coords *= [factor, factor, factor]
        point_set[:, :3] = coords

        return point_set

    def generate_chunks(self):
        """
            note: must be called before training
        """

        print("generate new chunks for {}...".format(self.phase))
        
        scene_list_wQuery = self.train_scene_list+self.query_scene_list
        
        for scene_id in scene_list_wQuery:
            scene = self.scene_data[scene_id]
            semantic = scene[:, 10].astype(np.int32)
            if self.use_multiview:
                feature = self.multiview_data[scene_id]

            coordmax = np.max(scene, axis=0)[:3]
            coordmin = np.min(scene, axis=0)[:3]
            
            for _ in range(5):
                curcenter = scene[np.random.choice(len(semantic), 1)[0],:3]
                curmin = curcenter-[0.75,0.75,1.5]
                curmax = curcenter+[0.75,0.75,1.5]
                curmin[2] = coordmin[2]
                curmax[2] = coordmax[2]
                curchoice = np.sum((scene[:, :3]>=(curmin-0.2))*(scene[:, :3]<=(curmax+0.2)),axis=1)==3
                cur_point_set = scene[curchoice]
                cur_semantic_seg = semantic[curchoice]
                if self.use_multiview:
                    cur_feature = feature[curchoice]

                if len(cur_semantic_seg)==0:
                    continue

                mask = np.sum((cur_point_set[:, :3]>=(curmin-0.01))*(cur_point_set[:, :3]<=(curmax+0.01)),axis=1)==3
                vidx = np.ceil((cur_point_set[mask,:3]-curmin)/(curmax-curmin)*[31.0,31.0,62.0])
                vidx = np.unique(vidx[:,0]*31.0*62.0+vidx[:,1]*62.0+vidx[:,2])
                isvalid = np.sum(cur_semantic_seg>0)/len(cur_semantic_seg)>=0.7 and len(vidx)/31.0/31.0/62.0>=0.02

                if isvalid:
                    break
            
            # store chunk
            if self.use_multiview:
                chunk = np.concatenate([cur_point_set, cur_feature], axis=1)
            else:
                chunk = cur_point_set
            
            # 중복도 허용해서 8192개의 포인트가 선택됨
            choices = np.random.choice(chunk.shape[0], self.npoints, replace=True)
            chunk = chunk[choices]
            self.chunk_data[scene_id] = chunk
            #print("scene_id는 : ",scene_id)
        
        print("done!\n")




class ScannetDatasetWholeScene():
    def __init__(self, scene_list, npoints=8192, is_weighting=True, use_color=False, use_normal=False, use_multiview=False):
        self.scene_list = scene_list
        self.npoints = npoints
        self.is_weighting = is_weighting
        self.use_color = use_color
        self.use_normal = use_normal
        self.use_multiview = use_multiview

        self._load_scene_file()

    def _load_scene_file(self):
        self.scene_points_list = []
        self.semantic_labels_list = []
        if self.use_multiview:
            multiview_database = h5py.File(CONF.MULTIVIEW, "r", libver="latest")
            self.multiview_data = []

        for scene_id in tqdm(self.scene_list):
            scene_data = np.load(CONF.SCANNETV2_FILE.format(scene_id))
            label = scene_data[:, 10].astype(np.int32)
            self.scene_points_list.append(scene_data)
            self.semantic_labels_list.append(label)

            if self.use_multiview:
                feature = multiview_database.get(scene_id)[()]
                self.multiview_data.append(feature)

        if self.is_weighting:
            labelweights = np.zeros(CONF.NUM_CLASSES)
            for seg in self.semantic_labels_list:
                tmp,_ = np.histogram(seg,range(CONF.NUM_CLASSES + 1))
                labelweights += tmp
            labelweights = labelweights.astype(np.float32)
            labelweights = labelweights/np.sum(labelweights)
            self.labelweights = 1/np.log(1.2+labelweights)
        else:
            self.labelweights = np.ones(CONF.NUM_CLASSES)

    @background()
    def __getitem__(self, index):
        start = time.time()
        scene_data = self.scene_points_list[index]

        # unpack
        point_set_ini = scene_data[:, :3] # include xyz by default
        color = scene_data[:, 3:6] / 255. # normalize the rgb values to [0, 1]
        normal = scene_data[:, 6:9]

        if self.use_color:
            point_set_ini = np.concatenate([point_set_ini, color], axis=1)

        if self.use_normal:
            point_set_ini = np.concatenate([point_set_ini, normal], axis=1)

        if self.use_multiview:
            multiview_features = self.multiview_data[index]
            point_set_ini = np.concatenate([point_set_ini, multiview_features], axis=1)

        semantic_seg_ini = self.semantic_labels_list[index].astype(np.int32)
        coordmax = point_set_ini[:, :3].max(axis=0)
        coordmin = point_set_ini[:, :3].min(axis=0)
        xlength = 1.5
        ylength = 1.5
        nsubvolume_x = np.ceil((coordmax[0]-coordmin[0])/xlength).astype(np.int32)
        nsubvolume_y = np.ceil((coordmax[1]-coordmin[1])/ylength).astype(np.int32)
        point_sets = list()
        semantic_segs = list()
        sample_weights = list()

        for i in range(nsubvolume_x):
            for j in range(nsubvolume_y):
                curmin = coordmin+[i*xlength, j*ylength, 0]
                curmax = coordmin+[(i+1)*xlength, (j+1)*ylength, coordmax[2]-coordmin[2]]
                mask = np.sum((point_set_ini[:, :3]>=(curmin-0.01))*(point_set_ini[:, :3]<=(curmax+0.01)), axis=1)==3
                cur_point_set = point_set_ini[mask,:]
                cur_semantic_seg = semantic_seg_ini[mask]
                if len(cur_semantic_seg) == 0:
                    continue

                choice = np.random.choice(len(cur_semantic_seg), self.npoints, replace=True)
                point_set = cur_point_set[choice,:] # Nx3
                semantic_seg = cur_semantic_seg[choice] # N
                mask = mask[choice]
                # if sum(mask)/float(len(mask))<0.01:
                #     continue

                sample_weight = self.labelweights[semantic_seg]
                sample_weight *= mask # N
                point_sets.append(np.expand_dims(point_set,0)) # 1xNx3
                semantic_segs.append(np.expand_dims(semantic_seg,0)) # 1xN
                sample_weights.append(np.expand_dims(sample_weight,0)) # 1xN

        point_sets = np.concatenate(tuple(point_sets),axis=0)
        semantic_segs = np.concatenate(tuple(semantic_segs),axis=0)
        sample_weights = np.concatenate(tuple(sample_weights),axis=0)

        fetch_time = time.time() - start

        return point_sets, semantic_segs, sample_weights, fetch_time

    def __len__(self):
        return len(self.scene_points_list)

def collate_random(data):
    '''
    for ScannetDataset: collate_fn=collate_random

    return: 
        coords               # torch.FloatTensor(B, N, 3)
        feats                # torch.FloatTensor(B, N, 3)
        semantic_segs        # torch.FloatTensor(B, N)
        sample_weights       # torch.FloatTensor(B, N)
        fetch_time           # float
    '''

    # load data
    (
        point_set, 
        semantic_seg, 
        sample_weight,
        fetch_time 
    ) = zip(*data)

    # convert to tensor
    point_set = torch.FloatTensor(point_set)
    semantic_seg = torch.LongTensor(semantic_seg)
    sample_weight = torch.FloatTensor(sample_weight)

    # split points to coords and feats
    coords = point_set[:, :, :3]
    feats = point_set[:, :, 3:]

    # pack
    batch = (
        coords,             # (B, N, 3)
        feats,              # (B, N, 3)
        semantic_seg,      # (B, N)
        sample_weight,     # (B, N)
        sum(fetch_time)          # float
    )

    return batch

def collate_wholescene(data):
    '''
    for ScannetDataset: collate_fn=collate_random

    return: 
        coords               # torch.FloatTensor(B, C, N, 3)
        feats                # torch.FloatTensor(B, C, N, 3)
        semantic_segs        # torch.FloatTensor(B, C, N)
        sample_weights       # torch.FloatTensor(B, C, N)
        fetch_time           # float
    '''

    # load data
    (
        point_sets, 
        semantic_segs, 
        sample_weights,
        fetch_time 
    ) = zip(*data)

    # convert to tensor
    point_sets = torch.FloatTensor(point_sets)
    semantic_segs = torch.LongTensor(semantic_segs)
    sample_weights = torch.FloatTensor(sample_weights)

    # split points to coords and feats
    coords = point_sets[:, :, :, :3]
    feats = point_sets[:, :, :, 3:]

    # pack
    batch = (
        coords,             # (B, N, 3)
        feats,              # (B, N, 3)
        semantic_segs,      # (B, N)
        sample_weights,     # (B, N)
        sum(fetch_time)          # float
    )

    return batch