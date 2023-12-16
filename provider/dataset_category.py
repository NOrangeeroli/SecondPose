import os,json
import math
import cv2
import glob
import numpy as np
import _pickle as cPickle
from PIL import Image
from scipy.spatial.transform import Rotation as R
import time

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from data_utils import (
    load_depth,
    load_composed_depth,
    get_bbox,
    fill_missing,
    get_bbox_from_mask,
    rgb_add_noise,
    random_rotate,
    random_scale,
)

defaultTrainconfig = {
   
  'data_dir': '../../data/NOCS',
  'sample_num': 2048,
  'random_rotate': True,
  'angle_range': 20
}

class TrainingDataset(Dataset):
    def __init__(self,
            config, 
            dataset='REAL275',
            mode='ts',
            num_img_per_epoch=-1,
            resolution=64,
            ds_rate=2,
            for_sim_feature = False,
            num_patches = 15,
            category = 0
    ):
        
        np.random.seed(0)
        self.category = category
        assert mode in ['ts','r','sim']
        self.config = config
        self.dataset = dataset
        self.mode = mode
        self.num_img_per_epoch = num_img_per_epoch

        self.resolution = resolution
        self.ds_rate = ds_rate
        self.for_sim_feature = for_sim_feature
        self.num_patches = num_patches
        try: 
            self.sample_num = self.config.sample_num
            self.data_dir = config.data_dir
        except:
            self.sample_num = self.config['sample_num']
            self.data_dir = config['data_dir']
        

        self.invalid_index = []
        syn_img_path = 'camera/train_list.txt'
        self.syn_intrinsics = [577.5, 577.5, 319.5, 239.5]
        self.syn_img_list = [os.path.join(syn_img_path.split('/')[0], line.rstrip('\n'))
                        for line in open(os.path.join(self.data_dir, syn_img_path))]
        #self.syn_img_list = []#CHANGE THIS
        #import pdb;pdb.set_trace()
        print('{} synthetic images are found.'.format(len(self.syn_img_list)))

        syn_category_path = 'camera/train_category_dict.json'
        self.syn_category_dict = json.load(open(os.path.join(self.data_dir, syn_category_path)))
        syn_category_dict_tmp = self.syn_category_dict
        for cls in syn_category_dict_tmp.keys():
                syn_category_dict_tmp[cls] = [[x[0], x[1], 'syn'] for x in syn_category_dict_tmp[cls]]
        self.reference_category_dict = syn_category_dict_tmp

        if self.dataset == 'REAL275':
            real_img_path = 'real/train_list.txt'
            self.real_intrinsics = [591.0125, 590.16775, 322.525, 244.11084]
            self.real_img_list = [os.path.join(real_img_path.split('/')[0], line.rstrip('\n'))
                            for line in open(os.path.join(self.data_dir, real_img_path))]
            print('{} real images are found.'.format(len(self.real_img_list)))
            real_category_path = 'real/train_category_dict.json'
            self.real_category_dict = json.load(open(os.path.join(self.data_dir, real_category_path)))
            real_category_dict_tmp = self.real_category_dict
            for cls in real_category_dict_tmp.keys():
                real_category_dict_tmp[cls] = [[x[0], x[1], 'real'] for x in real_category_dict_tmp[cls]]
            self.reference_category_dict = {cat:self.reference_category_dict[cat] + real_category_dict_tmp[cat] for cat in self.reference_category_dict.keys()}
        
        self.cls_list = sorted(list(self.reference_category_dict.keys()))
        self.xmap = np.array([[i for i in range(640)] for j in range(480)])
        self.ymap = np.array([[j for i in range(640)] for j in range(480)])
        self.sym_ids = [0, 1, 3]    # 0-indexed
        self.norm_scale = 1000.0    # normalization scale
        self.colorjitter = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)
        
        
        self.feature_instance_list = []
        for cls in self.cls_list:
            num_instance = len(self.reference_category_dict[cls])
            
            self.feature_instance_list+=[(cls, i) for i in range(num_instance)]

            
            
            

        if self.num_img_per_epoch != -1:
            self.reset()

    def __len__(self):
        if self.for_sim_feature:
            return len(self.feature_instance_list)

        if self.mode == 'ts':
            if self.num_img_per_epoch == -1:
                if self.dataset == 'REAL275':
                    return len(self.syn_img_list) + len(self.real_img_list)
                else:
                    return len(self.syn_img_list)
            else:
                return self.num_img_per_epoch
        elif self.mode in ['r','sim']:
            if self.num_img_per_epoch == -1:
                num_syn_instance = sum([len(self.syn_category_dict[k]) for k in self.syn_category_dict.keys()])
                    
                if self.dataset == 'REAL275':
                    num_real_instance =  sum([len(self.syn_category_dict[k]) for k in self.syn_category_dict.keys()])
                    return num_syn_instance + num_real_instance
                else:
                    return num_syn_instance
            else:
                return len(self.instance_index)
        
    def reset(self):
        assert self.num_img_per_epoch != -1
        def choice(x, y):
            if x<=y:
                return np.random.choice(x, y)
            else:
                return np.random.choice(x, y, replace = False)
        if self.mode == 'ts':
            if self.dataset == 'REAL275':
                num_syn_img = len(self.syn_img_list)
                num_syn_img, num_real_img = len(self.syn_img_list), len(self.real_img_list)
                num_syn_img_per_epoch = int(self.num_img_per_epoch*0.75)
                #num_syn_img_per_epoch = 0 #CHANGE THIS
                num_real_img_per_epoch = self.num_img_per_epoch - num_syn_img_per_epoch
                syn_img_index = choice(num_syn_img, num_syn_img_per_epoch)
                real_img_index = choice(num_real_img, num_real_img_per_epoch)
                real_img_index = -real_img_index - 1
                self.img_index = np.hstack([syn_img_index, real_img_index])

            else:
                num_syn_img = len(self.syn_img_list)
                num_syn_img_per_epoch = int(self.num_img_per_epoch)
                syn_img_index = choice(num_syn_img, num_syn_img_per_epoch)
                self.img_index = syn_img_index
            #import pdb;pdb.set_trace()
            
            np.random.shuffle(self.img_index)
        elif self.mode in ['r', 'sim']:            

            if self.dataset == 'REAL275':
                num_instance_per_epoch = self.num_img_per_epoch
                syn_category_num_dict = {cat: len(self.syn_category_dict[cat]) 
                                            for cat in self.syn_category_dict.keys()}
                num_syn_instance = sum([len(self.syn_category_dict[k]) for k in self.syn_category_dict.keys()])
                num_syn_instance_per_epoch = int(num_instance_per_epoch*0.75)
                syn_category_ratio = {cat:len(self.syn_category_dict[cat])/ num_syn_instance
                                        for cat in self.syn_category_dict.keys()}
                syn_category_sample_num = {cat: int(num_syn_instance_per_epoch*syn_category_ratio[cat]) 
                                            for cat in syn_category_ratio.keys()}
                syn_instance_index_dict = {cat: choice(syn_category_num_dict[cat], syn_category_sample_num[cat]) 
                                    for cat in self.syn_category_dict.keys()}
                # syn_instance_index = np.concatenate(
                #         [np.concatenate(
                #         (syn_instance_index_dict[cat], np.array([[int(cat)]*syn_instance_index_dict[cat].shape[0]]).T) ,axis = 1)
                #                         for cat in syn_instance_index_dict.keys()],axis = 0
                #     )
                # self.instance_index = syn_instance_index
                
                real_category_num_dict = {cat: len(self.real_category_dict[cat]) 
                                         for cat in self.real_category_dict.keys()}
                
                num_real_instance =  sum([len(self.real_category_dict[k]) for k in self.real_category_dict.keys()])
                
                #num_syn_img_per_epoch = 0 #CHANGE THIS
                num_real_instance_per_epoch = num_instance_per_epoch - num_syn_instance_per_epoch
                real_category_ratio = {cat:len(self.real_category_dict[cat])/ num_real_instance
                                       for cat in self.real_category_dict.keys()}
                real_category_sample_num = {cat: int(num_real_instance_per_epoch* real_category_ratio[cat]) 
                                           for cat in real_category_ratio.keys()}
                
                real_instance_index_dict = {cat: - choice(real_category_num_dict[cat], real_category_sample_num[cat])-1
                                 for cat in self.real_category_dict.keys()}

                instance_index_dict = {cat: np.concatenate([real_instance_index_dict[cat], syn_instance_index_dict[cat]]).reshape(-1,1) for cat in real_instance_index_dict.keys()}
                for cat in instance_index_dict.keys():
                    np.random.shuffle(instance_index_dict[cat])
                
                # real_instance_index = np.concatenate(
                #     [np.concatenate(
                #     (-real_instance_index_dict[cat] - 1, np.array([[int(cat)]*real_instance_index_dict[cat].shape[0]]).T) ,axis = 1)
                #                       for cat in real_instance_index_dict.keys()],axis = 0
                # )
                
                # self.instance_index = np.vstack([syn_instance_index, real_instance_index])
                self.instance_index = np.concatenate(
                    [np.concatenate(
                    (instance_index_dict[cat] , np.array([[int(cat)]*instance_index_dict[cat].shape[0]]).T) ,axis = 1)
                                      for cat in instance_index_dict.keys()],axis = 0
                )
        
            else:
                num_instance_per_epoch = self.num_img_per_epoch
                syn_category_num_dict = {cat: len(self.syn_category_dict[cat]) 
                                            for cat in self.syn_category_dict.keys()}
                num_syn_instance = sum([len(self.syn_category_dict[k]) for k in self.syn_category_dict.keys()])
                num_syn_instance_per_epoch = int(num_instance_per_epoch)
                syn_category_ratio = {cat:len(self.syn_category_dict[cat])/ num_syn_instance
                                        for cat in self.syn_category_dict.keys()}
                syn_category_sample_num = {cat: int(num_syn_instance_per_epoch*syn_category_ratio[cat]) 
                                            for cat in syn_category_ratio.keys()}
                syn_instance_index_dict = {cat: choice(syn_category_num_dict[cat], syn_category_sample_num[cat]).reshape(-1,1)
                                    for cat in self.syn_category_dict.keys()}
                syn_instance_index = np.concatenate(
                        [np.concatenate(
                        (syn_instance_index_dict[cat], np.array([[int(cat)]*syn_instance_index_dict[cat].shape[0]]).T) ,axis = 1)
                                        for cat in syn_instance_index_dict.keys()],axis = 0
                    )
                self.instance_index = syn_instance_index
            
             
            
            np.random.shuffle(self.instance_index)
        else:
            assert False
     

    def __getitem__(self, index):
        if self.for_sim_feature:
            while True:
                cls, idx = self.feature_instance_list[index]

                data_dict =  self._read_instance_from_category_dict(cls, idx)
                if data_dict is None:
                    index +=1
                    self.invalid_index.append(index)
                    continue
                data_dict['index'] = torch.IntTensor([idx]).long()
                data_dict['cls'] = torch.IntTensor([int(cls)]).long()
                return data_dict
                
        
        if self.mode =='ts':
            while True:
                image_index = self.img_index[index]
                data_dict = self._read_instance(image_index)
                #print('READ DATA',index,"/",self.__len__(),time.time()-st)
                if data_dict is None:
                    index = np.random.randint(self.__len__())
                    continue
                return data_dict
        elif self.mode in ['r','sim']:
            while True:
                index, cat = self.instance_index[index]
                data_dict = self._read_instance_cat(index, cat)
                #print('READ DATA',index,"/",self.__len__(),time.time()-st)
                if data_dict is None:
                    index = np.random.randint(self.__len__())
                    continue
                return data_dict
    def _read_instance_cat(self, index,cat):
        assert self.mode in [ 'r','sim']
        def get_data(index):
            if index>=0:
                instance_type = 'syn'
                img_path, instance_id, _ = self.syn_category_dict[str(cat)][index]
                cam_fx, cam_fy, cam_cx, cam_cy = self.syn_intrinsics
            else:
                instance_type = 'real'
                index = -index-1
                
                img_path, instance_id ,_= self.real_category_dict[str(cat)][index]
                cam_fx, cam_fy, cam_cx, cam_cy = self.real_intrinsics
            return self._load_data(img_path,
                                        instance_type, 
                                        cam_cx, cam_cy,cam_fx, cam_fy, instance_id)
        tuple = get_data(index)   
        if tuple is None :
            return None
        pts, rgb, translation, \
        rotation, size, cat_id, asym_flag, \
        rmin, rmax, cmin, cmax, choose, \
            rgb_raw, pts_raw, mask, rand_rotation= tuple


        v = rotation[:,2] / (np.linalg.norm(rotation[:,2])+1e-8)
        rho = np.arctan2(v[1], v[0])
        if v[1]<0:
            rho += 2*np.pi
        phi = np.arccos(v[2])

        vp_rotation = np.array([
            [np.cos(rho),-np.sin(rho),0],
            [np.sin(rho), np.cos(rho),0],
            [0,0,1]
        ]) @ np.array([
            [np.cos(phi),0,np.sin(phi)],
            [0,1,0],
            [-np.sin(phi),0,np.cos(phi)],
        ])
        ip_rotation = vp_rotation.T @ rotation

        rho_label = int(rho / (2*np.pi) * (self.resolution//self.ds_rate))
        phi_label = int(phi/np.pi*(self.resolution//self.ds_rate)) 
        


        ret_dict = {}
        ret_dict['rgb'] = torch.FloatTensor(rgb)
        ret_dict['rgb_raw'] = torch.FloatTensor(rgb_raw)
        ret_dict['pts'] = torch.FloatTensor(pts)
        ret_dict['pts_raw'] = torch.FloatTensor(pts_raw)
        # ret_dict['rmin'] = torch.IntTensor([rmin_first, rmin_second]).long()
        # ret_dict['rmax'] = torch.IntTensor([rmax_first, rmax_second]).long()
        # ret_dict['cmin'] = torch.IntTensor([cmin_first, cmin_second]).long()
        # ret_dict['cmax'] = torch.IntTensor([cmax_first, cmax_second]).long()
        ret_dict['choose'] = torch.IntTensor(choose).long()
        ret_dict['mask'] = torch.IntTensor(mask).long()
        
        ret_dict['category_label'] = torch.IntTensor([cat_id]).long()
        ret_dict['asym_flag'] = torch.FloatTensor([asym_flag])
        ret_dict['translation_label'] = torch.FloatTensor(translation)
        ret_dict['rotation_label'] = torch.FloatTensor(rotation)
        
        ret_dict['size_label'] = torch.FloatTensor(size)

        ret_dict['rho_label'] = torch.IntTensor([rho_label]).long()
        ret_dict['phi_label'] = torch.IntTensor([phi_label]).long()
        ret_dict['vp_rotation_label'] = torch.FloatTensor(vp_rotation)
        ret_dict['ip_rotation_label'] = torch.FloatTensor(ip_rotation)
        ret_dict['rand_rotation'] = torch.FloatTensor(rand_rotation)
        return ret_dict
        

    def _read_pair(self,pair_index):
        assert self.mode in [ 'r','sim']
        index_first, index_second , cat = pair_index
        # assert index_first * index_second >=0
        def get_data(index, cat):
            if index>=0:
                instance_type = 'syn'
                img_path, instance_id, _ = self.syn_category_dict[str(cat)][index]
                cam_fx, cam_fy, cam_cx, cam_cy = self.syn_intrinsics
            else:
                instance_type = 'real'
                index = -index-1
                
                img_path, instance_id ,_= self.real_category_dict[str(cat)][index]
                cam_fx, cam_fy, cam_cx, cam_cy = self.real_intrinsics
            return self._load_data(img_path,
                                        instance_type, 
                                        cam_cx, cam_cy,cam_fx, cam_fy, instance_id)
         

            
        

        tuple_first = get_data(index_first, cat)
        tuple_second = get_data(index_second, cat)
        if tuple_first is None or tuple_second is None:
            return None
        pts_first, rgb_first, translation_first, \
        rotation_first, size_first, cat_id_first, asym_flag_first, \
        rmin_first, rmax_first, cmin_first, cmax_first, choose_first, \
            rgb_raw_first, pts_raw_first, mask_first= tuple_first
        pts_second, rgb_second, translation_second, \
        rotation_second, size_second, cat_id_second, asym_flag_second, \
        rmin_second, rmax_second, cmin_second, cmax_second, choose_second,\
              rgb_raw_second , pts_raw_second, mask_second   = tuple_second
        assert cat_id_first == cat_id_second
        assert asym_flag_first == asym_flag_second
        assert cat_id_first == cat-1
        cat_id = cat_id_first
        asym_flag = asym_flag_first

        # assert  np.isclose([np.linalg.det(rotation_first),1])
        # assert  np.isclose([np.linalg.det(rotation_second),1])

        rotation = rotation_first# @ (rotation_second.T)
        assert np.isclose(np.linalg.det(rotation),1)
        
        
        v = rotation[:,2] / (np.linalg.norm(rotation[:,2])+1e-8)
        rho = np.arctan2(v[1], v[0])
        if v[1]<0:
            rho += 2*np.pi
        phi = np.arccos(v[2])

        vp_rotation = np.array([
            [np.cos(rho),-np.sin(rho),0],
            [np.sin(rho), np.cos(rho),0],
            [0,0,1]
        ]) @ np.array([
            [np.cos(phi),0,np.sin(phi)],
            [0,1,0],
            [-np.sin(phi),0,np.cos(phi)],
        ])
        ip_rotation = vp_rotation.T @ rotation

        rho_label = int(rho / (2*np.pi) * (self.resolution//self.ds_rate))
        phi_label = int(phi/np.pi*(self.resolution//self.ds_rate)) 
        


        ret_dict = {}
        if self.mode == 'sim':

            cos = (np.trace(rotation_first@ (rotation_second.T))-1)/2
            
            assert abs(cos)<=1.0001
            cos = np.clip(cos, -1, 1)
            ret_dict['cos'] = torch.FloatTensor([cos])

            rotation_angle_label  = np.arccos(cos)
            ret_dict['rotation_angle_label'] = torch.FloatTensor([rotation_angle_label])
        
        # def rotate_pts(pts, rotation):
        #     pts_shape = pts.shape
            
        #     return (rotation[None,:,:]@pts.reshape(-1,3)[:,:,None]).squeeze().reshape(pts_shape)
        # pts_first = rotate_pts(pts_first, rotation_second)
        # pts_second = rotate_pts(pts_second, rotation_second)


        ret_dict['rgb'] = torch.FloatTensor(np.stack([rgb_first, rgb_second],axis = 0))
        ret_dict['rgb_raw'] = torch.FloatTensor(np.stack([rgb_raw_first, rgb_raw_second],axis = 0))
        ret_dict['pts'] = torch.FloatTensor(np.stack([pts_first, pts_second],axis = 0))
        ret_dict['pts_raw'] = torch.FloatTensor(np.stack([pts_raw_first, pts_raw_second],axis = 0))
        # ret_dict['rmin'] = torch.IntTensor([rmin_first, rmin_second]).long()
        # ret_dict['rmax'] = torch.IntTensor([rmax_first, rmax_second]).long()
        # ret_dict['cmin'] = torch.IntTensor([cmin_first, cmin_second]).long()
        # ret_dict['cmax'] = torch.IntTensor([cmax_first, cmax_second]).long()
        ret_dict['choose'] = torch.IntTensor(np.stack([choose_first, choose_second], axis = 0)).long()
        ret_dict['mask'] = torch.IntTensor(np.stack([mask_first, mask_second], axis = 0)).long()
        
        ret_dict['category_label'] = torch.IntTensor([cat_id]).long()
        ret_dict['asym_flag'] = torch.FloatTensor([asym_flag])
        ret_dict['translation_label'] = torch.FloatTensor(np.stack([translation_first, translation_second],axis = 0))
        ret_dict['rotation_label'] = torch.FloatTensor(rotation)
        ret_dict['rotation_ref'] = torch.FloatTensor(rotation_second)
        ret_dict['size_label'] = torch.FloatTensor(np.stack([size_first, size_second],axis = 0))

        ret_dict['rho_label'] = torch.IntTensor([rho_label]).long()
        ret_dict['phi_label'] = torch.IntTensor([phi_label]).long()
        ret_dict['vp_rotation_label'] = torch.FloatTensor(vp_rotation)
        ret_dict['ip_rotation_label'] = torch.FloatTensor(ip_rotation)
        return ret_dict
    
    def _load_data(self,img_path,img_type, cam_cx, cam_cy,cam_fx, cam_fy, instance_id = -1, without_noise = False):
        #import pdb;pdb.set_trace()
        if self.mode == 'SIM':
            without_noise = True
        if self.dataset == 'REAL275':
            depth = load_composed_depth(img_path)
            depth = fill_missing(depth, self.norm_scale, 1)

        else:
            depth = load_depth(img_path)

        # mask
        with open(img_path + '_label.pkl', 'rb') as f:
            gts = cPickle.load(f)
        #print("READ MASK:",img_path,time.time()-st)
        
        assert(len(gts['class_ids'])==len(gts['instance_ids']))
        mask = cv2.imread(img_path + '_mask.png')[:, :, 2] #480*640

        
        if instance_id == -1:
            num_instance = len(gts['instance_ids'])
            instance_id = np.random.randint(0, num_instance)
        cat_id = gts['class_ids'][instance_id] - 1 # convert to 0-indexed
        rmin, rmax, cmin, cmax = get_bbox(gts['bboxes'][instance_id])
        mask = np.equal(mask, gts['instance_ids'][instance_id])
        mask = np.logical_and(mask , depth > 0)
        mask = mask[rmin:rmax, cmin:cmax]
        h,w = mask.shape
        # choose
        choose = mask.flatten().nonzero()[0]
        if len(choose)<=0:
            return None
        elif len(choose) <= self.sample_num:
            choose_idx = np.random.choice(np.arange(len(choose)), self.sample_num)
        else:
            choose_idx = np.random.choice(np.arange(len(choose)), self.sample_num, replace=False)
        choose = choose[choose_idx]

        # pts
        pts2 = depth.copy()[rmin:rmax, cmin:cmax].reshape((-1)) / self.norm_scale
        pts0 = (self.xmap[rmin:rmax, cmin:cmax].reshape((-1)) - cam_cx) * pts2 / cam_fx
        pts1 = (self.ymap[rmin:rmax, cmin:cmax].reshape((-1))- cam_cy) * pts2 / cam_fy
        pts = np.transpose(np.stack([pts0, pts1, pts2]), (1,0)).astype(np.float32) # 480*640*3
        if not without_noise:
            pts = pts + np.clip(0.001*np.random.randn(pts.shape[0], 3), -0.005, 0.005)

        pts_raw = pts#.reshape(h,w,3)
        
        

        

        # rgb
        rgb = cv2.imread(img_path + '_color.png')[:, :, :3]
        rgb = rgb[:, :, ::-1] #480*640*3
        rgb_raw = rgb[rmin:rmax, cmin:cmax]
        
        
        if not without_noise:
            rgb_raw = self.colorjitter(Image.fromarray(np.uint8(rgb_raw)))
        rgb_raw = np.array(rgb_raw)
        if img_type == 'syn' and not without_noise:
            rgb_raw = rgb_add_noise(rgb_raw)
        rgb_raw = rgb_raw.astype(np.float32).reshape((-1,3))/ 255.0
        rgb = rgb_raw[choose] 
        

        # gt
        translation = gts['translations'][instance_id].astype(np.float32)
        rotation = gts['rotations'][instance_id].astype(np.float32)
        size = gts['scales'][instance_id] * gts['sizes'][instance_id].astype(np.float32)


        if hasattr(self.config, 'random_rotate') and self.config.random_rotate and not without_noise:
            pts_raw, rotation, rand_rotation = random_rotate(pts_raw, rotation, translation, self.config.angle_range, return_rand_rotation = True)
        else:
            rand_rotation = np.eye(3)
        if self.mode == 'ts':
            pts = pts_raw[choose]
            
            pts, size = random_scale(pts, size, rotation, translation)

            center = np.mean(pts, axis=0)
            pts = pts - center[np.newaxis, :]
            translation = translation - center

            noise_t = np.random.uniform(-0.02, 0.02, 3)
            pts = pts + noise_t[None, :]
            translation = translation + noise_t
            return pts, rgb, translation, rotation, size, cat_id
        elif self.mode in  ['r', 'sim']:
            
            noise_t = np.random.uniform(-0.02, 0.02, 3)
            noise_s = np.random.uniform(0.8, 1.2, 1)
            if without_noise:
                pts_raw = pts_raw - translation[None, :]
                pts_raw = pts_raw / np.linalg.norm(size)
            else:
                
                pts_raw = pts_raw - translation[None, :] - noise_t[None, :]
                pts_raw = pts_raw / np.linalg.norm(size) * noise_s

            if cat_id in self.sym_ids:
                theta_x = rotation[0, 0] + rotation[2, 2]
                theta_y = rotation[0, 2] - rotation[2, 0]
                r_norm = math.sqrt(theta_x**2 + theta_y**2)
                s_map = np.array([[theta_x/r_norm, 0.0, -theta_y/r_norm],
                                    [0.0,            1.0,  0.0           ],
                                    [theta_y/r_norm, 0.0,  theta_x/r_norm]])
                rotation = rotation @ s_map

                asym_flag = 0.0
            else:
                asym_flag = 1.0

            # transform ZXY system to XYZ system
            pts = pts_raw[choose]
            pts_raw = pts_raw.reshape(h,w,3)
            rgb_raw = rgb_raw.reshape(h,w,3)
            
            rotation = rotation[:, (2,0,1)]
            
            rgb_raw = cv2.resize(rgb_raw, dsize=(self.num_patches*14,self.num_patches*14), interpolation=cv2.INTER_NEAREST)
            pts_raw = np.where((mask == 0)[:,:,None],np.nan, pts_raw)
            pts_raw = cv2.resize(pts_raw, dsize=(self.num_patches,self.num_patches), interpolation=cv2.INTER_NEAREST)
            mask = np.logical_not(np.isnan(pts_raw)).all(axis = -1)
            # choose
            choose = mask.flatten().nonzero()[0]
            if len(choose)<=0:
                return None
            elif len(choose) <= self.sample_num:
                choose_idx = np.random.choice(np.arange(len(choose)), self.sample_num)
            else:
                choose_idx = np.random.choice(np.arange(len(choose)), self.sample_num, replace=False)
            choose = choose[choose_idx]

            
            


            return pts, rgb, translation, rotation, size, cat_id, asym_flag, \
                rmin, rmax, cmin, cmax, choose, rgb_raw.copy(), pts_raw.copy(), mask.copy() , rand_rotation
        else: 
            assert False


    def _read_instance(self, image_index):
        assert self.mode == 'ts'
        if image_index>=0:
            img_type = 'syn'
            img_path = os.path.join(self.data_dir, self.syn_img_list[image_index])
            cam_fx, cam_fy, cam_cx, cam_cy = self.syn_intrinsics
        else:
            img_type = 'real'
            image_index = -image_index-1
            img_path = os.path.join(self.data_dir, self.real_img_list[image_index])
            cam_fx, cam_fy, cam_cx, cam_cy = self.real_intrinsics
        tuple_instance = self._load_data(img_path,
                                                        img_type, 
                                                        cam_cx, cam_cy,cam_fx, cam_fy)
        if tuple_instance is None:
            return None
        pts, rgb, translation, rotation, size, cat_id = tuple_instance
        
        
        ret_dict = {}
        ret_dict['pts'] = torch.FloatTensor(pts)
        ret_dict['rgb'] = torch.FloatTensor(rgb)
        ret_dict['category_label'] = torch.IntTensor([cat_id]).long()
        ret_dict['translation_label'] = torch.FloatTensor(translation)
        ret_dict['size_label'] = torch.FloatTensor(size)
        return ret_dict
    def _read_instance_from_category_dict(self, cls, idx ):
        img_path, instance_id, img_type = self.reference_category_dict[str(cls)][idx]
        
        if img_type == 'real':
            
            
            cam_fx, cam_fy, cam_cx, cam_cy = self.real_intrinsics

        else:
            
            
            cam_fx, cam_fy, cam_cx, cam_cy = self.syn_intrinsics
        tuple_instance = self._load_data(img_path,
                                        img_type, 
                                        cam_cx, cam_cy,cam_fx, cam_fy, instance_id = instance_id,without_noise=True)
        if tuple_instance is None:
            return None

        
        pts, rgb, translation, rotation, size, cat_id, asym_flag, \
                rmin, rmax, cmin, cmax, choose, rgb_raw, pts_raw, mask, rand_rotation = tuple_instance
        
        ret_dict = {}
        ret_dict['pts'] = torch.FloatTensor(pts)
        ret_dict['rgb'] = torch.FloatTensor(rgb)
        ret_dict['rgb_raw'] = torch.FloatTensor(rgb_raw)
        ret_dict['pts_raw'] = torch.FloatTensor(pts_raw)
        ret_dict['category_label'] = torch.IntTensor([cat_id]).long()
        ret_dict['translation_label'] = torch.FloatTensor(translation)
        ret_dict['size_label'] = torch.FloatTensor(size)
        ret_dict['rotation_label'] = torch.FloatTensor(rotation)
        ret_dict['rmin'] = torch.IntTensor([rmin]).long()
        ret_dict['rmax'] = torch.IntTensor([rmax]).long()
        ret_dict['cmin'] = torch.IntTensor([cmin]).long()
        ret_dict['cmax'] = torch.IntTensor([cmax]).long()
        ret_dict['choose'] = torch.IntTensor(choose).long()
        ret_dict['mask'] = torch.IntTensor(mask).long()
        return ret_dict
    def get_ref_data(self, clss, indexes):
        data_list = []
        assert len(clss) == len(indexes)
        for cls, index in zip(clss, indexes):
            assert len(index) == 1
            
            index = int(index[0].item())
            cls = int(cls.item())
            data_list.append( self._read_instance_from_category_dict(cls, index) )
        ret_dict = {}
        for k in data_list[0].keys():
            ret_dict[k] = torch.stack([d[k] for d in data_list], dim = 0)
        return ret_dict


            

       

            

        
    


class TestDataset():
    def __init__(self, config, dataset='REAL275', resolution=64, ds_rate = 2, for_sim_feature = False):
        self.dataset = dataset
        self.resolution = resolution
        self.data_dir = config.data_dir
        self.sample_num = config.sample_num
        # self.match_sample_num = 128
        # self.raw_size = 840
        self.num_patches = 15
        

        result_pkl_list = glob.glob(os.path.join(self.data_dir, 'detection', dataset, 'results_*.pkl'))
        self.result_pkl_list = sorted(result_pkl_list)
        
        n_image = len(result_pkl_list)
        print('no. of test images: {}\n'.format(n_image))
        if dataset == 'REAL275':
            category_path = os.path.join(self.data_dir, 'detection', 'real_test_category_dict.json')
            self.category_dict = json.load(open(category_path))
            

            
        elif dataset == 'CAMERA25':
            category_path = os.path.join(self.data_dir, 'detection', 'camera_test_category_dict.json')
            self.category_dict = json.load(open(category_path))
            reference_category_path = 'real/train_category_dict.json'

            

        else:
            assert False
        self.trainDataset = TrainingDataset(
        defaultTrainconfig,
        self.dataset,
        'r',
        resolution = resolution,
        ds_rate = ds_rate,
        num_img_per_epoch=1)




        self.xmap = np.array([[i for i in range(640)] for j in range(480)])
        self.ymap = np.array([[j for i in range(640)] for j in range(480)])
        self.sym_ids = [0, 1, 3]    # 0-indexed
        self.norm_scale = 1000.0    # normalization scale
        if dataset == 'REAL275':
            self.intrinsics = [591.0125, 590.16775, 322.525, 244.11084]
        else:
            self.intrinsics = [577.5, 577.5, 319.5, 239.5]


    def __len__(self):
        return len(self.result_pkl_list)

    def __getitem__(self, index):
        #return self._get_pair_random(index)
        ret = self._get_instance(index)
        
        return ret
    
    
    def _get_instance(self,index):
        return self._get_instance_by_image_index(index)
    def _get_pair_random(self,index):
        ret_dict = self._get_instance_by_image_index(index)
        reference_rgb = []
        reference_pts = []
        
        reference_rotation = []
        reference_cat_ids = []
        for rank, cls in enumerate(ret_dict['pred_class_ids']):
            cls = str(int(cls))
            idx = np.random.randint(len(self.trainDataset.reference_category_dict[cls]))
            refer_dict = self._get_instance_from_train_set(cls, idx)
            reference_rgb.append((rank,refer_dict['rgb']))
            reference_pts.append((rank,refer_dict['pts']))
            reference_rotation.append((rank,refer_dict['rotation_label']))
            reference_cat_ids.append((rank, refer_dict['category_label']))
            # print(refer_dict['gt_size'].shape)
        
        def process_list(l):
            l.sort(key = lambda x: x[0])
            return [x[1] for x in l]
        reference_rgb = process_list(reference_rgb)
        reference_pts = process_list(reference_pts)
        reference_rotation = process_list(reference_rotation)
        reference_cat_ids = process_list(reference_cat_ids)
        
        ret_dict['reference_rgb'] = torch.stack(reference_rgb)
        ret_dict['reference_pts'] = torch.stack(reference_pts)
        
        ret_dict['reference_rotation']= torch.stack(reference_rotation)
        ret_dict['reference_cat_ids'] = torch.stack(reference_cat_ids).reshape(-1)
        assert (ret_dict['reference_cat_ids']-ret_dict['category_label']).abs().sum()==0

        return ret_dict
    # def _get_reference_instance(self, cls, idx):
    #     reference_data_dict = self._get_instance_from_train_set(cls, idx)
    #     ret_dict = {}
    #     ret_dict['rgb'] = reference_data_dict['rgb']
    #     translation = reference_data_dict['gt_RTs'][ :3, 3]
    #     ret_dict['gt_translation'] = translation
    #     scale = reference_data_dict['gt_RTs'][ :3, :3].det()**(1/3)
    #     ret_dict['pts'] = (reference_data_dict['pts']-translation)/(scale + 1e-8)
    #     ret_dict['gt_rotation'] = reference_data_dict['gt_RTs'][ :3, :3]/scale
    #     ret_dict['gt_size'] = reference_data_dict['gt_scales'] * scale
        
    #     return ret_dict

    def _get_instance_from_train_set(self, cls, idx):
        return self.trainDataset._read_instance_from_category_dict(cls,idx)


    # def _get_instance_from_cat_dict(self, cls, idx):
        
    #     img_path, id = self.category_dict[cls][idx]
        
    #     return self._get_instance_by_path(img_path, id)

    def _get_instance_by_image_index(self, index):
        path = self.result_pkl_list[index]
        # print(path)
        ret_dict = self._get_instance_by_path(path)
        if ret_dict is None:
            return None
        ret_dict['index'] = index
        return ret_dict

    def _get_instance_by_path(self, path, instance_id=-1):
        with open(path, 'rb') as f:
            pred_data = cPickle.load(f)
        
        image_path = os.path.join(self.data_dir, pred_data['image_path'][5:])
        # print(image_path)
        pred_mask = pred_data['pred_masks']
        num_instance = len(pred_data['pred_class_ids'])
        assert instance_id<= num_instance-1
        assert type(instance_id) is int

        # rgb
        rgb = cv2.imread(image_path + '_color.png')[:, :, :3]
        rgb = rgb[:, :, ::-1] #480*640*3

        # pts
        cam_fx, cam_fy, cam_cx, cam_cy = self.intrinsics
        depth = load_depth(image_path) #480*640
        if self.dataset == 'REAL275':
            depth = fill_missing(depth, self.norm_scale, 1)

        xmap = self.xmap
        ymap = self.ymap
        pts2 = depth.copy() / self.norm_scale
        pts0 = (xmap - cam_cx) * pts2 / cam_fx
        pts1 = (ymap - cam_cy) * pts2 / cam_fy
        pts = np.transpose(np.stack([pts0, pts1, pts2]), (1,2,0)).astype(np.float32) # 480*640*3

        
        if instance_id == -1:
            all_rgb = []
            all_pts = []
            all_center = []
            all_cat_ids = []
            all_rgb_raw = []
            all_pts_raw = []
            all_mask = []
            all_choose = []
            
            flag_instance = torch.zeros(num_instance) == 1
            for j in range(num_instance):
                
                inst_mask = 255 * pred_mask[:, :, j].astype('uint8')
                mask = inst_mask > 0
                mask = np.logical_and(mask, depth>0)
                if np.sum(mask) > 16:
                    rmin, rmax, cmin, cmax = get_bbox_from_mask(mask)
                    cat_id = pred_data['pred_class_ids'][j] - 1 # convert to 0-indexed

                    pts_raw = pts[rmin:rmax, cmin:cmax, :]
                    
                    rgb_raw = rgb[rmin:rmax, cmin:cmax, :]
                    mask = mask[rmin:rmax, cmin:cmax]
                    choose = mask.flatten().nonzero()[0]
                    # if path == '../../data/NOCS/detection/REAL275/results_test_scene_1_0000.pkl':
                    #     import pdb;pdb.set_trace()

                    pts_raw = np.where((mask == 0)[:,:,None],np.nan, pts_raw)
                    

                    instance_pts = pts_raw.reshape((-1, 3))[choose, :].copy()
                    
                    rgb_raw = np.array(rgb_raw.copy()).astype(np.float32) / 255.0
                    instance_rgb = rgb_raw.copy().reshape((-1, 3))[choose, :] 
                    
                    center = np.mean(instance_pts, axis=0)
                    # import pdb;pdb.set_trace()
                    instance_pts = instance_pts - center[np.newaxis, :]
                    pts_raw = pts_raw - center[np.newaxis,np.newaxis, :]
                    if instance_pts.shape[0] <= self.sample_num:
                        choose_idx = np.random.choice(np.arange(instance_pts.shape[0]), self.sample_num)
                    else:
                        choose_idx = np.random.choice(np.arange(instance_pts.shape[0]), self.sample_num, replace=False)
                    instance_pts = instance_pts[choose_idx, :]
                    instance_rgb = instance_rgb[choose_idx, :]

                    
                    rgb_raw = cv2.resize(rgb_raw, dsize=(self.num_patches*14,self.num_patches*14), interpolation=cv2.INTER_NEAREST)
                    pts_raw = cv2.resize(pts_raw, dsize=(self.num_patches,self.num_patches), interpolation=cv2.INTER_NEAREST)
                    mask = np.logical_not(np.isnan(pts_raw)).all(axis = -1)
                    choose = mask.flatten().nonzero()[0]
                    if len(choose)<=0:
                        continue
                    elif len(choose) <= self.sample_num:
                        choose_idx = np.random.choice(np.arange(len(choose)), self.sample_num)
                    else:
                        choose_idx = np.random.choice(np.arange(len(choose)), self.sample_num, replace=False)
                    choose = choose[choose_idx]

                    all_pts_raw.append(torch.FloatTensor(pts_raw))
                    all_rgb_raw.append(torch.FloatTensor(rgb_raw))
                    all_choose.append(torch.IntTensor(choose).long())
                    all_mask.append(torch.IntTensor(mask).long())
                    all_pts.append(torch.FloatTensor(instance_pts))
                    all_rgb.append(torch.FloatTensor(instance_rgb))
                    all_center.append(torch.FloatTensor(center))
                    all_cat_ids.append(torch.IntTensor([cat_id]).long())
                    flag_instance[j] = 1

            ret_dict = {}

            ret_dict['gt_class_ids'] = torch.tensor(pred_data['gt_class_ids'])
            ret_dict['gt_bboxes'] = torch.tensor(pred_data['gt_bboxes'])
            ret_dict['gt_RTs'] = torch.tensor(pred_data['gt_RTs'])
            ret_dict['gt_scales'] = torch.tensor(pred_data['gt_scales'])
            ret_dict['gt_handle_visibility'] = torch.tensor(pred_data['gt_handle_visibility'])
            #ret_dict['index'] = index

            if len(all_pts) == 0:
                ret_dict['pred_class_ids'] = torch.tensor(pred_data['pred_class_ids'])
                ret_dict['pred_bboxes'] = torch.tensor(pred_data['pred_bboxes'])
                ret_dict['pred_scores'] = torch.tensor(pred_data['pred_scores'])

            else:
                ret_dict['pts'] = torch.stack(all_pts) # N*3
                ret_dict['rand_rotation'] = torch.eye(3)[None,:,:].repeat(ret_dict['pts'].shape[0],1,1)
                ret_dict['rgb'] = torch.stack(all_rgb)
                ret_dict['pts_raw'] = torch.stack(all_pts_raw) # N*3
                ret_dict['rgb_raw'] = torch.stack(all_rgb_raw)
                ret_dict['choose'] = torch.stack(all_choose)
                ret_dict['mask'] = torch.stack(all_mask)
                ret_dict['center'] = torch.stack(all_center)
                ret_dict['category_label'] = torch.stack(all_cat_ids).squeeze(1)
                ret_dict['pred_class_ids'] = torch.tensor(pred_data['pred_class_ids'])[flag_instance==1]
                ret_dict['pred_bboxes'] = torch.tensor(pred_data['pred_bboxes'])[flag_instance==1]
                ret_dict['pred_scores'] = torch.tensor(pred_data['pred_scores'])[flag_instance==1]

            return ret_dict
        else:
            assert False
            j = instance_id
            inst_mask = 255 * pred_mask[:, :, j].astype('uint8')
            mask = inst_mask > 0
            mask = np.logical_and(mask, depth>0)
            if np.sum(mask) > 16:
                rmin, rmax, cmin, cmax = get_bbox_from_mask(mask)
                choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]
                cat_id = pred_data['pred_class_ids'][j] - 1 # convert to 0-indexed

                instance_pts = pts[rmin:rmax, cmin:cmax, :].reshape((-1, 3))[choose, :].copy()
                instance_rgb = rgb[rmin:rmax, cmin:cmax, :].copy()
                instance_rgb = np.array(instance_rgb).astype(np.float32).reshape((-1, 3))[choose, :] / 255.0

                center = np.mean(instance_pts, axis=0)
                instance_pts = instance_pts - center[np.newaxis, :]

                if instance_pts.shape[0] <= self.sample_num:
                    choose_idx = np.random.choice(np.arange(instance_pts.shape[0]), self.sample_num)
                else:
                    choose_idx = np.random.choice(np.arange(instance_pts.shape[0]), self.sample_num, replace=False)
                instance_pts = instance_pts[choose_idx, :]
                instance_rgb = instance_rgb[choose_idx, :]

                pts=torch.FloatTensor(instance_pts)
                rgb=torch.FloatTensor(instance_rgb)
                center=torch.FloatTensor(center)
                cat_id=torch.IntTensor([cat_id]).long()
                
            ret_dict = {}
            
            try:
                ret_dict['gt_class_ids'] = torch.tensor(pred_data['gt_class_ids'][j])
            except:
                print(j, "WOOO", path,num_instance, pred_data['gt_class_ids'], pred_data['pred_class_ids'], np.sum(mask) > 16)
                
            ret_dict['gt_bboxes'] = torch.tensor(pred_data['gt_bboxes'][j])
            ret_dict['gt_RTs'] = torch.tensor(pred_data['gt_RTs'][j])
            ret_dict['gt_scales'] = torch.tensor(pred_data['gt_scales'][j])
            ret_dict['gt_handle_visibility'] = torch.tensor(pred_data['gt_handle_visibility'][j])
            # ret_dict['index'] = index
            ret_dict['pts'] = pts # N*3
            ret_dict['rgb'] = rgb
            ret_dict['center'] = center
            ret_dict['category_label'] = cat_id
            ret_dict['pred_class_ids'] = torch.tensor(pred_data['pred_class_ids'])[j]
            ret_dict['pred_bboxes'] = torch.tensor(pred_data['pred_bboxes'])[j]
            ret_dict['pred_scores'] = torch.tensor(pred_data['pred_scores'])[j]
            return ret_dict


