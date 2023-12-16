import torch
import torch.nn as nn
import torch.nn.functional as F

from module import SphericalFPN, V_Branch, I_Branch
from loss import SigmoidFocalLoss
from smap_utils import Feat2Smap


class Net(nn.Module):
    def __init__(self, resolution=64, ds_rate=2):
        super(Net, self).__init__()
        self.res = resolution
        self.ds_rate = ds_rate
        

        # data processing
        self.feat2smap = Feat2Smap(self.res)

        self.spherical_fpn = SphericalFPN(ds_rate=self.ds_rate, dim_in1=1, dim_in2=3)
        self.cos  = nn.CosineSimilarity(dim=1, eps=1e-6)

        # self.mlp = nn.Sequential(
        #     nn.Linear(256*2, 512, bias=False),
        #     nn.BatchNorm1d(512),
        #     nn.ReLU(),
        #     nn.Linear(512, 256, bias=False),
        #     nn.BatchNorm1d(256),
        #     nn.ReLU(),
        #     nn.Linear(256, 1),
        # )
    def extractor(self,inputs):
        dis_map, rgb_map = self.feat2smap(inputs['pts'], inputs['rgb'])
        x = self.spherical_fpn(dis_map, rgb_map)
        x = x.reshape((x.shape[0], x.shape[1], -1))

        x = torch.mean(x, dim = -1)
        return x

    def forward(self, inputs):
        #import pdb;pdb.set_trace()
        rgb1, rgb2 = inputs['rgb'][:,0,:,:], inputs['rgb'][:,1,:,:]
        pts1, pts2 = inputs['pts'][:,0,:,:], inputs['pts'][:,1,:,:]
        dis_map1, rgb_map1 = self.feat2smap(pts1, rgb1)
        dis_map2, rgb_map2 = self.feat2smap(pts2, rgb2)
        
        
        # backbone
        x1 = self.spherical_fpn(dis_map1, rgb_map1)
        x2 = self.spherical_fpn(dis_map2, rgb_map2)
        # import pdb;pdb.set_trace()

        x1 = x1.reshape((x1.shape[0], x1.shape[1], -1))
        x2 = x2.reshape((x2.shape[0], x2.shape[1], -1))

        # x1 = torch.max(x1, dim = -1)[0]
        # x2 = torch.max(x2, dim = -1)[0]
        x2 = torch.mean(x2, dim = -1)
        x1 = torch.mean(x1, dim = -1)


        # x1 = x1/torch.norm(x1, p=2, dim=-1, keepdim=True)
        # x2 = x2/torch.norm(x2, p=2, dim=-1, keepdim=True)
        # cos = torch.einsum('bi,bi->b', x1, x2)
        cos = self.cos(x1, x2)
        




        return cos

# class Net(nn.Module):
#     def __init__(self, resolution=64, ds_rate=2):
#         super(Net, self).__init__()
#         self.res = resolution
#         self.ds_rate = ds_rate
#         self.backbone = _Net(resolution, ds_rate)
        

#         self.offset = nn.Parameter(torch.tensor(-0.2, requires_grad=True))
        

        # self.mlp = nn.Sequential(
        #     nn.Linear(256*2, 512, bias=False),
        #     nn.BatchNorm1d(512),
        #     nn.ReLU(),
        #     nn.Linear(512, 256, bias=False),
        #     nn.BatchNorm1d(256),
        #     nn.ReLU(),
        #     nn.Linear(256, 1),
        # )

    # def forward(self, inputs):
        


    #     cos = self.backbone(inputs)
    #     cos = cos + self.offset




    #     return cos, self.offset



class Loss(nn.Module):
    def __init__(self, cfg):
        super(Loss, self).__init__()
        self.cfg = cfg
        self.l2loss = nn.MSELoss()
        self.l1loss = nn.L1Loss()
        

    def forward(self, pred, gt):
        # pred, offset = pred
        # pred = pred.reshape(-1)
        gt['cos'] = gt['cos'].reshape(-1)
        assert gt['cos'].shape == pred.shape
        angle_loss = self.l1loss(pred, gt['cos'])
        
        corr = torch.corrcoef(torch.stack((pred,gt['cos'])))[0][1]
        

        return {'loss':angle_loss, 'corr': corr, 'pred_cos_mean': pred.mean(), 'gt_cos_mean': gt['cos'].mean()}