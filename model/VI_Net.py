import torch
import torch.nn as nn
import torch.nn.functional as F

from module import SphericalFPN, V_Branch, I_Branch
from loss import SigmoidFocalLoss
from smap_utils import Feat2Smap
from rotation_utils import angle_of_rotation


class Net(nn.Module):
    def __init__(self, resolution=64, ds_rate=2):
        super(Net, self).__init__()
        self.res = resolution
        self.ds_rate = ds_rate
        self.ds_res = resolution//ds_rate

        # data processing
        self.feat2smap = Feat2Smap(self.res)

        self.spherical_fpn = SphericalFPN(ds_rate=self.ds_rate)
        self.v_branch = V_Branch(resolution=self.ds_res)
        self.i_branch = I_Branch(resolution=self.ds_res)

    def forward(self, inputs):
        rgb = inputs['rgb']
        pts = inputs['pts']
        dis_map, rgb_map = self.feat2smap(pts, rgb)
        
        # backbone
        x = self.spherical_fpn(dis_map, rgb_map)

        # viewpoint rotation
        vp_rot, rho_prob, phi_prob = self.v_branch(x, inputs)
        pred_vp_rot = self.v_branch._get_vp_rotation(rho_prob, phi_prob,{})

        # in-plane rotation
        ip_rot = self.i_branch(x, vp_rot)
        
        outputs = {
            'pred_rotation': vp_rot @ ip_rot,
            'pred_vp_rotation': pred_vp_rot,
            'pred_ip_rotation': ip_rot,
            'rho_prob': rho_prob,
            'phi_prob': phi_prob
        }
        return outputs


class Loss(nn.Module):
    def __init__(self, cfg):
        super(Loss, self).__init__()
        self.cfg = cfg
        self.l1loss = nn.L1Loss()
        self.sfloss = SigmoidFocalLoss()

    def forward(self, pred, gt):
        rho_prob = pred['rho_prob']
        rho_label = F.one_hot(gt['rho_label'].squeeze(1), num_classes=rho_prob.size(1)).float()
        rho_loss = self.sfloss(rho_prob, rho_label).mean()
        pred_rho = torch.max(torch.sigmoid(rho_prob),1)[1]
        rho_acc = (pred_rho.long() == gt['rho_label'].squeeze(1).long()).float().mean() * 100.0

        phi_prob =  pred['phi_prob']
        phi_label = F.one_hot(gt['phi_label'].squeeze(1), num_classes=phi_prob.size(1)).float()
        phi_loss = self.sfloss(phi_prob, phi_label).mean()
        pred_phi = torch.max(torch.sigmoid(phi_prob),1)[1]
        phi_acc = (pred_phi.long() == gt['phi_label'].squeeze(1).long()).float().mean() * 100.0

        vp_loss = rho_loss + phi_loss
        ip_loss = self.l1loss(pred['pred_rotation'], gt['rotation_label'])
        loss = self.cfg.vp_weight * vp_loss + ip_loss


        residual_angle = angle_of_rotation(pred['pred_rotation'].transpose(1,2) @ gt['rotation_label']).mean()

        vp_residual_angle = angle_of_rotation(pred['pred_vp_rotation'].transpose(1,2) @ gt['vp_rotation_label']).mean()
        ip_residual_angle = angle_of_rotation(pred['pred_ip_rotation'].transpose(1,2) @ gt['ip_rotation_label']).mean()
        
        loss = loss #+ vp_residual_angle + ip_residual_angle

        return {
            'loss': loss,
            'vp_loss': vp_loss,
            'ip_loss': ip_loss,
            'rho_acc': rho_acc,
            'phi_acc': phi_acc,
            'residual_angle':residual_angle,
            'vp_residual_angle':vp_residual_angle,
            'ip_residual_angle': ip_residual_angle
        }
