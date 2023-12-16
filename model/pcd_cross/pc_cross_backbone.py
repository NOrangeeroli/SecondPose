import torch.nn as nn
import torch.nn.functional as F
from .pointnet2_utils import PointNetSetAbstractionMsg, PointNetFeaturePropagation
from .pcd_cross import GeometricTransformer


class get_model(nn.Module):
    def __init__(self, output_dim):
        super(get_model, self).__init__()

        self.sa1 = PointNetSetAbstractionMsg(1024, [0.05, 0.1], [16, 32], 6, [[16, 16, 32], [32, 32, 64]])
        self.sa2 = PointNetSetAbstractionMsg(256, [0.1, 0.2], [16, 32], 32 + 64, [[64, 64, 128], [64, 96, 128]])
        self.sa3 = PointNetSetAbstractionMsg(64, [0.2, 0.4], [16, 32], 128 + 128, [[128, 196, 256], [128, 196, 256]])
        self.sa4 = PointNetSetAbstractionMsg(16, [0.4, 0.8], [16, 32], 256 + 256, [[256, 256, 512], [256, 384, 512]])
        # encoder

        # decoder
        self.fp4 = PointNetFeaturePropagation(512 + 512 + 256 + 256, [256, 256])
        self.fp3 = PointNetFeaturePropagation(128 + 128 + 256, [256, 256])
        self.fp2 = PointNetFeaturePropagation(32 + 64 + 256, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, output_dim, 1)

        # transformer for point cloud cross attention
        self.cross_attn_model = GeometricTransformer(input_dim=1024,
                                                     output_dim=1024,
                                                     hidden_dim=1024,
                                                     num_heads=2,
                                                     blocks=["self", "cross", "self", "cross", "self", "cross"],
                                                     sigma_d=0.2,
                                                     sigma_a=15,
                                                     angle_k=3, )

    def forward(self, ref_p, src_p):
        # share weights encoding
        l0_points = src_p
        l0_xyz = src_p[:, :3, :]
        # import pdb;pdb.set_trace()
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        l0_points_r = ref_p
        l0_xyz_r = ref_p[:, :3, :]

        l1_xyz_r, l1_points_r = self.sa1(l0_xyz_r, l0_points_r)
        l2_xyz_r, l2_points_r = self.sa2(l1_xyz_r, l1_points_r)
        l3_xyz_r, l3_points_r = self.sa3(l2_xyz_r, l2_points_r)
        l4_xyz_r, l4_points_r = self.sa4(l3_xyz_r, l3_points_r)

        _, l4_points = self.cross_attn_model(ref_points=l4_xyz_r.permute(0, 2, 1),
                                             src_points=l4_xyz.permute(0, 2, 1),
                                             ref_feats=l4_points_r.permute(0, 2, 1),
                                             src_feats=l4_points.permute(0, 2, 1),)
        l4_points = l4_points.permute(0, 2, 1)

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)

        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        x = self.conv2(x)
        x = x.permute(0, 2, 1)
        return x


if __name__ == '__main__':
    import torch

    model = get_model(256).cuda()
    ref = torch.rand(64, 9, 2048).cuda()  # bs x c x nofpoints
    src = torch.rand(64, 9, 2048).cuda() # bs x c x nofpoints
    # the first 3 dimensions are xyz coordinates, the last 6 dimensions are features, including color 3 and normal 3
    res_x = model(ref, src)
    import pdb;pdb.set_trace()
    print(res_x.shape)
