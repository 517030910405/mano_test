import manopth.manolayer
ManoLayer = manopth.manolayer.ManoLayer
a = ManoLayer()
import torch
torch.random.manual_seed(4)
theta = torch.rand((1,3+6))
beta = torch.rand((1,10))
# glob = torch.rand((1,3))
res = a.forward(th_pose_coeffs=theta,th_betas=beta)
import numpy as np

res = np.array(res)
np.save("ori_mano",res)
print(res.shape)