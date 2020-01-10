import torch
from manopth.manolayer import ManoLayer
from manopth.manolayer2 import ManoLayer as ManoLayer2

from manopth import demo
torch.random.manual_seed(5)
batch_size = 10
# Select number of principal components for pose space
ncomps = 6

# Initialize MANO layer
mano_layer = ManoLayer(
    mano_root='mano/models', use_pca=True, ncomps=ncomps, flat_hand_mean=True, center_idx=0)
mano_layer2 = ManoLayer2(
    template_root='mano/models', ncomps=ncomps)

# Generate random shape parameters
random_shape = torch.rand(batch_size, 10)
# Generate random pose parameters, including 3 values for global axis-angle rotation
random_pose = torch.rand(batch_size, ncomps + 3)

# Forward pass through MANO layer
hand_verts, hand_joints = mano_layer(random_pose, random_shape)
hand_verts2, hand_joints2 = mano_layer2(random_pose, random_shape)
print(hand_verts.shape)
print(hand_joints.shape)
print(torch.equal(hand_verts,hand_verts2))
print(torch.equal(hand_joints,hand_joints2))
demo.display_hand({
    'verts': hand_verts,
    'joints': hand_joints
}, mano_faces=mano_layer.th_faces, savename="1")
demo.display_hand({
    'verts': hand_verts2,
    'joints': hand_joints2
}, mano_faces=mano_layer.th_faces, savename="2")
print(hand_joints[0,0])
print(hand_joints2[0,0])