import os
import numpy as np
import torch
import torch.nn as nn
from manopth import rodrigues_layer, rotproj, rot6d
from manopth.tensutils import (th_posemap_axisang, th_with_zeros, th_pack,
                               subtract_flat_id, make_list)
from mano.webuser.smpl_handpca_wrapper_HAND_only import ready_arguments


class ManoLayer(nn.Module):
    __constants__ = [
        'rot',
        'ncomps',
        'kintree_parents',
        'check',
        'hand_side',
        'root_idx',
    ]

    def __init__(
        self,
        root_idx=0,
        flat_hand_mean=True,
        ncomps=6,
        hand_side='right',
        template_root='mano/models',
        scale_milimeter=True,
    ):
        """
        Args:
            root_idx: index of root joint in our computations,
                if -1 centers on estimate of palm as middle of base
                of middle finger and wrist
            flat_hand_mean: if True, (0, 0, 0, ...) pose coefficients match
                flat hand, else match average hand pose
            template_root: path to MANO pkl files for left and right hand
            ncomps: number of PCA components form pose space (<45)
            side: 'right' or 'left'
        """
        super(ManoLayer, self).__init__()

        self.root_idx = root_idx
        self.rot = 3  # we use axisang so use 3 number to represent rotation

        self.flat_hand_mean = flat_hand_mean
        self.hand_side = hand_side  # left or right
        self.scale_milimeter = scale_milimeter

        self.ncomps = ncomps

        if hand_side == 'right':
            self.mano_path = os.path.join(template_root, 'MANO_RIGHT.pkl')
        elif hand_side == 'left':
            self.mano_path = os.path.join(template_root, 'MANO_LEFT.pkl')

        smpl_data = ready_arguments(self.mano_path)

        hands_components = smpl_data['hands_components']

        self.smpl_data = smpl_data

        self.register_buffer(
            'th_betas',
            torch.Tensor(smpl_data['betas'].r).unsqueeze(0)
        )

        self.register_buffer(
            'th_shapedirs',
            torch.Tensor(smpl_data['shapedirs'].r)
        )

        self.register_buffer(
            'th_posedirs',
            torch.Tensor(smpl_data['posedirs'].r)
        )

        self.register_buffer(
            'th_v_template',
            torch.Tensor(smpl_data['v_template'].r).unsqueeze(0)
        )

        self.register_buffer(
            'th_J_regressor',
            torch.Tensor(np.array(smpl_data['J_regressor'].toarray()))
        )

        self.register_buffer(
            'th_weights',
            torch.Tensor(smpl_data['weights'].r)
        )

        self.register_buffer(
            'th_faces',
            torch.Tensor(smpl_data['f'].astype(np.int32)).long()
        )

        # Get hand mean
        hands_mean = np.zeros(
            hands_components.shape[1]
        ) if flat_hand_mean else smpl_data['hands_mean']

        hands_mean = hands_mean.copy()
        th_hands_mean = torch.Tensor(hands_mean).unsqueeze(0)

        # Save as axis-angle
        self.register_buffer('th_hands_mean', th_hands_mean)
        selected_components = hands_components[:ncomps]
        self.register_buffer(
            'th_selected_comps',
            torch.Tensor(selected_components)
        )

        # Kinematic chain params
        self.kintree_table = smpl_data['kintree_table']
        parents = list(self.kintree_table[0].tolist())
        self.kintree_parents = parents

    def forward(self, th_pose_coeffs, th_betas, poses_root=None):
        """
        Args:
        th_trans (Tensor (batch_size x ncomps)): if provided, applies trans to joints and vertices
        th_betas (Tensor (batch_size x 10)): if provided, uses given shape parameters for hand shape
        else centers on root joint (9th joint)
        """
        batch_size = th_pose_coeffs.shape[0]
        # Get axis angle from PCA components and coefficients

        # Remove global rot coeffs
        # th_pose_coeffs: (B, ( 3 + ncomps )) the initial 3 is the axis-angle
        th_hand_pose_coeffs = th_pose_coeffs[:, self.rot:self.rot + self.ncomps]  #(B, 6)

        th_full_hand_pose = th_hand_pose_coeffs.mm(self.th_selected_comps) # (B, 6) x (6, 45) -> (B, 45)

        # Concatenate back global rot
        th_full_pose = torch.cat(
            [
                th_pose_coeffs[:, :self.rot],
                (self.th_hands_mean + th_full_hand_pose)
            ], dim=1
        ) # (B, 48) 3+45

        # compute rotation matrixes from axis-angle while skipping global rotation
        th_pose_map, th_rot_map = th_posemap_axisang(th_full_pose)   #(B, 144), (B, 144)
        root_rot = th_rot_map[:, :9].view(batch_size, 3, 3)  #(B, 9) -> (B, 3, 3)
        th_rot_map = th_rot_map[:, 9:]  #(B, 135) 15*9
        th_pose_map = th_pose_map[:, 9:]  #(B, 135) 15*9

        # Full axis angle representation with root joint

        th_v_shaped = torch.matmul(
            self.th_shapedirs,  #(778, 3, 10)
            th_betas.transpose(1, 0)  #(B, 10) -> (10, B)
        ).permute(2, 0, 1) + self.th_v_template #(B, 778, 3)

        th_j = torch.matmul(self.th_J_regressor, th_v_shaped) #(16, 778) * (B, 778, 3) -> (B, 16, 3)

        # th_pose_map should have shape 20x135

        th_v_posed = th_v_shaped + \
            torch.matmul(
                self.th_posedirs, #(778, 3, 135)
                th_pose_map.transpose(0, 1) #(135, B)
            ).permute(2, 0, 1) #(B, 778, 3)
        ##### Final T pose with transformation done ! #####

        # Global rigid transformation

        root_j = th_j[:, 0, :].contiguous().view(batch_size, 3, 1)
        root_trans = th_with_zeros(torch.cat([root_rot, root_j], 2))

        all_rots = th_rot_map.view(th_rot_map.shape[0], 15, 3, 3)
        lev1_idxs = [1, 4, 7, 10, 13]
        lev2_idxs = [2, 5, 8, 11, 14]
        lev3_idxs = [3, 6, 9, 12, 15]
        lev1_rots = all_rots[:, [idx - 1 for idx in lev1_idxs]]
        lev2_rots = all_rots[:, [idx - 1 for idx in lev2_idxs]]
        lev3_rots = all_rots[:, [idx - 1 for idx in lev3_idxs]]
        lev1_j = th_j[:, lev1_idxs]
        lev2_j = th_j[:, lev2_idxs]
        lev3_j = th_j[:, lev3_idxs]

        # From base to tips
        # Get lev1 results
        all_transforms = [root_trans.unsqueeze(1)]
        lev1_j_rel = lev1_j - root_j.transpose(1, 2)
        lev1_rel_transform_flt = th_with_zeros(torch.cat([lev1_rots, lev1_j_rel.unsqueeze(3)], 3).view(-1, 3, 4))
        root_trans_flt = root_trans.unsqueeze(1).repeat(1, 5, 1, 1).view(root_trans.shape[0] * 5, 4, 4)
        lev1_flt = torch.matmul(root_trans_flt, lev1_rel_transform_flt)
        all_transforms.append(lev1_flt.view(all_rots.shape[0], 5, 4, 4))

        # Get lev2 results
        lev2_j_rel = lev2_j - lev1_j
        lev2_rel_transform_flt = th_with_zeros(torch.cat([lev2_rots, lev2_j_rel.unsqueeze(3)], 3).view(-1, 3, 4))
        lev2_flt = torch.matmul(lev1_flt, lev2_rel_transform_flt)
        all_transforms.append(lev2_flt.view(all_rots.shape[0], 5, 4, 4))

        # Get lev3 results
        lev3_j_rel = lev3_j - lev2_j
        lev3_rel_transform_flt = th_with_zeros(torch.cat([lev3_rots, lev3_j_rel.unsqueeze(3)], 3).view(-1, 3, 4))
        lev3_flt = torch.matmul(lev2_flt, lev3_rel_transform_flt)
        all_transforms.append(lev3_flt.view(all_rots.shape[0], 5, 4, 4))

        reorder_idxs = [0, 1, 6, 11, 2, 7, 12, 3, 8, 13, 4, 9, 14, 5, 10, 15]
        th_results = torch.cat(all_transforms, 1)[:, reorder_idxs]
        th_results_global = th_results

        joint_js = torch.cat([th_j, th_j.new_zeros(th_j.shape[0], 16, 1)], 2)
        tmp2 = torch.matmul(th_results, joint_js.unsqueeze(3))
        th_results2 = (th_results - torch.cat([tmp2.new_zeros(*tmp2.shape[:2], 4, 3), tmp2], 3)).permute(0, 2, 3, 1)

        th_T = torch.matmul(th_results2, self.th_weights.transpose(0, 1))

        th_rest_shape_h = torch.cat(
            [
                th_v_posed.transpose(2, 1),
                torch.ones(
                    (batch_size, 1, th_v_posed.shape[1]),
                    dtype=th_T.dtype,
                    device=th_T.device
                ),
            ], dim=1
        )

        th_verts = (th_T * th_rest_shape_h.unsqueeze(1)).sum(2).transpose(2, 1)
        th_verts = th_verts[:, :, :3] #(B, 778, 3)
        th_jtr = th_results_global[:, :, :3, 3]
        # In addition to MANO reference joints we sample vertices on each finger
        # to serve as finger tips
        if self.hand_side == 'right':
            tips = th_verts[:, [745, 317, 444, 556, 673]]
        else:
            tips = th_verts[:, [745, 317, 445, 556, 673]]

        th_jtr = torch.cat([th_jtr, tips], 1)

        # Reorder joints to match visualization utilities
        th_jtr = th_jtr[:, [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]] #(B, 21, 3)

        # root_relative:
        root_joint = th_jtr[:, self.root_idx].unsqueeze(1)  #(B, 1, 3)
        th_jtr = th_jtr - root_joint
        th_verts = th_verts - root_joint

        # add the global root position to all verts and joints
        if poses_root is not None:
            poses_root = poses_root.unsqueeze(1) #(B, 1, 3)
            th_jtr = th_jtr + poses_root
            th_verts = th_verts + poses_root

        # Scale to milimeters
        if self.scale_milimeter:
            th_verts = th_verts * 1000
            th_jtr = th_jtr * 1000

        return th_verts, th_jtr  # (B, 778, 3), (B, 21, 3)
