import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Module
import neural_renderer as nr

from pose.manopth.manopth.manolayer import ManoLayer
from .theta_regressor import ThetaRegressor

class EncEncoder(nn.Module):
    def __init__(self, inp_ch, out_ch, name='enc'):
        super(EncEncoder, self).__init__()
        self.name=name
        self.inp_ch = inp_ch
        self.out_ch = out_ch
        self.encenc = nn.Sequential()

        #  (64x64 -> 1x1)
        ds_reslu = [
            32,
            16,
            8,
            4,
            2,
            1,
        ]

        for reslu in ds_reslu:
            if reslu == 4 or reslu == 2:
                mid_ch = inp_ch * 2
            elif reslu == 1:
                mid_ch = self.out_ch
            else:
                mid_ch = inp_ch

            kernel_size = 3
            self.encenc.add_module(
                name = self.name + '_conv_{}'.format(reslu),
                module = nn.Conv2d(
                    inp_ch,
                    mid_ch,
                    kernel_size=kernel_size,
                    stride=2,
                    padding=(kernel_size-1)//2,
                    bias=True
                )
            )

            if reslu != 1:
                self.encenc.add_module(
                    name = self.name + '_bn_{}'.format(reslu),
                    module = nn.BatchNorm2d(mid_ch)
                )

                self.encenc.add_module(
                    name = self.name + '_relu_{}'.format(reslu),
                    module = nn.LeakyReLU(inplace=True)
                )
            inp_ch = mid_ch

    def forward(self, x):
        batch_size = x.shape[0]
        return self.encenc(x).reshape(batch_size, -1)  #(B, 2048)


class ManoRender(nn.Module):
    def __init__(
        self,
        fill_back=True,
    ):
        super(ManoRender, self).__init__()
        self.fill_back=fill_back

    ''' Render Depth '''
    def forward(
        self,
        vertices,
        faces,
        Ks=None,
        Rs=None,
        ts=None,
        dist_coeffs=None,
        bbxs=None,
        image_size=64,
        orig_size=64,
        anti_aliasing=False,
        far = 100.0,
    ):
        # batch_size = vertices.shape(0)
        if self.fill_back:
            faces = torch.cat(
                (
                    faces,
                    faces[:, :, list(reversed(range(faces.shape[-1])))]
                ),
                dim=1,
            ).to(vertices.device).detach()

        if Ks is None:
            print("K must not None if render depthmap")
            raise Exception()

        if Rs is None:
            Rs = torch.Tensor(
                [
                    [1,0,0],
                    [0,1,0],
                    [0,0,1],
                ]
            ).view((1,3,3)).to(vertices.device)

        if ts is None:
            ts = torch.Tensor([0,0,0]).view((1,3)).to(vertices.device)

        if dist_coeffs is None:
            dist_coeffs = torch.Tensor([[0., 0., 0., 0., 0.]]).to(vertices.device)

        ''' xyz -> uvd '''
        vertices = self.projection(
            vertices, Ks, Rs, ts, dist_coeffs, orig_size, bbxs=bbxs
        )

        faces = nr.vertices_to_faces(vertices, faces)
        # rasteriation
        rast = nr.rasterize_depth(faces, image_size, anti_aliasing, far=far)
        # normalize to 0~1
        rend = self.normalize_depth(rast, far=far)
        return rend

    def normalize_depth(self, img, far):
        img_inf = torch.eq(img, far * torch.ones_like(img)).type(torch.float32) #Bool Tensor
        img_ok = 1-img_inf #Bool Tensor
        img_no_back = img_ok * img #Float Tensor
        img_max = img_no_back.max(dim=1,keepdim=True)[0] #batch of max value
        img_max = img_max.max(dim=2,keepdim=True)[0]

        img_min = img.min(dim=1,keepdim = True)[0] #batch of min values
        img_min = img_min.min(dim=2,keepdim = True)[0]

        new_depth = (img_max - img)/(img_max - img_min)
        new_depth = torch.max(new_depth, torch.zeros_like(new_depth))
        new_depth = torch.min(new_depth, torch.ones_like(new_depth))
        return new_depth

    def projection(
        self,
        vertices,
        Ks,
        Rs,
        ts,
        dist_coeffs,
        orig_size,
        bbxs=None,
        eps=1e-9,
    ):
        '''
        Calculate projective transformation of vertices given a projection matrix
        Input parameters:
        K: batch_size * 3 * 3 intrinsic camera matrix
        R, t: batch_size * 3 * 3, batch_size * 1 * 3 extrinsic calibration parameters
        dist_coeffs: vector of distortion coefficients
        orig_size: original size of image captured by the camera
        Returns: For each point [X,Y,Z] in world coordinates [u,v,z]
        where u,v are the coordinates of the projection in
        pixels and z is the depth
        Modified by Li Jiasen: add bbx
        '''

        # instead of P*x we compute x'*P'
        vertices = torch.matmul(vertices, Rs.transpose(2,1)) + ts
        x, y, z = vertices[:, :, 0], vertices[:, :, 1], vertices[:, :, 2]
        x_ = x / (z + eps)
        y_ = y / (z + eps)

        # Get distortion coefficients from vector
        k1 = dist_coeffs[:, None, 0]
        k2 = dist_coeffs[:, None, 1]
        p1 = dist_coeffs[:, None, 2]
        p2 = dist_coeffs[:, None, 3]
        k3 = dist_coeffs[:, None, 4]

        # we use x_ for x' and x__ for x'' etc.
        r = torch.sqrt(x_ ** 2 + y_ ** 2)
        x__ = x_*(1 + k1*(r**2) + k2*(r**4) + k3*(r**6)) + 2*p1*x_*y_ + p2*(r**2 + 2*x_**2)
        y__ = y_*(1 + k1*(r**2) + k2*(r**4) + k3 *(r**6)) + p1*(r**2 + 2*y_**2) + 2*p2*x_*y_
        vertices = torch.stack([x__, y__, torch.ones_like(z)], dim=-1)
        vertices = torch.matmul(vertices, Ks.transpose(1,2))
        u, v = vertices[:, :, 0], vertices[:, :, 1]
        if bbxs is not None:
            u = (u - bbxs[:,0:1])/bbxs[:,2:3] * orig_size
            v = (v - bbxs[:,1:2])/bbxs[:,3:4] * orig_size

        # map u,v from [0, img_size] to [-1, 1] to use by the renderer
        v = orig_size - v
        u = 2 * (u - orig_size / 2.) / orig_size
        v = 2 * (v - orig_size / 2.) / orig_size
        vertices = torch.stack([u, v, z], dim=-1)
        return vertices


class NetMano(nn.Module):
    def __init__(
        self,
        mano_ncomps=6,
        mano_root_idx=0,
        mano_flat_hand_mean=True,
        mano_hand_side='right',
        mano_template_root='mano/models',
        mano_scale_milimeter=False,
        reg_inp_encs=['hm','dep'],
        reg_inp_enc_res=64,
        reg_niter=10,
        reg_nfeats=2048,
        njoints=21,
    ):
        super(NetMano, self).__init__()
        self.reg_inp_encs = reg_inp_encs
        self.ncomps = mano_ncomps
        self.render_res = reg_inp_enc_res

        self.reg_ntheta = 3 + mano_ncomps + 10 # 3 rots, 6 ncomps, 10 betas = 19
        enc_ch = len(reg_inp_encs) * 256

        self.enc_conv = nn.Conv2d(enc_ch, 256, kernel_size=1, stride=1, bias=True)
        self.enc_encoder = EncEncoder(inp_ch=256, out_ch=reg_nfeats, name='encenc')


        self.pred_ch = njoints + 1  # njoints heatmap + 1 depth
        self.pred_conv = nn.Conv2d(self.pred_ch, 256, kernel_size=1, stride=1, bias=True)
        self.pred_encoder = EncEncoder(inp_ch=256, out_ch=reg_nfeats, name='predenc')

        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.bn = nn.BatchNorm2d(256)

        fc_layers = [
            reg_nfeats + self.reg_ntheta, # 2048 + 19
            1024,
            1024,
            self.reg_ntheta  # 19
        ]

        use_dropout = [True, True, False]
        drop_prob = [0.3, 0.3, 0.3]

        self.regressor = ThetaRegressor(
            fc_layers=fc_layers,
            use_dropout=use_dropout,
            drop_prob=drop_prob,
            ncomps=mano_ncomps,
            iterations=reg_niter,
        )

        self.manolayer = ManoLayer(
            root_idx=mano_root_idx,
            flat_hand_mean=mano_flat_hand_mean,
            ncomps=mano_ncomps,
            hand_side=mano_hand_side,
            template_root=mano_template_root,
            scale_milimeter=mano_scale_milimeter,
        )

        template_pth = os.path.join(mano_template_root, 'HAND_TEMPLATE_RIGHT.obj')
        self.template = nr.Mesh.fromobj(template_pth)

        self.manorender = ManoRender()


    def forward(self, encodings, hms, deps, poses_root, Ks, bbxs):

        ### prepare encodings
        batch_size = hms.shape[0]
        enc_list = []
        for key in self.reg_inp_encs:
            enc_list.append(encodings[key])

        enc = torch.cat(enc_list, dim=1) #(B, 256x2, 64, 64)
        enc = self.enc_conv(enc)  #(B, 256, 64, 64)
        enc = self.bn(enc)
        enc = self.leaky_relu(enc)
        enc = self.enc_encoder(enc) #(B, 2048)

        x = torch.cat((hms, deps), dim=1) #(B, 22, 64, 64)
        x = self.pred_conv(x)  #(B, 256, 64, 64)
        x = self.bn(x)
        x = self.leaky_relu(x)
        x = self.pred_encoder(x) #(B, 2048)

        x = x + enc

        thetas = self.regressor(x)
        theta = thetas[-1]
        th_pose_coeffs = theta[:, :(3+self.ncomps)] #(B, 0:9)
        th_betas = theta[:, (3+self.ncomps):] #(B, 9:19)

        verts, joints = self.manolayer(th_pose_coeffs, th_betas, poses_root)
        faces = self.template.faces.unsqueeze(0).repeat((batch_size, 1, 1))

        rendered = self.manorender(
            vertices=verts,
            faces=faces,
            Ks=Ks,
            bbxs=bbxs,
            far=100.0,
            image_size=self.render_res,
            orig_size=self.render_res,
        )

        return verts, joints, rendered








