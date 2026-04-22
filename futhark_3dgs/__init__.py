# This file is a modified version of the inria group's 3DGS differentiable rasterizer module 
# https://github.com/graphdeco-inria/diff-gaussian-rasterization/blob/main/diff_gaussian_rasterization/__init__.py
#
# I have adapted it to use my futhark rasterizer instead of their cuda rasterizer.


from typing import NamedTuple
import torch.nn as nn
import numpy as np
import torch
from futhark_server import Server
import os


def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)


def to_numpy(v):
    if isinstance(v, torch.Tensor):
        return v.detach().cpu().numpy().astype(np.float32)
    return np.array(v).astype(np.float32)

def to_torch(v):
    return torch.from_numpy(np.copy(v)).cuda()


class Futhark_Rasterization_Server(Server):
    def __init__(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        rasterizer_path = os.path.join(base_dir, '..', 'futhark_rasterizer', 'rasterizer')
        rasterizer_path = os.path.abspath(rasterizer_path)
        super().__init__(rasterizer_path)

def rasterize_gaussians(
    means3D,
    means2D,
    sh,
    colors_precomp,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    raster_settings,
):
    return _RasterizeGaussians.apply(
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    )

class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings
    ):
        # pass our params to the futhark server through stdin. It's a major
        # bummer that we have to detach our tensors from the gpu to feed them to 
        # the server via stdin where they just get written to the gpu again. We basically
        # go gpu -> cpu -> gpu. Kinda sus..
        server = raster_settings.futhark_server
        inputs = {
            'bg':           to_numpy(raster_settings.bg),
            'means3D':      to_numpy(means3D),
            'colors':       to_numpy(colors_precomp),
            'opacities':    to_numpy(opacities),
            'scales':       to_numpy(scales),
            'rotations':    to_numpy(rotations),
            'viewmatrix':   to_numpy(raster_settings.viewmatrix),
            'projmatrix':   to_numpy(raster_settings.projmatrix),
            'tanfovx':      np.float32(raster_settings.tanfovx),
            'tanfovy':      np.float32(raster_settings.tanfovy),
            'image_height': np.int64(raster_settings.image_height),
            'image_width':  np.int64(raster_settings.image_width),
            'ssim_kernel_size': np.int32(11),
            'ssim_kernel_sigma': np.float32(1.5),
            'gt_image':     to_numpy(raster_settings.gt_image).transpose(1,2,0),
            'lambda' :      np.float32(0.2)
        }
        # provide all inputs to the server
        for name, value in inputs.items():
            server.put_value(name, value)

        # call our all-inclusive grad function
        output_vars = ['dmeans3d', 
                     #  'dmeans2d', 
                       'dcolors', 'dopacities', 'dscales', 'drotations', 'color', 'radii', 'loss']
        server.cmd_call("grad2", *output_vars, *inputs.keys())

        # collect outputs
        outputs = {}
        for var in output_vars:
            outputs[var] = server.get_value(var)
        
        # free variables from the server (we will replace them with new values next time we call the rasterizer)
        server.cmd_free(*output_vars, *inputs.keys())
        
        # they store the color in a weird way. we mimic
        outputs['color'] = np.transpose(outputs['color'], (2, 0, 1))

        # they also have this mysterious third column in their 2d mean derivatives. we mimic:
        #outputs['dmeans2d'] = np.hstack([outputs['dmeans2d'], np.zeros((outputs['dmeans2d'].shape[0], 1))])
        
        invdepths = np.array([])

        # save the derivatives for the "backward pass" (we really just did both passes)
        ctx.save_for_backward(
            to_torch(outputs['dmeans3d']), 
            #to_torch(outputs['dmeans2d']), 
            to_torch(outputs['dmeans3d']), 
            to_torch(outputs['dcolors']), 
            to_torch(outputs['dopacities']), 
            to_torch(outputs['dscales']), 
            to_torch(outputs['drotations']))
        
        
        # print('#################################################### FUTHARK #######################################################')
        # grads = {
        #     'dmeans3d': to_torch(outputs['dmeans3d']), 
        #     'dmeans2d': to_torch(outputs['dmeans2d']), 
        #     'dcolors' :to_torch(outputs['dcolors']), 
        #     'dopacities':to_torch(outputs['dopacities']), 
        #     'dscales':to_torch(outputs['dscales']), 
        #     'drotations':to_torch(outputs['drotations'])
        # }
        # for name, grad in grads.items():
        #     try:
        #         print(name, grad[:5])
        #     except:
        #        pass

        return (torch.tensor(outputs['color'], device='cuda', dtype=torch.float32), 
                torch.tensor(outputs['radii'], device='cuda', dtype=torch.int32), 
                invdepths,
                torch.tensor(outputs['loss'], device='cuda', dtype=torch.float32))

    # this method is a bit weird. Because our futhark AD is fused, we've done both
    # passes in "forward", we are simply returning the results of that automatic backward
    # pass here
    @staticmethod
    
    def backward(ctx, _, __, ___, ____):

        # Restore grads from context
        grad_means3D, grad_means2D, grad_colors_precomp, grad_opacities,grad_scales,grad_rotations = ctx.saved_tensors


        grads = (
            grad_means3D,
            grad_means2D,
            torch.Tensor([]), # sh gradients
            grad_colors_precomp,
            grad_opacities,
            grad_scales,
            grad_rotations,
            torch.Tensor([]),
            None,
        )

        return grads

class GaussianRasterizationSettingsFuthark(NamedTuple):
    image_height: int
    image_width: int 
    tanfovx : float
    tanfovy : float
    bg : torch.Tensor
    scale_modifier : float
    viewmatrix : torch.Tensor
    projmatrix : torch.Tensor
    sh_degree : int
    campos : torch.Tensor
    prefiltered : bool
    debug : bool
    antialiasing : bool
    futhark_server : Server
    gt_image: torch.Tensor

class GaussianRasterizerFuthark(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings


    def forward(self, means3D, means2D, opacities, shs = None, colors_precomp = None, scales = None, rotations = None, cov3D_precomp = None):
        
        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')
        
        if ((scales is None or rotations is None) and cov3D_precomp is None) or ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')
        
        if shs is None:
            shs = torch.Tensor([])
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])

        # Invoke C++/CUDA rasterization routine
        return rasterize_gaussians(
            means3D,
            means2D,
            shs,
            colors_precomp,
            opacities,
            scales, 
            rotations,
            cov3D_precomp,
            raster_settings, 
        )

