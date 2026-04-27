from futhark_3dgs.util import look_at, getProjectionMatrix
import numpy as np
from futhark_3dgs import Futhark_Rasterization_Server
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

W = 1000

output_vars = ['dmeans3d', 'dmeans2d', 'dcolors', 'dopacities', 'dscales', 'drotations', 'pix', 'radii', 'loss']

fovx = 1 #1.4028140929797817
fovy = 1 #0.8753571332164317
cam_pos = np.array([1,0,1.0], dtype=np.float32)
target = np.array([0,0,0.0], dtype=np.float32)
view_matrix = np.array(look_at(cam_pos, target), dtype= np.float32).T
proj_matrix = getProjectionMatrix(0.01, 100, fovx, fovy).T
#fused_proj = np.array(proj_matrix @ view_matrix, dtype=np.float32) # this may have to go the other way around
fused_proj = np.array(view_matrix @ proj_matrix, dtype=np.float32) # this may have to go the other way around

# just test to make sure that our gaussian is on screen
mean3d = [0,-0.1,0,1]
hom = fused_proj.T @ np.array(mean3d)
ndc = hom / (hom[3] + 1e-7)
pix = ((ndc+1) * W - 1) * 0.5   
print(pix)

inputs = {
    'bg':           np.array([0,0,0],                       dtype=np.float32),
    'means3d':      np.array([mean3d[:3]],                  dtype=np.float32),
    'colors':       np.array([[1,0,0]],                     dtype=np.float32),
    'opacities':    np.array([[0.01]],                         dtype=np.float32),
    'scales':       np.array([[0.1,0.5,0.1]],               dtype=np.float32),
    'rotations':    np.array([[0.0,0.0,0.1,0.8]],                   dtype=np.float32),
    'viewmatrix':   view_matrix,
    'projmatrix':   fused_proj,
    'tanfovx':      np.float32(np.tan(fovx*0.5)),
    'tanfovy':      np.float32(np.tan(fovy*0.5)),
    'image_height' : np.int64(W),
    'image_width':  np.int64(W),
    'ssim_kernel_size': np.int32(11),
    'ssim_kernel_sigma': np.float32(1.5),
    'gt_image':     np.zeros(shape=(W,W,3), dtype=np.float32),
    'lambda' :      np.float32(0.2)
    }

with Futhark_Rasterization_Server() as server:
    any_succeed = True
    n = 100000
    while any_succeed:
        inputs['means3d'] = np.array(np.random.random_sample((n,3)) - 0.5, dtype=np.float32)
        inputs['colors'] = np.array(np.random.random_sample((n,3)), dtype=np.float32)
        inputs['opacities'] = np.array(np.random.random_sample((n,1)), dtype=np.float32)
        inputs['scales'] = np.array(0.001 * np.random.random_sample((n,3)), dtype=np.float32)
        inputs['rotations'] = np.array(2*np.random.random_sample((n,4)) - 1, dtype=np.float32)

        # test the input
        for name, value in inputs.items():
            server.put_value(name, value)
        for f in ['grad', 'grad2', 'grad3']: 
            print(f'{f}: {n}')
            try:    
                server.cmd_call(f, *output_vars, *inputs.keys())
                pix = server.get_value('pix')
            except:
                print(f'{f} failed with {n} gaussians')
            server.cmd_free(*output_vars)

        server.cmd_free(*inputs.keys())
        plt.imsave(f"./images/output{n}.png", pix)
        n = int(n*2)


#srun --partition=gpu --gres=gpu:titanrtx:1 --time=01:00:00 --mem=12G --cpus-per-task=2 --pty bash
