import numpy as np
import json
import subprocess
import time
import re
import matplotlib.pyplot as plt
import futhark_server
from tqdm import tqdm
import os


def view(yaw, pitch, roll, x, y, z):
    Y = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0,0,1]])
    P = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    R = np.array([
        [1,0,0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    rot = Y@P@R
    print(rot@rot.T)
    V = np.array([
        [rot[0][0],rot[0][1],rot[0][2],x],
        [rot[1][0],rot[1][1],rot[1][2],y],
        [rot[2][0],rot[2][1],rot[2][2],z],
        [0,0,0,1]
    ])
    return np.array(np.transpose(V),  dtype=np.float32)




def look_at(eye, target, up=np.array([0,1,0.0])):
    # which way is forward?
    forward = target - eye 
    forward /= np.linalg.norm(forward) # normalize

    # the "right" vector is orthogonal to forward and up
    right = np.cross(forward, up)
    right /= np.linalg.norm(right) # normalize

    # make sure that our up really is orthogonal to our other axes
    up = np.cross(right, forward)

    # use opengl style of -forward actually being forward
    R = np.stack([right, up, -forward])
    t = -R @ eye # this is the formula for translation

    V = np.eye(4, dtype=np.float32)
    V[:3,:3] = R # set the rotation part of the matrix
    V[:3, 3] = t # set the translation part

    print(V.T)

    return V.T


R = np.array([[ -0.9923,  0.0558,  0.1102 ],
[  0.0404,  0.9896, -0.1378 ],
[ -0.1167, -0.1323, -0.9843 ]])

t = np.array([0.8557, 0.0910, 3.4138])

eye = -R.T @ t
print(eye)
#raise Exception("Stop")

json_name = 'debug_rasterizer_settings.json'
np_names = [
    'colors_precomp',
    'means3D', 
    'scales',
    'cov3Ds_precomp',
    'opacities',
    'rotations',
    'sh']

# load the inputs that get fed to the rasterizer.
inps = {}
for np_name in np_names:
    with open(f'./rasterizer_inps/debug_{np_name}.npy', 'rb') as f:
        np_array = np.load(f)
        inps[np_name] = np_array

with open(f'./rasterizer_inps/{json_name}', 'r') as f:
    json_data = json.load(f)
    inps.update(json_data)

n = 10000000000 # how many gaussians we want to render


# prep  inputs as numpy arrays with correct dtypes
inputs = {
    'bg':           np.array([0,0,0],                       dtype=np.float32),
    'means3D':      np.array(inps['means3D'].tolist()[:n],     dtype=np.float32),
    'colors':       np.array(inps['colors_precomp'].tolist()[:n], dtype=np.float32),
    'opacities':    np.array(inps['opacities'].tolist()[:n],   dtype=np.float32),
    'scales':       np.array(inps['scales'].tolist()[:n],      dtype=np.float32),
    'rotations':    np.array(inps['rotations'].tolist()[:n],   dtype=np.float32),
    'viewmatrix':   np.array(inps['viewmatrix'],               dtype=np.float32),
    'projmatrix':   np.array(inps['projmatrix'],               dtype=np.float32),
    'tanfovx':      np.float32(inps['tanfovx']),
    'tanfovy':      np.float32(inps['tanfovy']),
    'image_height': np.int64(inps['image_height']),
    'image_width':  np.int64(inps['image_width']),
    'ssim_kernel_size': np.int32(11),
    'ssim_kernel_sigma': np.float32(1.5),
    'gt_image': np.array([[[0.0,0.0,0.0]]],                                dtype=np.float32),
    'lambda' : np.float32(0.2)
}

with futhark_server.Server('./rasterizer') as server:

    # store each input as a named variable
    for name, value in inputs.items():
        server.put_value(name, value)

    frames = 50
    image_dir = '/mnt/c/users/alexm/2026_spring/Thesis/tandt/train/images'
    for filename in os.listdir(image_dir):
        if filename.endswith('.jpg'):
            gt = np.array(plt.imread(f'{image_dir}/{filename}'), dtype=np.float32)
        else:
            continue
        server.cmd_free('gt_image')
        server.put_value('gt_image',np.array(gt/255, dtype=np.float32))
        print(*inputs.keys())
        #bg means3D colors opacities scales rotations viewmatrix projmatrix tanfovx tanfovy image_height image_width ssim_kernel_size ssim_kernel_sigma gt_image
        server.cmd_call(
            "grad",
            'output',                   
            *inputs.keys()
        )
        result = server.get_value('output')
        print(result.shape)
        raise Exception("stop")
#     for i in tqdm(range(0,frames)):
#         server.cmd_free('viewmatrix')
#         r = np.linalg.norm([1.24382517, 3.278445])
#         eye = np.array([r * np.cos(np.pi * 2 * (0.17 + (i/(30*frames)))), 0.31384408, r * np.sin(np.pi * 2 * (0.17 + (i/(30*frames))))])
#         target = np.array([1.3605, 0.4461, 4.2627])
#         V = look_at(eye,target)
#         server.put_value('viewmatrix', V)
# #         call the function: outputs first, then inputs
#         server.cmd_call(
#             "rasterize",
#             'output',                   
#             *inputs.keys()
#         )
#         result = server.get_value('output')
#         plt.imsh+ow(result)
#         plt.axis("off")
#         plt.savefig(f'images/{i:02d}.png', bbox_inches="tight", pad_inches=0)
#         server.cmd_free('output')
       # raise Exception("Stop")
        

# The view matrix we were given:
# [
#     [-0.9923450350761414, 0.055815912783145905, 0.11016339063644409, 0.0], 
#     [0.04036509618163109, 0.9896361827850342, -0.13780750334262848, 0.0], 
#     [-0.1167135238647461, -0.1323058307170868, -0.9843135476112366, 0.0], 
#     [0.8556888103485107, 0.09102020412683487, 3.413832426071167, 1.0]]


