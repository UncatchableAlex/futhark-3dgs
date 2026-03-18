import numpy as np
import json
import subprocess
import time
import re
import matplotlib.pyplot as plt
import futhark_server
from tqdm import tqdm



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
    V = np.array([
        [rot[0][0],rot[0][1],rot[0][2],x],
        [rot[1][0],rot[1][1],rot[1][2],y],
        [rot[2][0],rot[2][1],rot[2][2],z],
        [0,0,0,1]
    ])
    return np.array(np.transpose(V),  dtype=np.float32)


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
    'bg':           np.array(inps['bg'],                       dtype=np.float32),
    'means3D':      np.array(inps['means3D'].tolist()[:n],     dtype=np.float32),
    'colors':       np.array(inps['colors_precomp'].tolist()[:n], dtype=np.float32),
    'opacities':    np.array(inps['opacities'].tolist()[:n],   dtype=np.float32),
    'scales':       np.array(inps['scales'].tolist()[:n],      dtype=np.float32),
    'rotations':    np.array(inps['rotations'].tolist()[:n],   dtype=np.float32),
    'viewmatrix':   np.array(inps['viewmatrix'],               dtype=np.float32),
    'projmatrix':   np.array(inps['projmatrix'],               dtype=np.float32),
    'tanfovx':      np.float32(inps['tanfovx']),
    'tanfovy':      np.float32(inps['tanfovy']),
    'image_height': np.int32(inps['image_height']),
    'image_width':  np.int32(inps['image_width']),
}

with futhark_server.Server('./rasterizer') as server:

    # store each input as a named variable
    for name, value in inputs.items():
        server.put_value(name, value)

    frames = 20

    for i in tqdm(range(0,frames)):
        yaw = 2*np.pi*i/frames + np.pi
        pitch = 0
        roll = 0
        x = 5*np.cos(2*np.pi*i/frames)
        z = 5*np.sin(2*np.pi*i/frames)
        y = 0
        server.cmd_free('viewmatrix')
        V = view(yaw, pitch, roll, x, y, z)
        server.put_value('viewmatrix', V)

        #  call the function: outputs first, then inputs
        server.cmd_call(
            'rasterize',
            'output',                   
            *inputs.keys()
        )
        result = server.get_value('output')
        plt.imshow(result)
        plt.axis("off")
        plt.savefig(f'images/{i:02d}.png', bbox_inches="tight", pad_inches=0)
        server.cmd_free('output')
        
