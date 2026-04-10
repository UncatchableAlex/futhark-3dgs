import numpy as np
import json
import futhark_server
from tqdm import tqdm
import matplotlib.pyplot as plt

json_name = 'debug_rasterizer_settings.json'
image_dir = '/home/mjk711/gaussian-splatting/tandt/train/images'
rasterizer_inps = './rasterizer_inps'

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
    with open(f'{rasterizer_inps}/debug_{np_name}.npy', 'rb') as f:
        np_array = np.load(f)
        inps[np_name] = np_array

with open(f'./rasterizer_inps/{json_name}', 'r') as f:
    json_data = json.load(f)
    inps.update(json_data)

test_forward = False

# All Gaussians
n = inps['means3D'].shape[0]

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
    'gt_image': np.array(inps['means3D'].tolist()[:n],     dtype=np.float32),
    'lambda' :      np.float32(0.2)}

with futhark_server.Server('./rasterizer') as server:
    # store each input as a named variable
    for name, value in inputs.items():
        server.put_value(name, value)

    filename = f'00124.jpg'
    gt = np.array(plt.imread(f'{image_dir}/{filename}'), dtype=np.float32)
    server.cmd_free('gt_image')
    server.put_value('gt_image',np.array(gt/255, dtype=np.float32))
    server.cmd_call(
        "grad",
        'dmeans', 
        'dcolors', 
        'dopacities', 
        'dscales', 
        'drotations', 
        'loss', 
        'radii',                
        *inputs.keys()
    )
    result = server.get_value('loss')
    print(filename, result)
