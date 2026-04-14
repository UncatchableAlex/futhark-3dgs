import numpy as np
import json
import futhark_server
from tqdm import tqdm
import matplotlib.pyplot as plt


# the whole point of this script is to test the accuracy of grad

# provide file paths to relevant files
image_path = '/home/mjk711/gaussian-splatting/tandt/train/images/00124.jpg'
rasterizer_inps = '../rasterizer_inps'
rasterizer_path = '../futhark_rasterizer/rasterizer'

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

with open(f'{rasterizer_inps}/debug_rasterizer_settings.json', 'r') as f:
    json_data = json.load(f)
    inps.update(json_data)

test_forward = False

# number of Gaussians
n = 200000

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

with futhark_server.Server(rasterizer_path) as server:
    # store each input as a named variable
    for name, value in inputs.items():
        server.put_value(name, value)

    gt = np.array(plt.imread(image_path), dtype=np.float32)
    server.cmd_free('gt_image')
    
    # normalize jpg values upon loading. we need rbg values in [0,1] but jpg has [0,255]
    server.put_value('gt_image',np.array(gt/255, dtype=np.float32))
    output_vars = ['dmeans3d', 'dmeans2d', 'dcolors', 'dopacities', 'dscales', 'drotations', 'pix', 'radii', 'loss']

    # warmup run
    server.cmd_call("grad", *output_vars, *inputs.keys())

    dmeans3d = server.get_value('dmeans3d')
    dmeans2d = server.get_value('dmeans2d')
    dcolors = server.get_value('dcolors')
    dscales = server.get_value('dscales')
    drotations = server.get_value('drotations')
    pix = server.get_value('pix')

    # how many samples we observe:
    m = 5
    print('dmeans3d', dmeans3d[:m])
    print('demans2d', dmeans2d[:m])
    print('dcolors', dcolors[:m])
    print('dscales', dscales[:m])
    print('drotations', drotations[:m]) 

    plt.imshow(pix)
    plt.axis("off")
    plt.savefig(f'image.png', bbox_inches="tight", pad_inches=0)   