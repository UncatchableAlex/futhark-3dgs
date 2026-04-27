import numpy as np
import json
import futhark_server
from tqdm import tqdm
import matplotlib.pyplot as plt
import time


# the whole point of this script is to test the accuracy and performance of
# calculating gaussian gradients


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

ns = [100, 1000, 5000, 10000, 20000, 30000, 40000, 50000,100000,150000]
batch_size = 30
times = np.zeros((3,len(ns)))


with futhark_server.Server(rasterizer_path) as server:
    gt = np.array(plt.imread(image_path), dtype=np.float32) / 255.0
    for i,n in enumerate(ns):
        inputs = {
            'bg':           np.array([0,0,0],                               dtype=np.float32),
            'means3D':      np.array(inps['means3D'].tolist()[:n],          dtype=np.float32),
            'colors':       np.array(inps['colors_precomp'].tolist()[:n],   dtype=np.float32),
            'opacities':    np.array(inps['opacities'].tolist()[:n],        dtype=np.float32),
            'scales':       np.array(inps['scales'].tolist()[:n],           dtype=np.float32),
            'rotations':    np.array(inps['rotations'].tolist()[:n],        dtype=np.float32),
            'viewmatrix':   np.array(inps['viewmatrix'],                    dtype=np.float32),
            'projmatrix':   np.array(inps['projmatrix'],                    dtype=np.float32),
            'tanfovx':      np.float32(inps['tanfovx']),
            'tanfovy':      np.float32(inps['tanfovy']),
            'image_height': np.int64(inps['image_height']),
            'image_width':  np.int64(inps['image_width']),
            'ssim_kernel_size':  np.int32(11),
            'ssim_kernel_sigma': np.float32(1.5),
            'gt_image':     np.array(gt, dtype=np.float32),
            'lambda':       np.float32(0.2),
        }

        for name, value in inputs.items():
            server.put_value(name, value)

        output_vars = ['dmeans3d', 'dmeans2d', 'dcolors', 'dopacities', 'dscales', 'drotations', 'pix', 'radii', 'loss']
        print(f"n={n}")
        # loop each of our grad functions
        for j,f in enumerate(['grad', 'grad2', 'grad3']):
            # warmup run
            server.cmd_call(f, *output_vars, *inputs.keys())
            server.cmd_free(*output_vars)

            # timed runs
            start = time.perf_counter()
            for _ in range(batch_size):
                server.cmd_call(f, *output_vars, *inputs.keys())
                server.cmd_free(*output_vars)
            elapsed = time.perf_counter() - start
            times[j][i] = elapsed/batch_size
            print(f"f={f}: {elapsed:.4f}s")

        server.cmd_free(*inputs.keys())
        server.cmd_clear()


# srun --time=01:00:00 --partition=gpu --gres=gpu:1 --mem=8G --cpus-per-task=2 --pty bash