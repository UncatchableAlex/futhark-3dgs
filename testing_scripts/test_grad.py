import numpy as np
import json
import futhark_server
from tqdm import tqdm
import matplotlib.pyplot as plt


# the whole point of this script is to test the accuracy of grad

def cosine_similarity(a, b):
    a_flat = a.flatten()
    b_flat = b.flatten()
    return np.dot(a_flat, b_flat) / (np.linalg.norm(a_flat) * np.linalg.norm(b_flat))

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
    'bg':           np.array([0,0,0],                       dtype=np.float64),
    'means3D':      np.array(inps['means3D'].tolist()[:n],     dtype=np.float64),
    'colors':       np.array(inps['colors_precomp'].tolist()[:n], dtype=np.float64),
    'opacities':    np.array(inps['opacities'].tolist()[:n],   dtype=np.float64),
    'scales':       np.array(inps['scales'].tolist()[:n],      dtype=np.float64),
    'rotations':    np.array(inps['rotations'].tolist()[:n],   dtype=np.float64),
    'viewmatrix':   np.array(inps['viewmatrix'],               dtype=np.float64),
    'projmatrix':   np.array(inps['projmatrix'],               dtype=np.float64),
    'tanfovx':      np.float64(inps['tanfovx']),
    'tanfovy':      np.float64(inps['tanfovy']),
    'image_height': np.int64(inps['image_height']),
    'image_width':  np.int64(inps['image_width']),
    'ssim_kernel_size': np.int32(11),
    'ssim_kernel_sigma': np.float64(1.5),
    'gt_image': np.array(inps['means3D'].tolist()[:n],     dtype=np.float64),
    'lambda' :      np.float64(0.2)}

with futhark_server.Server(rasterizer_path) as server:
    # store each input as a named variable
    for name, value in inputs.items():
        server.put_value(name, value)

    gt = np.array(plt.imread(image_path), dtype=np.float64)
    server.cmd_free('gt_image')
    
    # normalize jpg values upon loading. we need rbg values in [0,1] but jpg has [0,255]
    server.put_value('gt_image',np.array(gt/255, dtype=np.float64))
    output_vars = ['dmeans3d', 'dmeans2d', 'dcolors', 'dopacities', 'dscales', 'drotations', 'pix', 'radii', 'loss']
    
    # run
    server.cmd_call("grad", *output_vars, *inputs.keys())

    outs1 = {}

    # how many samples we observe:
    m = 5
    for var in output_vars:
        outs1[var] = server.get_value(var)
        server.cmd_free(var)
      #  print(f'{var}: {outs1[var][:m]}')

    output_vars = ['dmeans3d', 'dcolors', 'dopacities', 'dscales', 'drotations', 'pix', 'radii', 'loss']
    
    # run
    server.cmd_call("grad2", *output_vars, *inputs.keys())
    
    outs2 = {}
    for var in output_vars:
        outs2[var] = server.get_value(var)
        server.cmd_free(var)

    print("cosine similarity: dmeans3d ", cosine_similarity(outs1['dmeans3d'], outs2['dmeans3d']))
    
    print("magnitude dmeans3d grad: ", np.linalg.norm(outs1['dmeans3d'].flatten()))
    print("magnitude dmeans3d grad2: ", np.linalg.norm(outs2['dmeans3d'].flatten()))

    # print("cosine similarity: dmeans2d ", cosine_similarity(dmeans2d_1, dmeans2d_2))

    # print("magnitude dmeans2d grad: ", np.linalg.norm(dmeans2d_1.flatten()))
    # print("magnitude dmeans2d proj: ", np.linalg.norm(dmeans2d_2.flatten()))

    
    # print("cosine similarity: dcolors ", cosine_similarity(outs1['dcolors'], outs2['dcolors']))
    # print("cosine similarity: dscales ", cosine_similarity(dscales_1, dscales_2))
    # print("cosine similarity: drotations ", cosine_similarity(drotations_1, drotations_2))

    # plt.imshow(pix_2)
    # plt.axis("off")
    # plt.savefig(f'image.png', bbox_inches="tight", pad_inches=0)   