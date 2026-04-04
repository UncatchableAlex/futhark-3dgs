import numpy as np
import json
import subprocess
import time
import re
import matplotlib.pyplot as plt
import os



json_name = 'debug_rasterizer_settings.json'
np_names = ['colors_precomp','means3D', 'scales','cov3Ds_precomp', 'opacities', 'rotations', 'sh']


# load the inputs that get fed to the rasterizer.
inps = {}
for np_name in np_names:
    with open(f'./rasterizer_inps/debug_{np_name}.npy', 'rb') as f:
        np_array = np.load(f)
        inps[np_name] = np_array

with open(f'./rasterizer_inps/{json_name}', 'r') as f:
    json_data = json.load(f)
    inps.update(json_data)

n = 100000000 # how many gaussians we want to render
inps_data = [
    inps['bg'],
    inps['means3D'].tolist()[:n],
    inps['colors_precomp'].tolist()[:n],
    inps['opacities'].tolist()[:n],
    inps['scales'].tolist()[:n],
    inps['rotations'].tolist()[:n],
 #   float(inps['scale_modifier']),
    #[[[0,0,0],[0,0,0],[0,0,0]]],
    inps['viewmatrix'],
    inps['projmatrix'],
    float(inps['tanfovx']),
    float(inps['tanfovy']),
    int(inps['image_height']),
    int(inps['image_width']),
   # int(11),
   # 1.5,
   # []
    #[[[0,0,0],[0,0,0],[0,0,0]]],
#    int(inps['sh_degree']),
#    inps['campos'],
#    'false',
#    'false',
]

image_dir = '/mnt/c/users/alexm/2026_spring/Thesis/tandt/train/images'
for filename in os.listdir(image_dir):
    if filename.endswith('.jpg'):
        gt = np.array(plt.imread(f'{image_dir}/{filename}'), dtype=np.float32)
    else:
        continue
   # inps_data[-1] = (gt/255).tolist()

# assemble the parameters that we will jam into futhark's stdin
    futhark_input = "\n".join(str(item) for item in inps_data)
    print('inputting')
    proc = subprocess.run(
        ["./rasterizer", "-e", "rasterize"],
        input=futhark_input.encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    

    # futhark prints its debug tracing through stderr
    print("sterr: ", proc.stderr.decode())

    # check for errors
    # if proc.returncode:
    #     print("futhark error:", proc.stderr.decode())
    #     raise RuntimeError("futhark execution failed")

    # get rid of the extra f32 type indicators that futhark puts on every number in its output
    output = re.sub(r'f32', '', proc.stdout.decode())
    #print(output)

# turn the futhark output back into a python list. Save it to an image
    rgb = eval(output)
    plt.imshow(np.array(rgb))
    plt.axis("off")
    plt.savefig("image.png", bbox_inches="tight", pad_inches=0)
    break