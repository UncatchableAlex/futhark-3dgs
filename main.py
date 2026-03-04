import numpy as np
import json
import subprocess
import time



json_name = 'debug_rasterizer_settings.json'
np_names = ['colors_precomp','means3D', 'scales','cov3Ds_precomp', 'opacities', 'rotations', 'sh']


# load the inputs that get fed to the rasterizer.
inps = {}
for np_name in np_names:
    with open(f'./rasterizer_inps/debug_{np_name}.npy', 'rb') as f:
        np_array = np.load(f)
        inps[np_name] = np_array
        # print(f"{np_name}: {np_array.shape, np_array.dtype}")

with open(f'./rasterizer_inps/{json_name}', 'r') as f:
    json_data = json.load(f)
    inps.update(json_data)
    # for key, value in json_data.items():
    #     print(f'{key}: {type(value)}')



inps_data = [
    inps['bg'],
    inps['means3D'].tolist(),
    inps['colors_precomp'].tolist(),
    inps['opacities'].tolist(),
    inps['scales'].tolist(),
    inps['rotations'].tolist(),
    float(inps['scale_modifier']),
    #[[[0,0,0],[0,0,0],[0,0,0]]],
    inps['viewmatrix'],
    inps['projmatrix'],
    float(inps['tanfovx']),
    float(inps['tanfovy']),
    int(inps['image_height']),
    int(inps['image_width']),
    #[[[0,0,0],[0,0,0],[0,0,0]]],
    int(inps['sh_degree']),
    inps['campos'],
    'false',
    'false',
]

futhark_input = "\n".join(str(item) for item in inps_data)

start = time.time()
proc = subprocess.run(
    ["./rasterizer", "-e", "rasterize"],
    input=futhark_input.encode("utf-8"),
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE
)

# check for errors
if proc.returncode:
    print("futhark error:", proc.stderr.decode())
    raise RuntimeError("futhark execution failed")


output = proc.stdout.decode()
print(f'result: {output}')