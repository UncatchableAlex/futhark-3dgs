import numpy as np
import json
import subprocess
import futhark_server
from tqdm import tqdm
from futhark_3dgs.util import look_at
from futhark_3dgs import Futhark_Rasterization_Server

output_dir = './image_outputs'
rasterizer_inps = '../rasterizer_inps'

# the number of frames to render
frames = 120
batch_size = 30

# the fps of the resulting video
fps = 15

# how many camera z-axes we are away from our target
lambd = 1

# the rotation part of the view matrix we were given
R = np.array([[ -0.9923,  0.0558,  0.1102 ],
[  0.0404,  0.9896, -0.1378 ],
[ -0.1167, -0.1323, -0.9843 ]])

# the translation part of the view matrix we were given
t = np.array([0.8557, 0.0910, 3.4138])


# compute camera intrinsics
given_eye = -R.T @ t
look_dir = -R[:,2] # camera's z axis
up = R[:,1] # camera's y axis
given_target = given_eye + (lambd*look_dir)

# distance from origin on xz plane
#r = np.linalg.norm([(lambd*look_dir)[0], (lambd*look_dir)[2]])
r = np.linalg.norm([given_eye[0], given_eye[2]])

# radians from target on zx plane
given_rads = np.arctan2(given_eye[2], given_eye[0]) / (2*np.pi)

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
    with open(f'{rasterizer_inps}/debug_{np_name}.npy', 'rb') as f:
        np_array = np.load(f)
        inps[np_name] = np_array

with open(f'{rasterizer_inps}/{json_name}', 'r') as f:
    json_data = json.load(f)
    inps.update(json_data)

n = 200_000 # how many gaussians we want to render


test_forward = False

view_matrices = np.zeros((batch_size, 4,4), dtype=np.float32)
proj_matrices = np.zeros((batch_size, 4,4), dtype=np.float32)

# prep  inputs as numpy arrays with correct dtypes
inputs = {
    'bg':           np.array([0,0,0],                       dtype=np.float32),
    'means3D':      np.array(inps['means3D'].tolist()[:n],     dtype=np.float32),
    'colors':       np.array(inps['colors_precomp'].tolist()[:n], dtype=np.float32),
    'opacities':    np.array(inps['opacities'].tolist()[:n],   dtype=np.float32),
    'scales':       np.array(inps['scales'].tolist()[:n],      dtype=np.float32),
    'rotations':    np.array(inps['rotations'].tolist()[:n],   dtype=np.float32),
    'viewmatrices':   np.array(inps['viewmatrix'],   dtype=np.float32),
    'projmatrices':   np.array(inps['projmatrix'],   dtype=np.float32),
    'tanfovx':      np.float32(inps['tanfovx']),
    'tanfovy':      np.float32(inps['tanfovy']),
    'image_height': np.int64(inps['image_height']),
    'image_width':  np.int64(inps['image_width']),
}    

# extract the actual projection matrix from the fused projmatrix
true_proj = inputs['projmatrices'] @ np.linalg.inv(inputs['viewmatrices'])


with Futhark_Rasterization_Server() as server:

    # store each input as a named variable
    for name, value in inputs.items():
        server.put_value(name, value)

    height = int(inputs['image_height'])
    width = int(inputs['image_width'])

    # chatgpt ffmpeg magic
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",  # overwrite
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}",
        "-r", f"{fps}",  # framerate
        "-i", "-",   # stdin
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",  # pad to satisfy H.264
        "-an",
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        "output.mp4"
    ]

    ffmpeg_proc = subprocess.Popen(
        ffmpeg_cmd, 
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL, # silence annoying ffmpeg stdout/stderr chatter
        stderr=subprocess.DEVNULL)

    i = 0
    pbar = tqdm(total=frames)

    while i < frames:

        # fill view_matrices and proj_matrices
        j = 0
        while j < np.minimum(batch_size, frames - i):
            # no idea how we chose these magic numbers

            eye = np.array([
                r * np.cos(np.pi * 2 * (given_rads + ((i+j)/(30*frames)))),       # X
                given_eye[1],                                                 # height above ground (constant)
                r * np.sin(np.pi * 2 * (given_rads + ((i+j)/(30*frames))))])      # Z
            
            # t = (i+j) / frames
            # eye = given_eye + t * (given_target - given_eye)
            view_matrices[j] = look_at(eye, given_target,up)
            proj_matrices[j] = true_proj @ view_matrices[j]
            j += 1
        # free both view and proj matrix
        server.cmd_free('viewmatrices')
        server.cmd_free('projmatrices')

        # put new view and projection matrices in
        server.put_value('viewmatrices', view_matrices)
        server.put_value('projmatrices', proj_matrices)
        # call the function: outputs first, then inputs
        server.cmd_call(
            "batch_rasterize",
            'pixels',                 
            *inputs.keys()
        )
        result = server.get_value('pixels')
        # force-feed this batch of rasterized images into ffmpeg
        for j in range(np.minimum(batch_size, frames - i)):
            frame = result[j]

            # ensure uint8 RGB
            if frame.dtype != np.uint8:
                frame = (255 * np.clip(frame, 0, 1)).astype(np.uint8)

            ffmpeg_proc.stdin.write(frame.tobytes())
            pbar.update(1)
        
        # free our result
        server.cmd_free('pixels')

        i += j + 1

    # close our ffmpeg process
    ffmpeg_proc.stdin.close()
    ffmpeg_proc.wait()
    pbar.close()