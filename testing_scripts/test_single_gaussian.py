from futhark_3dgs.util import look_at, getProjectionMatrix
import numpy as np
from futhark_3dgs import Futhark_Rasterization_Server
import matplotlib.pyplot as plt
from PIL import Image
from prettytable import PrettyTable
import subprocess



output_vars_1 = ['dmeans3d', 'dmeans2d', 'dcolors', 'dopacities', 'dscales', 'drotations', 'pix', 'radii', 'loss']
output_vars_2 = ['dmeans3d', 'dmeans2d', 'dcolors', 'dopacities', 'dscales', 'drotations', 'pix', 'radii', 'loss']

def truncate_float_arr(arr):
    middle = ", ".join(f"{x:.4f}" for x in arr)
    return f'({middle})'

def finite_difference(server, inputs, f, test_var, output_vars):
    for name, value in inputs.items():
        server.put_value(name, value)

    server.cmd_call(f, *output_vars, *inputs.keys())
    ad_grad_2 = server.get_value('d' + test_var)[0]
    # clear the output
    for var in output_vars:
        server.cmd_free(var)
    
    # we will track the finite difference loss gradient w.r.t. test_var using 10 different finite differences:
    finite_differences = []
    for x in np.logspace(-6, -5, 5):
        finite_difference = []
        m = inputs[test_var].shape[1]
        for i in range(m):
            # get one-hot perturbance vector
            server.cmd_free(test_var)
            x_vec = np.zeros(m, dtype=np.float32)
            x_vec[i] = x

            # perterb in the positive direction
            server.put_value(test_var, inputs[test_var] + x_vec)
            server.cmd_call(f, *output_vars, *inputs.keys())
            perturbed_loss_pos = server.get_value('loss')
            # clear the output
            for var in output_vars:
                server.cmd_free(var)

            # free the test_variable
            server.cmd_free(test_var)

            # perterb in the negative direction
            server.put_value(test_var, inputs[test_var] - x_vec)
            server.cmd_call(f, *output_vars, *inputs.keys())
            perturbed_loss_neg = server.get_value('loss')
            # clear the output
            for var in output_vars:
                server.cmd_free(var)

            # track the finite difference derivative
            finite_difference.append((perturbed_loss_pos - perturbed_loss_neg) / (2*x))

        finite_differences.append(finite_difference)

    server.cmd_clear()
    avg = np.average(finite_differences, axis=0)
    std = np.std(finite_differences, axis=0)
    zscore_2 = (np.array(ad_grad_2) - avg) / std
    
    return  ad_grad_2, avg, std, zscore_2

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
pix = ((ndc+1) * 200 - 1) * 0.5   
print(pix)

# gt = np.zeros(shape=(200,200,3), dtype=np.float32)
# gt[:,:] = [1,0,0] # set to all blue pixels

gt = np.asarray(Image.open('/home/mjk711/gaussian-splatting/submodules/futhark-3dgs/testing_scripts/images/gt_gaussian.png'), dtype=np.float32)/255


inputs = {
    'bg':           np.array([0,0,0],                       dtype=np.float32),
    'means3d':      np.array([mean3d[:3]],                  dtype=np.float32),
    'colors':       np.array([[1,0,0]],                     dtype=np.float32),
    'opacities':    np.array([[1]],                         dtype=np.float32),
    'scales':       np.array([[0.1,0.5,0.1]],               dtype=np.float32),
    'rotations':    np.array([[0.0,0.0,0.1,0.8]],                   dtype=np.float32),
    'viewmatrix':   view_matrix,
    'projmatrix':   fused_proj,
    'tanfovx':      np.float32(np.tan(fovx*0.5)),
    'tanfovy':      np.float32(np.tan(fovy*0.5)),
    'image_height' : np.int64(200),
    'image_width':  np.int64(200),
    'ssim_kernel_size': np.int32(11),
    'ssim_kernel_sigma': np.float32(1.5),
    'gt_image': gt,
    'lambda' :      np.float32(0.2)
    }
# for key in inputs.keys():
#     try:
#         inputs[key] = inputs[key].tolist()
#     except:
#         pass
# futhark_input = " ".join(str(val) for key,val in inputs.items())

# print('inputting')
# proc = subprocess.run(
#     ["../futhark_rasterizer/rasterizer", "-e", "grad"],
#     input=futhark_input.encode("utf-8"),
#     stdout=subprocess.PIPE,
#     stderr=subprocess.PIPE
# )
#     # futhark prints its debug tracing through stderr
# print("sterr: ", proc.stderr.decode())

# #check for errors
# if proc.returncode:
#     print("futhark error:", proc.stderr.decode())
#     raise RuntimeError("futhark execution failed")

# # get rid of the extra f32 type indicators that futhark puts on every number in its output
# output = proc.stdout.decode()
# #print(output)

# raise Exception('stop')



with Futhark_Rasterization_Server() as server:

    for f in ['grad1', 'grad2', 'grad3']:
        print(f'##########################################  {f}  #############################################')
        t = PrettyTable()
        t.field_names = [
            'var',
            'AD',
            'FD',
            'z-score']
        
        for var in ['means3d', 'colors', 'opacities', 'scales', 'rotations']:
            (ad, fd, sig, zscore) = finite_difference(server, inputs, 'grad2', var, output_vars_1)

            t.add_row([
                var, 
                truncate_float_arr(ad), 
                truncate_float_arr(fd), 
                truncate_float_arr(zscore)])
        print(t) 
           
    # make the picture
    for name, value in inputs.items():
        server.put_value(name, value)
        
    server.cmd_call("grad", *output_vars_1, *inputs.keys())
    outs1 = {}
    for var in output_vars_1:
        outs1[var] = server.get_value(var)
        server.cmd_free(var)
        if np.array([outs1[var]]).flatten().shape[0] < 100:
            print(f'{var}: {outs1[var]}')

    plt.imsave("output.png", outs1['pix'])
