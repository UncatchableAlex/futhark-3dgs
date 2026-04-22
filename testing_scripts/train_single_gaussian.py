from futhark_3dgs.util import look_at, getProjectionMatrix
import numpy as np
from futhark_3dgs import Futhark_Rasterization_Server
import matplotlib.pyplot as plt
from PIL import Image
from prettytable import PrettyTable
import subprocess



output_vars = ['dmeans3d', 'dmeans2d', 'dcolors', 'dopacities', 'dscales', 'drotations', 'pix', 'radii', 'loss']
output_vars_2 = ['dmeans3d', 'dcolors', 'dopacities', 'dscales', 'drotations', 'pix', 'radii', 'loss']

grads = ['dmeans3d', 'dmeans2d', 'dcolors', 'dopacities', 'dscales', 'drotations']
grads_2 = ['dmeans3d', 'dcolors', 'dopacities', 'dscales', 'drotations']

def truncate_float_arr(arr):
    middle = ", ".join(f"{x:.4f}" for x in arr)
    return f'({middle})'

def finite_difference(server, inputs, test_var):
    for name, value in inputs.items():
        server.put_value(name, value)

    server.cmd_call('grad2', *output_vars_2, *inputs.keys())
    ad_grad_2 = server.get_value('d' + test_var)[0]
    # clear the output
    for var in output_vars_2:
        server.cmd_free(var)
    
    # we will track the finite difference loss gradient w.r.t. test_var using 10 different finite differences:
    finite_differences = []
    for x in np.logspace(-6, -4, 5):
        finite_difference = []
        m = inputs[test_var].shape[1]
        for i in range(m):
            # get one-hot perturbance vector
            server.cmd_free(test_var)
            x_vec = np.zeros(m, dtype=np.float64)
            x_vec[i] = x

            # perterb in the positive direction
            server.put_value(test_var, inputs[test_var] + x_vec)
            server.cmd_call('grad2', *output_vars_2, *inputs.keys())
            perturbed_loss_pos = server.get_value('loss')
            # clear the output
            for var in output_vars_2:
                server.cmd_free(var)

            # free the test_variable
            server.cmd_free(test_var)

            # perterb in the negative direction
            server.put_value(test_var, inputs[test_var] - x_vec)
            server.cmd_call("grad2", *output_vars_2, *inputs.keys())
            perturbed_loss_neg = server.get_value('loss')
            # clear the output
            for var in output_vars_2:
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
cam_pos = np.array([1,0,1.0], dtype=np.float64)
target = np.array([0,0,0.0], dtype=np.float64)
view_matrix = np.array(look_at(cam_pos, target), dtype= np.float64).T
proj_matrix = getProjectionMatrix(0.01, 100, fovx, fovy).T
#fused_proj = np.array(proj_matrix @ view_matrix, dtype=np.float64) # this may have to go the other way around
fused_proj = np.array(view_matrix @ proj_matrix, dtype=np.float64) # this may have to go the other way around

# just test to make sure that our gaussian is on screen
mean3d = [0,-0.1,0,1]
hom = fused_proj.T @ np.array(mean3d)
ndc = hom / (hom[3] + 1e-7)
pix = ((ndc+1) * 200 - 1) * 0.5   
print(pix)

# gt = np.zeros(shape=(200,200,3), dtype=np.float64)
# gt[:,:] = [1,0,0] # set to all blue pixels

gt = np.asarray(Image.open('./images/gt_gaussian.png'), dtype=np.float64)/255


inputs = {
    'bg':           np.array([0,0,0],                       dtype=np.float64),
    'means3d':      np.array([mean3d[:3]],                  dtype=np.float64),
    'colors':       np.array([[1,0,0]],                     dtype=np.float64),
    'opacities':    np.array([[1]],                         dtype=np.float64),
    'scales':       np.array([[0.1,0.1,0.1]],               dtype=np.float64),
    'rotations':    np.array([[0,0,0,0]],                   dtype=np.float64),
    'viewmatrix':   view_matrix,
    'projmatrix':   fused_proj,
    'tanfovx':      np.float64(np.tan(fovx*0.5)),
    'tanfovy':      np.float64(np.tan(fovy*0.5)),
    'image_height' : np.int64(200),
    'image_width':  np.int64(200),
    'ssim_kernel_size': np.int32(11),
    'ssim_kernel_sigma': np.float64(1.5),
    'gt_image': gt,
    'lambda' :      np.float64(0.2)
    }
# for key in inputs.keys():
#     try:
#         inputs[key] = inputs[key].tolist()
#     except:
#         pass
# futhark_input = " ".join(str(val) for key,val in inputs.items())
# # print(futhark_input)
# # raise Exception('stop')

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
    lr = 0.001
    t = PrettyTable()
    t.field_names = [
        'var',
        'AD2',
        'FD',
        'z-score2']
    (ad2, fd, sig, zscore2) = finite_difference(server, inputs, 'means3d')

    t.add_row([
        'means3d', 
        truncate_float_arr(ad2), 
        truncate_float_arr(fd), 
        truncate_float_arr(zscore2)])
    print('##########################################  GRAD2 (Ground truth reference)  #############################################')
    print(t)
    
    output_vars = ['dmeans3d', 'dmeans2d', 'dmeans2d_fd', 'dcolors', 'dopacities', 'dscales', 'drotations', 'pix', 'radii', 'loss']
    
    # run
    for name, value in inputs.items():
        server.put_value(name, value)
        
    server.cmd_call("grad", *output_vars, *inputs.keys())
    print('##########################################  GRAD (suspicious function) #############################################')
    outs1 = {}
    for var in output_vars:
        outs1[var] = server.get_value(var)
        server.cmd_free(var)
        if np.array([outs1[var]]).flatten().shape[0] < 100:
            print(f'{var}: {outs1[var]}')
