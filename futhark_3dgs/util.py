import numpy as np

def look_at(eye, target, up=np.array([0,1,0.0])):
    # which way is forward?
    forward = target - eye 
    forward /= np.linalg.norm(forward) # normalize

    # the "right" vector is orthogonal to forward and up
    right = np.cross(forward, up)
    right /= np.linalg.norm(right) # normalize

    # make sure that our up really is orthogonal to our other axes
    up = np.cross(right, forward)

    # they DON'T use the opengl style of -z being forward. +z is forward, apparently
    R = np.stack([right, up, forward])
    t = -R @ eye # this is the formula for translation

    V = np.eye(4, dtype=np.float32)
    V[:3,:3] = R # set the rotation part of the matrix
    V[:3, 3] = t # set the translation part

    return V

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
  #  print(rot@rot.T)
    V = np.array([
        [rot[0][0],rot[0][1],rot[0][2],x],
        [rot[1][0],rot[1][1],rot[1][2],y],
        [rot[2][0],rot[2][1],rot[2][2],z],
        [0,0,0,1]
    ])
    return np.array(np.transpose(V),  dtype=np.float32)


# lightly adapted from the 3dgs util implementation by inria
# https://github.com/graphdeco-inria/gaussian-splatting/blob/main/utils/graphics_utils.py
def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = np.tan((fovY / 2))
    tanHalfFovX = np.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = np.zeros((4, 4))

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P