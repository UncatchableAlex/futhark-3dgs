import numpy as np
from futhark_3dgs.util import getProjectionMatrix


view_matrix = np.array([[-0.99234504,  0.05581591,  0.11016339,  0.        ],
 [ 0.0403651,   0.9896362,  -0.1378075,   0.        ],
 [-0.11671352, -0.13230583, -0.98431355,  0.        ],
 [ 0.8556888,   0.0910202,   3.4138324,   1.        ]])


view_matrix_no_translation = view_matrix[:3,:3]

mean = np.array([0,-0.1,0])
mean_cam = mean @ view_matrix_no_translation  # the real 3D mean, not the gradient
x, y, z = mean_cam[0], mean_cam[1], mean_cam[2]

# gradient of loss w.r.t. 3D mean (what you have)
dmean = np.array([-0.0107, -0.0076, -0.0107])
dmean_cam = dmean @ view_matrix_no_translation

# recover dL/d(mean_2d) — just scale by z
# dL_dmean2d = np.array([
#     z * dmean_cam[0],
#     z * dmean_cam[1],
# ])
# print(dL_dmean2d)

dL_dmean2d = np.array([4.13364429e-05, -6.30420738e-03])

J = [
    [0.915244, 0.000000, -0.915244],
    [-0.064718, 1.294350, -0.064718]
]

dL_dm3d = dL_dmean2d@J

target = np.array([-0.0107, -0.0076, -0.0107])

print(target/dL_dm3d)
print(dL_dm3d/target)


