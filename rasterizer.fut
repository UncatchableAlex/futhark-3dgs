-- original cuda control flow from kerbl et al.:
--
-- entry: RasterizeGaussiansCuda (rasterize_points.cu)
--                  |
--                  v
--            forward (rasterizer_impl.cu)
--             /                  \
--            v                    v
--    preprocess (forward.cu)      render (forward.cu)
--          |                          |
--          v                          v
--    preprocessCUDA (forward.cu)    renderCUDA (forward.cu)  

import "util"

-- check if a point is in the view frustum with a quick heuristic approximation
def in_frustum (p: [3]f32) (view_matrix: [4][4]f32) : bool = 
        -- bring the point into camera space 
    let pc = transform_point_4x3 p view_matrix  

        -- check that the point isn't too close to the camera in the z-axis. 
        -- Remember that the z-axis points in the same direction as the camera
    in pc[2] <= 0.2 


-- given parameters for an 3D elipse (rotation and scaling), calculate the corresponding 
-- correlation matrix for a 3D gaussian. Note that because correlation matrices are symmetrical,
-- we get away with only returning the top half
def compute_cov_3D (scale: [3]f32) (rotation: [4]f32): [6]f32 =
    let R = quat_to_mat rotation
    let S = scale_to_mat scale
    let M = matmul_3x3 R S
    let sigma = matmul_3x3 (transpose M) M
    in [sigma[0][0], sigma[0][1], sigma[0][2], sigma[1][1], sigma[1][2], sigma[2][2]]


-- calculate the 2D coveriance matrix of a 3D gaussian
-- projected into screen space using equations 29 and 31
-- specified in "EWA Splatting" (Zwicker et al., 2002)
def compute_cov_2D 
    (mean: [3]f32) 
    (focal_x: f32) 
    (focal_y: f32) 
    (tan_fovx: f32) 
    (tan_fovy: f32) 
    (cov3D: [6]f32) 
    (viewmatrix: [4][4]f32) : [3]f32 = 

    -- FOV magic from kerbl et al. (??)
    let t = transform_point_4x3 mean viewmatrix
    let limx = 1.3 * tan_fovx
    let limy = 1.3 * tan_fovy
    let txtz = t[0] / t[2]
    let tytz = t[1] / t[2]
    let t0 = t[2] * f32.min limx (f32.max (-limx) txtz)
    let t1 = t[2] * f32.min limy (f32.max (-limy) tytz)

    -- equation 29 zwicker et al.
    let J = [
        [focal_x / t[2], 0, -(focal_x*t0) / (t[2]*t[2])],
        [0, focal_y/t[2], -(focal_y*t1)/(t[2]*t[2])],
        [0,0,0] -- we don't need this row because we will truncate to a 2x2 in a future step
    ]

    -- get the rotation part of the view matrix
    let W = [
        [viewmatrix[0][0], viewmatrix[0][1], viewmatrix[0][2]],
        [viewmatrix[1][0], viewmatrix[1][1], viewmatrix[1][2]],
        [viewmatrix[2][0], viewmatrix[2][1], viewmatrix[2][2]]
    ]

    let T = matmul_3x3 W J

    -- set up our cov3D from the upper-triangular representation
    let Vk = [
		[cov3D[0], cov3D[1], cov3D[2]],
		[cov3D[1], cov3D[3], cov3D[4]],
		[cov3D[2], cov3D[4], cov3D[5]]
    ]
    
    -- equation 31 zwicker et al.
    let cov = matmul_3x3 (matmul_3x3 (transpose T) Vk) T

    -- low pass filter (equation 33 zwicker et al.)
    -- we drop the 3rd row and column and return the top half of the symmetrical 2x2 cov mat
    in [cov[0][0] + 0.3, cov[0][1], cov[1][1] + 0.3]


-- preprocess each gaussian by determining which tiles it touches, the size of its largest radius
-- its conic, 
def preprocess 
   -- (P: i32) -- n
    --(D: i32) -- degree (for SH)
    --(M: i32) -- for SH (kinda mysterious)
    (mean: [3]f32)
    (scales: [3]f32)
    --(scale_modifier: f32)
    (rotations: [4]f32)
    --(opacities: [1]f32)
    --(shs: [][3][]f32)
    --(clamped: [3*n]: i8) -- for SH (big buffer)
    --(cov3D_precomp: [][3][3]f32) -- if python were giving us the 3d covariance matrices
    --(colors_precomp: [3]f32)
    (viewmatrix: [4][4]f32)
    (projmatrix: [4][4]f32)
    --(cam_pos: [3]f32)
    (W: i32) (H: i32)
    (focal_x: f32) (focal_y: f32)
    (tan_fovx: f32) (tan_fovy: f32)
    (idx: i32) (tilesize: i32): ([3]f32, [2]f32, []u64)  =  -- conic, mean2d, keys
        -- the homogeneous coordinates of the gaussian's mean
        let p_hom = transform_point_4x4 mean projmatrix
        let p_w = 1 / (p_hom[3] + 1e-7) -- avoid divide by 0

        -- the gassian mean in clip space
        let p_proj = [p_hom[0]*p_w, p_hom[1]*p_w, p_hom[2]*p_w]
        let cov3D = compute_cov_3D scales rotations
        let cov = compute_cov_2D mean focal_x focal_y tan_fovx tan_fovy cov3D viewmatrix
        let det = (cov[0] * cov[2] - cov[1] * cov[1])

        -- if the matrix is singular, give it a 0 covariance matrix so it doesn't ruin everything
        let det_inv = if det == 0 then 0 else 1/det 

        -- take the invserse of the 
        -- https://en.wikipedia.org/wiki/Invertible_matrix#Methods_of_matrix_inversion
        let conic = [cov[2]*det_inv, -cov[1]*det_inv, cov[0]*det_inv] 


        -- find the range occupied by the gaussian in screen space. We use 
        -- eigenvalues of the 2d covariance matrix. We want to compute a bounding 
        -- rectangle of the gaussian and find which screen-space tiles our gaussian
        -- overlaps
        let mid = 0.5 * (cov[0] + cov[2])
        let inner = mid * mid - det
        
        -- remember: the eigenvalues of gaussians represent the variances along their principle
        -- axis. Basically how streched they are along their axes (their eigenvectors).
        -- There's a trick for calculating eigenvalues: https://www.johndcook.com/blog/2021/05/07/trick-for-2x2-eigenvalues/
        let lambda1 = mid + f32.sqrt (f32.max 0.1 inner)
        let lambda2 = mid - f32.sqrt (f32.max 0.1 inner)

        -- get the radius of 3 standard deviations using the larger principle axis
        let std3 = f32.ceil <| 3 * f32.sqrt (f32.max lambda1 lambda2)
        let pix = [ndc_to_pix p_proj[0] W, ndc_to_pix p_proj[1] H]
        let (ylo,yhi,xlo,xhi) = get_rect_2d [mean[0], mean[1]] std3 W H tilesize

        -- make one key per tile that this gaussian overlaps. Each key should define that tile's instance of this gaussian
        -- in a total ordering based on tile # and depth. The key should also be able to identify this gaussian. We will
        -- accomplish this by making the key a u64 where the lower 32 bits are this gaussian's index, the next 16 bits are the depth
        -- and the most significant 16 bits are the tile number. 
        let base = ((u64.f32 p_proj[0]) << 32) & u64.i32 idx
        let colspan = xhi-xlo
        let tiles = (yhi-ylo) * (xhi-xlo)
        let keys = map (\i ->    -- must we do a map here? Should this be sequential?
            let row = ylo + i/colspan
            let col = xlo + (i % colspan)
            let tileidx = row * (W/tilesize) + col -- get the global index of this tile
            in base & (u64.i32 tileidx << 48)
        ) (1...tiles) 
        in (conic, pix, keys)

-- Our exposed rasterize function. In this function, we take the means, scales, 
-- rotations, colors, and opacities of the 3D Gaussians, as well as the camera 
-- parameters, and output a rasterized image. WHEW
entry rasterize [n]
    (background: [3]f32)
    (means3D: [n][3]f32)
    (colors: [n][3]f32)
    (opacities: [n][1]f32)
    (scales: [n][3]f32)
    (rotations: [n][4]f32)
    (scale_modifier: f32)
    --(cov3D_precomp: [][3][3]f32)
    (view_matrix: [4][4]f32)
    (proj_matrix: [4][4]f32)
    (tan_fovx: f32)
    (tan_fovy: f32)
    (image_height: i32)
    (image_width: i32)
    --(sh: [][3][]f32)
    (degree: i32)
    (campos: [3]f32)
    (prefiltered: bool)
    (debug: bool) :  [n]([3]f32, [2]f32, []u64) =
    --: [][]f32 =
      --  let P = n
        let H = image_height
        let W = image_width
        -- let meansum = map (\xs -> xs[0] + xs[1] + xs[2]) means3D
        -- let colorsum = map  (\xs -> xs[0] + xs[1] + xs[2]) colors
        -- in  [[dot meansum colorsum]]
        let focalx = (f32.i32 W) / (2*tan_fovx)
        let focaly = (f32.i32 H) / (2*tan_fovy)
        
        in map4 
            (\mean scale rotation i -> 
                (preprocess mean scale rotation view_matrix proj_matrix W H focalx focaly tan_fovx tan_fovy (i32.i64 i) 16) :> ([3]f32, [2]f32, []u64))
            means3D scales rotations (iota n)