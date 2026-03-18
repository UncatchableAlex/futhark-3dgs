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
import "lib/github.com/diku-dk/sorts/radix_sort"

-- check if a point is in the view frustum with a quick heuristic approximation
def in_frustum (p: [3]f32) (view_matrix: [4][4]f32) (proj_matrix: [4][4]f32): bool = 
        -- bring the point into camera space 
    let pc = transform_point_4x3 p view_matrix  
    let p_hom = transform_point_4x4 p proj_matrix
    let p_w = 1 / (p_hom[3] + 0.0000001) -- prevent divide-by-zeros 
    let p_projx = p_hom[0]*p_w
    let p_projy = p_hom[1]*p_w
        -- check that the point isn't too close to the camera in the z-axis. 
        -- Remember that the z-axis points in the same direction as the camera
    in pc[2] > 0.2 && p_projx > -1.3 && p_projx < 1.3 && p_projy > -1.3 && p_projy < 1.3


type Gaussian2D = {
    opacity:f32,
    color:[3]f32,
    mean:(f32,f32),
    conic:(f32,f32,f32),
    depth:f32,
    tiles_touching:i32,
    bounding_box:(i32,i32,i32,i32),
    valid:bool
}

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

    -- FOV magic from kerbl et al. 2023 (??)
    let t = transform_point_4x3 mean viewmatrix
    let limx = 1.3 * tan_fovx
    let limy = 1.3 * tan_fovy
    let txtz = t[0] / t[2]
    let tytz = t[1] / t[2]
    let t0 =  t[2] * f32.min limx (f32.max (-limx) txtz)
    let t1 =  t[2] * f32.min limy (f32.max (-limy) tytz)
    let t2 =  t[2]

    -- equation 29 zwicker et al.
    let J =  [
        [focal_x / t[2], 0, -(focal_x*t0) / (t2*t2)],
        [0, focal_y/t[2], -(focal_y*t1)/(t2*t2)],
        [0,0,0] -- we don't need this row because we will truncate to a 2x2 in a future step
    ]

    -- get the rotation part of the view matrix
    let W =  [
        [viewmatrix[0][0], viewmatrix[0][1], viewmatrix[0][2]],
        [viewmatrix[1][0], viewmatrix[1][1], viewmatrix[1][2]],
        [viewmatrix[2][0], viewmatrix[2][1], viewmatrix[2][2]]
    ]

    let T = matmul_3x3 J W

    -- set up our cov3D from the upper-triangular representation
    let Vk = [
		[cov3D[0], cov3D[1], cov3D[2]],
		[cov3D[1], cov3D[3], cov3D[4]],
		[cov3D[2], cov3D[4], cov3D[5]]
    ]
    
    -- equation 31 zwicker et al.
    let cov = matmul_3x3 (matmul_3x3 T Vk) (transpose T)

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
    (opacity: f32) (color: [3]f32)
    (tilesize: i32): Gaussian2D  =  -- conic, mean2d, bounding box, num_tiles, depth, opacity, color

        -- immediately return an empty value if the gaussian is not in view
        let in_view = in_frustum mean viewmatrix projmatrix

        let badG: Gaussian2D = {
            opacity=opacity,
            color=color,
            mean=(0,0), 
            conic=(0,0,0),
            depth=0, 
            tiles_touching=0, 
            bounding_box=(0,0,0,0),
            valid=false
            }
            -- if the gaussian isn't in the view frustum, return a badGaussian so it doesn't contribute to the rasterization
        in if not in_view then badG else
        -- the homogeneous coordinates of the gaussian's mean
        let p_hom = transform_point_4x4 mean projmatrix
        let p_w = 1 / (p_hom[3] + 1e-7) -- avoid divide by 0

        -- the gassian mean in clip space
        let p_proj = [p_hom[0]*p_w, p_hom[1]*p_w, p_hom[2]*p_w]
        let cov3D = compute_cov_3D scales rotations
        let cov = compute_cov_2D mean focal_x focal_y tan_fovx tan_fovy cov3D viewmatrix
        let det = cov[0] * cov[2] - cov[1] * cov[1]
        -- if the matrix is singular, return with an invalid gaussian so it doesn't contribute to the rasterization
        in if det == 0 then badG 
        else

        let det_inv =  1/det 

        -- take the invserse of the 2d covariance matrix
        -- https://en.wikipedia.org/wiki/Invertible_matrix#Methods_of_matrix_inversion
        let conic: (f32,f32,f32) = (cov[2]*det_inv, -cov[1]*det_inv, cov[0]*det_inv)


        -- find the range occupied by the gaussian in screen space. We use 
        -- eigenvalues of the 2d covariance matrix. We want to compute a bounding 
        -- rectangle of the gaussian and find which screen-space tiles our gaussian
        -- overlaps
        
        -- remember: the eigenvalues of gaussians represent the variances along their principle
        -- axis. Basically how streched they are along their axes (their eigenvectors).
        -- There's a trick for calculating eigenvalues: https://www.johndcook.com/blog/2021/05/07/trick-for-2x2-eigenvalues/
        let mid = 0.5 * (cov[0] + cov[2])
        let inner = mid * mid - det
        let lambda1 = mid + f32.sqrt (f32.max 0.1 inner)
        let lambda2 = mid - f32.sqrt (f32.max 0.1 inner)
        let larger_principle_axis = f32.sqrt (f32.max lambda1 lambda2)

        -- get the radius of 3 standard deviations using the larger principle axis
        let std3 = f32.ceil <| 3 * larger_principle_axis
        -- convert homogeneous coordinates to pixel coordinates
        let pix = (ndc_to_pix p_proj[0] W, ndc_to_pix p_proj[1] H)
        let depth:f32 = p_proj[2] -- keep depth info separate

        -- calculate a bounding box for the support of our gaussian in pixel space. Define the box with 3 std deviations from the mean
        -- in the most stretched direction (either lambda1 or lambda2) 
        let (ylo,yhi,xlo,xhi): (i32,i32,i32,i32) = get_rect_2d pix std3 W H tilesize
        let num_tiles:i32 = (yhi - ylo) * (xhi-xlo)
        in if num_tiles == 0 then badG else {
                opacity=opacity,
                color=color,
                mean=pix, 
                conic=conic,
                depth=depth, 
                tiles_touching=num_tiles, 
                bounding_box=(ylo,yhi,xlo,xhi),
                valid=true
                }


-- make one key per tile that this gaussian overlaps. Each key should define that tile's instance of this gaussian
-- in a total ordering based on tile # and depth. The key should also be able to identify this gaussian. We will
-- accomplish this by making the key a u64 where the lower 32 bits are the depth and the upper 32 bits are the tile
def generateElem [n] 
    -- (boxes: [n](i32,i32,i32,i32)) 
    -- (depths: [n]f32) 
    (gs: [n]Gaussian2D)
    (prefix_sum: [n]i32) 
    (W: i32)
    (tilesize: i32) 
    (idx: i64): (u64, i32) =
    -- search to find which gaussian this index refers to
    let bs = binary_search prefix_sum (i32.i64 idx) u64.i32 --binary_search
    let n_idx = if prefix_sum[bs] == i32.i64 idx then bs else bs - 1

    -- extract data for this gaussian
    let box: (i32, i32, i32, i32) = gs[n_idx].bounding_box
    let depth = gs[n_idx].depth
    let (ylo,_,xlo,xhi) = box
    --let box' = #[trace] [ylo,yhi,xlo,xhi]

    -- extract the tile that this gaussian/idx refers to
    -- this gaussian spans many tiles. Which of those tiles is this idx?
    let offset = (i32.i64 idx) - prefix_sum[n_idx] 
    let tile_y = if offset < 0 then ylo + (offset / (xhi - xlo)) else ylo + (offset / (xhi - xlo))
    let tile_x =  xlo + (offset % (xhi - xlo))
    let tile = tile_y * (W/tilesize) + tile_x
    let upper =  ((u64.i32 tile) u64.<< 32)
    let key = upper | (u64.u32 <| f32.to_bits depth)
    in (key,n_idx)

-- Calculate the color of each pixel!
def pixel_color [n] [m]
    (tilesize: i64) 
    (W: i64)
    (sorted_keys: [m]u64) 
    (sorted_indices: [m]i32)
    (bg: [3]f32)
    (gs: [n]Gaussian2D)
    (pix_x: i64)
    (pix_y: i64) : [3]f32 = --(f32, f32, f32)  = -- output: rgb 
        -- our pixel's tile
        let tile_y = pix_y / tilesize
        let tile_x = pix_x / tilesize
        let tile = tile_y * (W / tilesize) + tile_x


        let start = binary_search sorted_keys ((u64.i64 tile) << 32) id
        let end = binary_search sorted_keys ((u64.i64 (tile + 1)) << 32) id

        -- This whole for-loop is just equation 2 from the 3dgs paper (Kerbl et al. 2023)
        let ((pr, pg, pb), bg_T, _) = loop ((r,g,b), T, i) = ((0.0f32, 0.0, 0.0), 1.0, start) while T > 0.0001 && i < end do
            let mean = gs[sorted_indices[i]].mean
            let opacity = gs[sorted_indices[i]].opacity
            let g_color = gs[sorted_indices[i]].color
            let conic = gs[sorted_indices[i]].conic
            
            -- the distance of this pixel from the ith gaussian's mean
            let dx = f32.i64 pix_x - mean.0
            let dy = f32.i64 pix_y - mean.1

            -- gaussian equation. See eq 19 from Zwicker et al. 2001
            let power = -0.5 * (conic.0*dx*dx + 2*conic.1*dx*dy + conic.2*dy*dy)
            in 
                if power > 0 
                then ((r,g,b), T, i+1) else
            let alpha = f32.min 0.99 (opacity * f32.exp power)

            -- We don't include gaussians with tiny alphas. This is to reduce numerical instability
            -- when differentiating. See 3DGS Kerbl et al. 2023 appendix
            in if alpha < 1 / 255 
                then ((r,g,b), T, i+1) 
                else 

            let T' = T * (1 - alpha)  -- calculate new transmittance
            
            -- accumulate alpha from this gaussian into each color band
            let color' = (            
                g_color[0] * alpha * T + r, 
                g_color[1] * alpha * T + g, 
                g_color[2] * alpha * T + b)
            in if T' < 0.0001 then ((r,g,b), T', i+1) else (color', T', i+1)

        -- add the color contribution of the background
        in [
            f32.min 1 (bg_T * bg[0] + pr), 
            f32.min 1 (bg_T * bg[1] + pg), 
            f32.min 1 (bg_T * bg[2] + pb)]
        


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
   -- (scale_modifier: f32)
    --(cov3D_precomp: [][3][3]f32)
    (view_matrix: [4][4]f32)
    (proj_matrix: [4][4]f32)
    (tan_fovx: f32)
    (tan_fovy: f32)
    (image_height: i32)
    (image_width: i32)
    --(sh: [][3][]f32)
  --  (degree: i32)
  --  (campos: [3]f32)
   -- (prefiltered: bool)
   -- (debug: bool) --:  [n]([3]f32, [2]f32, [3]u64) =
                    : [i64.i32 image_height][i64.i32 image_width][3]f32 = 
                    --:f32 = 
        let H = image_height
        let W = image_width
        let tilesize = 16
        let focalx = (f32.i32 W) / (2*tan_fovx)
        let focaly = (f32.i32 H) / (2*tan_fovy)
        let proj_matrix' = transpose proj_matrix
        let view_matrix' = transpose view_matrix

        -- preprocess each gaussian in parallel. This generates our list of Gaussian2D records.
        -- Cull any gaussians not in frame
        let preprocessed = filter (\g -> g.depth > 0) <| map5
            (\mean scale rotation opacity color -> 
                preprocess mean scale rotation view_matrix' proj_matrix' W H focalx focaly tan_fovx tan_fovy opacity[0] color tilesize)
            means3D scales rotations opacities colors

        let num_tiles = map (\g -> g.tiles_touching) preprocessed

        -- get the prefix sum of the number of tiles each gaussian touches
        let prefix_sum = rotate (-1) <| scan (+) 0 num_tiles
        let unsorted_size = prefix_sum[0]
        let prefix_sum[0] = 0 -- fix the first element of the prefix sum since rotate will have messed it up

        let sorted_list = blocked_radix_sort_by_key
            256i16                 -- block size for a100
            (\(k,_) -> k)          -- extract the key
            64                     -- sort on 64 bit keys
            (\i k -> i32.u64 (k >> (u64.i32 i) & 1)) -- get the ith bit
            (map (generateElem preprocessed prefix_sum W tilesize) (iota <| i64.i32 unsorted_size)) -- the kv pairs we are sorting where the key is (tile, depth) as a u64
        
        -- the sorted gaussian keys are the keys of each copy of each gaussian in the big list.
        -- Each key contains a tile the given gaussian overlaps and the depth of that gaussian in the scene.
        -- The sorted indices list contains the indices of each gaussian in the gaussian list.
        let (sorted_gaussian_keys, sorted_gaussian_indices) = unzip sorted_list

        -- define a function to find the color of a pixel
        let f = pixel_color 16 (i64.i32 W) sorted_gaussian_keys sorted_gaussian_indices background preprocessed

        -- tabulate on each pixel using our function
        let pixels = tabulate_2d (i64.i32 H) (i64.i32 W) (\y x -> f x y) :> [i64.i32 image_height][i64.i32 image_width][3]f32
        let pixel_sum = map (map (\ls -> ls[0] + ls[1] + ls[2])) pixels
        let p2 = map (reduce (+) 0) pixel_sum
        let p3 = reduce (+) 0 p2
        in pixels

    -- entry rasterize' [n]
    -- (background: [3]f32)
    -- (means3D: [n][3]f32)
    -- (colors: [n][3]f32)
    -- (opacities: [n][1]f32)
    -- (scales: [n][3]f32)
    -- (rotations: [n][4]f32)
    -- (view_matrix: [4][4]f32)
    -- (proj_matrix: [4][4]f32)
    -- (tan_fovx: f32)
    -- (tan_fovy: f32)
    -- (image_height: i32)
    -- (image_width: i32) =
    -- vjp (\(means3D, colors, opacities, scales, rotations) ->
    --         rasterize background means3D colors opacities scales rotations
    --             view_matrix proj_matrix tan_fovx tan_fovy image_height image_width)
    --     (means3D, colors, opacities, scales, rotations)
    --     1.0f32