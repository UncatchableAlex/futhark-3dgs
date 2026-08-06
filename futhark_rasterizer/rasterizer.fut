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

def TILESIZE = 16i32
def MAX_GAUSSIANS_PER_PIX = 8192i32

type Gaussian2D =
  { opacity: f32
  , color: [3]f32
  , mean_clip: (f32, f32)
  , conic: (f32, f32, f32)
  , depth: f32
  , tiles_touching: i32
  , bounding_box: (i32, i32, i32, i32)
  , radius: i32
  , valid: bool
  }

-- given parameters for an 3D elipse (rotation and scaling), calculate the corresponding
-- correlation matrix for a 3D gaussian. Note that because correlation matrices are symmetrical,
-- we get away with only returning the top half
--
-- equation 6 from kerbl et al.
def compute_cov_3D (scale: [3]f32) (rotation: [4]f32) : [6]f32 =
  let R = quat_to_mat rotation
  let S = scale_to_mat scale
  let M = matmul_3x3 R S
  let sigma = matmul_3x3 M (transpose M)
  in [sigma[0][0], sigma[0][1], sigma[0][2], sigma[1][1], sigma[1][2], sigma[2][2]]

-- calculate the 2D coveriance matrix of a 3D gaussian
-- projected into screen space using equations 29 and 31
-- specified in "EWA Splatting" (Zwicker et al., 2002)
def compute_cov_2D (cmean: (f32, f32, f32))
                   (focal_x: f32)
                   (focal_y: f32)
                   (tan_fovx: f32)
                   (tan_fovy: f32)
                   (cov3D: [6]f32)
                   (viewmatrix: [4][4]f32) : (f32, f32, f32) =
  -- FOV magic from kerbl et al. 2023 (??)
  let limx = 1.3 * tan_fovx
  let limy = 1.3 * tan_fovy
  let t = cmean
  let txtz = t.0 / t.2
  let tytz = t.1 / t.2
  let M0 = t.2 * f32.min limx (f32.max (-limx) txtz)
  let M1 = t.2 * f32.min limy (f32.max (-limy) tytz)
  let t2 = t.2
  -- equation 29 zwicker et al.
  let J =
    [ [focal_x / t2, 0, -(focal_x * M0) / (t2 * t2)]
    , [0, focal_y / t2, -(focal_y * M1) / (t2 * t2)]
    , [0, 0, 0]
    ]
  -- we don't need this row because we will truncate to a 2x2 in a future step

  -- get the rotation part of the view matrix
  let W =
    [ [viewmatrix[0][0], viewmatrix[0][1], viewmatrix[0][2]]
    , [viewmatrix[1][0], viewmatrix[1][1], viewmatrix[1][2]]
    , [viewmatrix[2][0], viewmatrix[2][1], viewmatrix[2][2]]
    ]
  let T = matmul_3x3 J W
  -- set up our cov3D from the upper-triangular representation
  let Vk =
    [ [cov3D[0], cov3D[1], cov3D[2]]
    , [cov3D[1], cov3D[3], cov3D[4]]
    , [cov3D[2], cov3D[4], cov3D[5]]
    ]
  -- equation 31 zwicker et al.
  let cov = matmul_3x3 (matmul_3x3 T Vk) (transpose T)
  -- low pass filter (equation 33 zwicker et al.)
  -- we drop the 3rd row and column and return the top half of the symmetrical 2x2 cov mat
  in (cov[0][0] + 0.3, cov[0][1], cov[1][1] + 0.3)

-- preprocess each gaussian by determining which tiles it touches, the size of its largest radius
-- its conic,
def preprocess -- (P: i32) -- n
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
               --(projmatrix: [4][4]f32)
               --(cam_pos: [3]f32)
               (W: i32)
               (H: i32)
               (focal_x: f32)
               (focal_y: f32)
               (tan_fovx: f32)
               (tan_fovy: f32)
               (opacity: f32)
               (color: [3]f32) : Gaussian2D =
  -- convert world space coordinates to camera space coordinates
  let cmean = transform_point_4x3 mean viewmatrix

  -- get the mean of the gaussian in clip space
  let clip = (
      cmean.0 / (cmean.2 * tan_fovx), 
      cmean.1 / (cmean.2 * tan_fovy)
    )


  -- get the pixel of the gaussian's mean in screen space
  let pix = (ndc_to_pix clip.0 W, ndc_to_pix clip.1 H)

  let depth: f32 = cmean.2
  -- keep depth info separate

  -- determine if this gaussian is in view
  -- let margin_x = 0.15 * f32.i32 W
  -- let margin_y = 0.15 * f32.i32 H
  let in_view =
    depth > 0.2
    -- && pix.0 > -margin_x
    -- && pix.0 < f32.i32 W + margin_x
    -- && pix.1 > -margin_y
    -- && pix.1 < f32.i32 H + margin_y
  let badG: Gaussian2D =
    { opacity = opacity
    , color = color
    , mean_clip = (0, 0)
    , conic = (0, 0, 0)
    , depth = 0
    , tiles_touching = 0
    , bounding_box = (0, 0, 0, 0)
    , radius = 0
    , valid = false
    }
  -- if the gaussian isn't in the view frustum, return a badGaussian so it doesn't contribute to the rasterization
  in if not in_view
     then badG
     else let cov3D = compute_cov_3D scales rotations
          let cov = compute_cov_2D cmean focal_x focal_y tan_fovx tan_fovy cov3D viewmatrix
          let det = cov.0 * cov.2 - cov.1 * cov.1
          -- if the matrix is singular, return with an invalid gaussian so it doesn't contribute to the rasterization
          in if det == 0
             then badG
             else let det_inv = 1 / det
                  -- take the invserse of the 2d covariance matrix
                  -- https://en.wikipedia.org/wiki/Invertible_matrix#Methods_of_matrix_inversion
                  let conic: (f32, f32, f32) = (cov.2 * det_inv, -cov.1 * det_inv, cov.0 * det_inv)
                  -- find the range occupied by the gaussian in screen space. We use
                  -- eigenvalues of the 2d covariance matrix. We want to compute a bounding
                  -- rectangle of the gaussian and find which screen-space tiles our gaussian
                  -- overlaps

                  -- remember: the eigenvalues of gaussians represent the variances along their principle
                  -- axis. Basically how streched they are along their axes (their eigenvectors).
                  -- There's a trick for calculating eigenvalues: https://www.johndcook.com/blog/2021/05/07/trick-for-2x2-eigenvalues/
                  let mid = 0.5 * (cov.0 + cov.2)
                  let inner = mid * mid - det
                  let lambda1 = mid + f32.sqrt (f32.max 0.1 inner)
                  let lambda2 = mid - f32.sqrt (f32.max 0.1 inner)
                  let larger_principle_axis = f32.max lambda1 lambda2
                  -- get the radius of 3 standard deviations using the larger principal axis
                  let std3 = f32.ceil <| 3 * (f32.sqrt larger_principle_axis)
                  -- calculate a bounding box for the support of our gaussian in pixel space. Define the box with 3 std deviations from the mean
                  -- in the most stretched direction (either lambda1 or lambda2)
                  let (ylo, yhi, xlo, xhi): (i32, i32, i32, i32) = get_rect_2d pix std3 W H TILESIZE
                  let num_tiles: i32 = (yhi - ylo) * (xhi - xlo)
                  in if num_tiles == 0
                     then badG
                     else { opacity = opacity
                          , color = color
                          , mean_clip = clip
                          , conic = conic
                          , depth = depth
                          , tiles_touching = num_tiles
                          , bounding_box = (ylo, yhi, xlo, xhi)
                          , radius = i32.f32 std3
                          , valid = true
                          }

-- make one key per tile that this gaussian overlaps. Each key should define that tile's instance of this gaussian
-- in a total ordering based on tile # and depth. The key should also be able to identify this gaussian. We will
-- accomplish this by making the key a u64 where the lower 32 bits are the depth and the upper 32 bits are the tile
def generateElem [n]
                 (gs: [n]Gaussian2D)
                 (prefix_sum: [n]i32)
                 (W: i32)
                 (idx: i64) : (u64, i32) =
  -- search to find which gaussian this index refers to
  let bs = binary_search prefix_sum (i32.i64 idx) u64.i32
  --binary_search
  let n_idx = if prefix_sum[bs] == i32.i64 idx then bs else i32.max 0 (bs - 1)
  -- this is the gaussian idx refers to

  -- extract data for this gaussian
  let (ylo, _, xlo, xhi): (i32, i32, i32, i32) = gs[n_idx].bounding_box
  let depth = gs[n_idx].depth
  -- extract the tile that this gaussian/idx refers to
  -- this gaussian spans many tiles. Which of those tiles is this idx?
  let offset = (i32.i64 idx) - prefix_sum[n_idx]
  let tile_y = ylo + (offset / (xhi - xlo))
  let tile_x = xlo + (offset % (xhi - xlo))
  let tile = tile_y * ((W + TILESIZE - 1) / TILESIZE) + tile_x
  let upper = ((u64.i32 tile) u64.<< 32)
  let key = upper | (u64.u32 <| f32.to_bits <| f32.f32 depth)
  in (key, n_idx)

-- Calculate the color of each pixel using an upper limit on the
-- number of gaussians per pixel
def pixel_color_train [n] [m]
                      (W: i64)
                      (H: i64)
                      (sorted_keys: [m]u64)
                      (sorted_indices: [m]i32)
                      (bg: [3]f32)
                      (gs: [n]Gaussian2D)
                      (pix_x: i64)
                      (pix_y: i64) : [3]f32 =
  -- our pixel's tile
  let ts = i64.i32 TILESIZE
  let tile_y = pix_y / ts
  let tile_x = pix_x / ts
  let tile = tile_y * ((W + ts - 1) / ts) + tile_x
  -- the index of the first gaussian in the sorted list in the given tile
  let start = binary_search sorted_keys ((u64.i64 tile) << 32) id
  -- the index of the last gaussian in the sorted list in the given tile
  let end = binary_search sorted_keys ((u64.i64 (tile + 1)) << 32) id
  -- This whole for-loop is just equation 2 from the 3dgs paper (Kerbl et al. 2023)
  let ((pr, pg, pb), bg_T) =
    --#[stripmine(128)]
    loop ((r, g, b), T) = ((0.0f32, 0.0f32, 0.0f32), 1.0f32)
    for i < MAX_GAUSSIANS_PER_PIX do
      let j = start + i
      -- if our loop goes out of bounds, stop calculating
      in if j >= end || T < 0.0001
         then ((r, g, b), T)
         else -- otherwise, keep calculating colors based on the intersecting gaussians
              let mean_clip = gs[sorted_indices[j]].mean_clip
              let opacity = gs[sorted_indices[j]].opacity
              let g_color = gs[sorted_indices[j]].color
              let conic = gs[sorted_indices[j]].conic
              let mean_pix = (ndc_to_pix mean_clip.0 (i32.i64 W), ndc_to_pix mean_clip.1 (i32.i64 H))
              -- the distance of this pixel from the ith gaussian's mean
              let dx = f32.i64 pix_x - mean_pix.0
              let dy = f32.i64 pix_y - mean_pix.1
              -- gaussian equation. See eq 19 from Zwicker et al. 2001
              let power = -0.5 * (conic.0 * dx * dx + 2 * conic.1 * dx * dy + conic.2 * dy * dy)
              in if power > 0
                 then -- numerical degenerate case something went wrong upstream, probably a near singular covariance matrix
                      ((r, g, b), T)
                 else let alpha = f32.min 0.99 (opacity * f32.exp power)
                      -- We don't include gaussians with tiny alphas. This is to reduce numerical instability
                      -- when differentiating. See 3DGS Kerbl et al. 2023 appendix
                      in if alpha < (1f32 / 255f32)
                         then ((r, g, b), T)
                         else let T' = T * (1 - alpha)
                              -- calculate new transmittance

                              -- accumulate alpha from this gaussian into each color band
                              let color' =
                                ( g_color[0] * alpha * T + r
                                , g_color[1] * alpha * T + g
                                , g_color[2] * alpha * T + b
                                )
                              in if T' < 0.0001 then ((r, g, b), T') else (color', T')
  -- add the color contribution of the background
  in [ f32.min 1 (bg_T * bg[0] + pr)
     , f32.min 1 (bg_T * bg[1] + pg)
     , f32.min 1 (bg_T * bg[2] + pb)
     ]

-- Calculate the color of each pixel using as many iterations as necessary per pixel
def pixel_color_test [n] [m]
                     (W: i64)
                     (H: i64)
                     (sorted_keys: [m]u64)
                     (sorted_indices: [m]i32)
                     (bg: [3]f32)
                     (gs: [n]Gaussian2D)
                     (pix_x: i64)
                     (pix_y: i64) : [3]f32 =
  --(f32, f32, f32)  = -- output: rgb
  -- our pixel's tile
  let ts = i64.i32 TILESIZE
  let tile_y = pix_y / ts
  let tile_x = pix_x / ts
  let tile = tile_y * ((W + ts - 1) / ts) + tile_x
  -- the index of the first gaussian in the sorted list in the given tile
  let start = binary_search sorted_keys ((u64.i64 tile) << 32) id
  -- the index of the last gaussian in the sorted list in the given tile
  let end = binary_search sorted_keys ((u64.i64 (tile + 1)) << 32) id
  -- This whole for-loop is just equation 2 from the 3dgs paper (Kerbl et al. 2023)
  let ((pr, pg, pb), bg_T, _) =
    loop ((r, g, b), T, i) = ((0.0f32, 0.0, 0.0), 1.0, start)
    while T > 0.0001 && i < end do
      let mean_clip = gs[sorted_indices[i]].mean_clip
      -- let mean = (ndc_to_pix mean_clip.0 (i32.i64 W), ndc_to_pix mean_clip.1 (i32.i64 H))
      let opacity = gs[sorted_indices[i]].opacity
      let g_color = gs[sorted_indices[i]].color
      let conic = gs[sorted_indices[i]].conic
      let mean_pix = (ndc_to_pix mean_clip.0 (i32.i64 W), ndc_to_pix mean_clip.1 (i32.i64 H))

      -- the distance of this pixel from the ith gaussian's mean
      let dx = f32.i64 pix_x - mean_pix.0
      let dy = f32.i64 pix_y - mean_pix.1
      -- gaussian equation. See eq 19 from Zwicker et al. 2001
      let power = -0.5 * (conic.0 * dx * dx + 2 * conic.1 * dx * dy + conic.2 * dy * dy)
      in if power > 0
         then ((r, g, b), T, i + 1)
         else let alpha = f32.min 0.99 (opacity * f32.exp power)
              -- We don't include gaussians with tiny alphas. This is to reduce numerical instability
              -- when differentiating. See 3DGS Kerbl et al. 2023 appendix
              in if alpha < 1f32 / 255f32
                 then ((r, g, b), T, i + 1)
                 else let T' = T * (1 - alpha)
                      -- calculate new transmittance

                      -- accumulate alpha from this gaussian into each color band
                      let color' =
                        ( g_color[0] * alpha * T + r
                        , g_color[1] * alpha * T + g
                        , g_color[2] * alpha * T + b
                        )
                      in if T' < 0.0001 then ((r, g, b), T', i + 1) else (color', T', i + 1)
  -- add the color contribution of the background
  in [ f32.min 1 (bg_T * bg[0] + pr)
     , f32.min 1 (bg_T * bg[1] + pg)
     , f32.min 1 (bg_T * bg[2] + pb)
     ]

def rasterize2dGaussians [n]
                         (g2ds: [n]Gaussian2D)
                         (background: [3]f32)
                         (image_height: i64)
                         (image_width: i64)
                         (train: bool) : ([n]i32, [image_height][image_width][3]f32) =
  let H = i32.i64 image_height
  let W = i32.i64 image_width
  let radii = map (\g -> g.radius) g2ds
  let g2ds_culled = filter (\g -> g.valid) g2ds
  let num_tiles = map (\g -> g.tiles_touching) g2ds_culled
  -- get the prefix sum of the number of tiles each gaussian touches
  let prefix_sum = rotate (-1) <| scan (+) 0 num_tiles
  let tiles_touched = prefix_sum[0]
  let prefix_sum[0] = 0
  -- fix the first element of the prefix sum since rotate will have messed it up

  let sorted_list =
    blocked_radix_sort_by_key 256i16
                              (\(k, _) -> k)                           -- block size for a100 gpu
                              64                                       -- extract the key
                              (\i k -> i32.u64 (k >> (u64.i32 i) & 1)) -- sort on 64 bit keys
                              (map (generateElem g2ds_culled prefix_sum W) (iota <| i64.i32 tiles_touched)) -- get the ith bit
  -- the kv pairs we are sorting where the key is (tile, depth) as a u64

  -- the sorted gaussian keys are the keys of each copy of each gaussian in the big list.
  -- Each key contains a tile the given gaussian overlaps and the depth of that gaussian in the scene.
  -- The sorted indices list contains the indices of each gaussian in the gaussian list.
  let (sorted_gaussian_keys, sorted_gaussian_indices) = unzip sorted_list
  -- define a function to find the color of a pixel
  let f_test = pixel_color_test image_width image_height sorted_gaussian_keys sorted_gaussian_indices background g2ds_culled
  let f_train = pixel_color_train image_width image_height sorted_gaussian_keys sorted_gaussian_indices background g2ds_culled
  -- tabulate on each pixel using our function
  let pixels =
    if train
    then tabulate_2d (i64.i32 H) (i64.i32 W) (\y x -> f_train x y) :> [image_height][image_width][3]f32
    else tabulate_2d (i64.i32 H) (i64.i32 W) (\y x -> f_test x y) :> [image_height][image_width][3]f32
  in (radii, pixels)

def compute2dGaussians [n]
                       (means3D: [n][3]f32)
                       (colors: [n][3]f32)
                       (opacities: [n][1]f32)
                       (scales: [n][3]f32)
                       (rotations: [n][4]f32)
                       (view_matrix: [4][4]f32)
                       (tan_fovx: f32)
                       (tan_fovy: f32)
                       (H: i32)
                       (W: i32) : [n]Gaussian2D =
  let focalx = (f32.i32 W) / (2 * tan_fovx)
  let focaly = (f32.i32 H) / (2 * tan_fovy)
  let view_matrix' = transpose view_matrix
  -- preprocess each gaussian in parallel. This generates our list of Gaussian2D records.
  in map5 (\mean scale rotation opacity color ->
             preprocess mean scale rotation view_matrix' W H focalx focaly tan_fovx tan_fovy opacity[0] color)
          means3D
          scales
          rotations
          opacities
          colors

-- Our exposed rasterize function. In this function, we take the means, scales,
-- rotations, colors, and opacities of the 3D Gaussians, as well as the camera
-- parameters, and output a rasterized image.
entry rasterize [n]
                (background: [3]f32) -- the color of the background
                (means3D: [n][3]f32)
                (colors: [n][3]f32)  -- must be rgb
                (opacities: [n][1]f32)
                (scales: [n][3]f32)  -- scale vectors
                (rotations: [n][4]f32) 
                -- (scale_modifier: f32) -- unused
                --(cov3D_precomp: [][3][3]f32) -- unused
                (view_matrix: [4][4]f32) -- encodes rotation and translation
                -- (proj_matrix: [4][4]f32)  -- unused
                (tan_fovx: f32) -- encodes focal (x) length
                (tan_fovy: f32) -- encodes focal (y) length
                (image_height: i64)  -- in pixels
                (image_width: i64) : -- in pixels
              --(sh: [][3][]f32) -- unused
              --(degree: i32) -- unused
              --(campos: [3]f32) -- unused
              --(prefiltered: bool) -- unused
              --(debug: bool) --unused
([n]i32, [image_height][image_width][3]f32) =
  let H = i32.i64 image_height
  let W = i32.i64 image_width
  -- An array of 2D Gaussian records derived from the scene parameters
  let gaussians =
    compute2dGaussians means3D
                       colors
                       opacities
                       scales
                       rotations
                       view_matrix
                       tan_fovx
                       tan_fovy
                       H
                       W
  -- rasterize the 2D Gaussians into a pixel array
  in rasterize2dGaussians gaussians
                          background
                          image_height
                          image_width
                          false

-- https://en.wikipedia.org/wiki/Structural_similarity_index_measure#Algorithm
def ssim3 [n] [m] [o]
          (A: [n][n]f32)
          (X: [m][o][3]f32)
          (Y: [m][o][3]f32)
          (c1: f32)
          (c2: f32) : f32 =
  let sum =
    reduce (+) 0
    <| flatten_3d
       <| tabulate_3d m o 3 (\x y i ->
                               let x' = x - (n / 2)
                               let y' = y - (n / 2)
                               let nn = n * n
                               let (muX, muY, muXY, muX2, muY2) =
                                 loop (muX, muY, muXY, muX2, muY2) = (0, 0, 0, 0, 0)
                                 for j < nn do
                                   let a = x' + (j / n)
                                   let b = y' + (j % n)
                                   let k = A[j / n][j % n]
                                   let inbounds = a >= 0 && b >= 0 && a < m && b < o
                                   in (if inbounds
                                       then ( k * X[a][b][i] + muX
                                            , k * Y[a][b][i] + muY
                                            , k * X[a][b][i] * Y[a][b][i] + muXY
                                            , k * X[a][b][i] * X[a][b][i] + muX2
                                            , k * Y[a][b][i] * Y[a][b][i] + muY2
                                            )
                                       else (muX, muY, muXY, muX2, muY2))
                               let muXX = muX * muX
                               let muYY = muY * muY
                               let sigX = muX2 - muXX
                               let sigY = muY2 - muYY
                               let sigXY = muXY - (muX * muY)
                               in (2 * muX * muY + c1) * (2 * sigXY + c2) / ((muXX + muYY + c1) * (sigX + sigY + c2)))
  in sum / (f32.i64 <| m * o * 3)

-- a normalized 2d gaussian distribution: https://en.wikipedia.org/wiki/Gaussian_function
-- we can multiply 2 1d gaussians together to get the 2d version.
def gdist2d (n: i64) (sigma: f32) =
  let g1d =
    map (\i ->
           let num = (i - (n / 2)) ** 2
           let denom = 2 * (sigma ** 2)
           in f32.exp (-f32.i64 num / denom))
        (iota n)
  let sum = foldl (+) 0 g1d
  in tabulate_2d n n (\x y -> g1d[x] * g1d[y] / (sum * sum))

def loss (image_height: i64)
         (image_width: i64)
         (ssim_kernel_size: i32)
         (ssim_kernel_sigma: f32)
         (pix: [image_height][image_width][3]f32)
         (gt_image: [image_height][image_width][3]f32)
         (lambda: f32) : -- percentage of our loss that is ssim (the rest is L1)
f32 =
  -- calculate the ssim3 loss
  let kernel = gdist2d (i64.i32 ssim_kernel_size) ssim_kernel_sigma
  let c1 = 0.01 ** 2
  let c2 = 0.03 ** 2
  let ssim = ssim3 kernel pix gt_image c1 c2

  -- calculate the L1 loss
  let l1abs = map2 (\a b -> f32.abs <| a - b) (flatten_3d pix) (flatten_3d gt_image)
  let l1 = (reduce (+) 0 l1abs) / (f32.i64 <| image_height * image_width * 3)
  
  -- The formula given on Wikipedia has says dssim = (1-ssim)/2, but they do it like this
  -- in kerbl et al's code, so we copy them.
  let dssim = 1 - ssim
  -- Eq 7 Kerbl et al. calls for a linear combination of L1 and DSSIM losses
  in ((1 - lambda)) * l1 + (lambda * dssim)

entry grad [n]
           (background: [3]f32)
           (means3D: [n][3]f32)
           (colors: [n][3]f32)
           (opacities: [n][1]f32)
           (scales: [n][3]f32)
           (rotations: [n][4]f32)
           (view_matrix: [4][4]f32)
           (tan_fovx: f32)
           (tan_fovy: f32)
           (image_height: i64)
           (image_width: i64)
           (ssim_kernel_size: i32)
           (ssim_kernel_sigma: f32)
           (gt_image: [image_height][image_width][3]f32)
           (lambda: f32) : -- percentage of our loss that is dssim (the rest is L1)
([n][3]f32, [n][3]f32, [n][3]f32, [n][1]f32, [n][3]f32, [n][4]f32, [image_height][image_width][3]f32, [n]i32, f32) =
  let H = i32.i64 image_height
  let W = i32.i64 image_width
  -- calculate the 2D means of the gaussians and dm_2D/dm_3D
  let project_means_3d_to_2d =
    map (\mean ->
           let cmean = transform_point_4x3 mean (transpose view_matrix)
           in (cmean.0 / (cmean.2 * tan_fovx), cmean.1 / (cmean.2 * tan_fovy)))
  -- pix y

  -- get the first row of the dm_2/dm_3 jacobian
  let (means2D, dmeans2D_means3D_1s) = vjp2 project_means_3d_to_2d means3D (rep (1, 0))
  -- get the second row of the dm_2/dm_3 jacobian
  let dmeans2D_means3D_2s = vjp project_means_3d_to_2d means3D (rep (0, 1))
  -- compose jacobian of form
  -- [
  --  [[d2D_g1_1/d3D_g1_1,  d2D_g1_1/d3D_g1_2, d2D_g1_1/d3D_g1_3],
  --  [d2D_g1_2/d3D_g1_1,  d2D_g1_2/d3D_g1_2, d2D_g1_2/d3D_g1_3]],
  --  ...
  --  [[d2D_gn_1/d3D_gn_1, d2D_gn_1/d3D_gn_2, d2D_gn_1/d3D_gn_3]]
  --  [d2D_gn_2/d3D_gn_1, d2D_gn_2/d3D_gn_2, d2D_gn_2/d3D_gn_3]]
  --
  -- where d2D_gi_j/d3D_gk_l is the derivative of jth coordinate the 2D screen-space mean of the ith gaussian w.r.t. the kth coordiante of the 3D world-space mean of the lth gaussian
  let dmeans2D_means3D = transpose ([dmeans2D_means3D_1s, dmeans2D_means3D_2s])
  -- a function that takes 3d means and returns conics
  let to_conic =
    \(means3D, scales, rotations) ->
      (let gaussians = compute2dGaussians means3D colors opacities scales rotations view_matrix tan_fovx tan_fovy H W
       in map (\(g: Gaussian2D) -> g.conic) gaussians)
  -- get the dconic/dm_3 jacobian
  let (conic_2d, dconic_mean3D_1s) = jvp2 to_conic (means3D, scales, rotations) (rep [1, 0, 0], rep [0, 0, 0], rep [0, 0, 0, 0])
  -- 1st column
  let dconic_mean3D_2s = jvp to_conic (means3D, scales, rotations) (rep [0, 1, 0], rep [0, 0, 0], rep [0, 0, 0, 0])
  -- 2nd column
  let dconic_mean3D_3s = jvp to_conic (means3D, scales, rotations) (rep [0, 0, 1], rep [0, 0, 0], rep [0, 0, 0, 0])
  -- 3rd column
  let dconic_mean3D = transpose ([dconic_mean3D_1s, dconic_mean3D_2s, dconic_mean3D_3s])
  -- get the dconic/ds jacobian
  let dconics_scales =
    transpose ([ jvp to_conic (means3D, scales, rotations) (rep [0, 0, 0], rep [1, 0, 0], rep [0, 0, 0, 0])
               , -- 1st column
                 jvp to_conic (means3D, scales, rotations) (rep [0, 0, 0], rep [0, 1, 0], rep [0, 0, 0, 0])
               , -- 2nd column
                 jvp to_conic (means3D, scales, rotations) (rep [0, 0, 0], rep [0, 0, 1], rep [0, 0, 0, 0])
               ])
  -- 3rd column

  -- get the dconic/dR jacobian
  let dconics_rotations =
    transpose ([ jvp to_conic (means3D, scales, rotations) (rep [0, 0, 0], rep [0, 0, 0], rep [1, 0, 0, 0])
               , -- 1st column
                 jvp to_conic (means3D, scales, rotations) (rep [0, 0, 0], rep [0, 0, 0], rep [0, 1, 0, 0])
               , -- 2nd column
                 jvp to_conic (means3D, scales, rotations) (rep [0, 0, 0], rep [0, 0, 0], rep [0, 0, 1, 0])
               , -- 3rd column
                 jvp to_conic (means3D, scales, rotations) (rep [0, 0, 0], rep [0, 0, 0], rep [0, 0, 0, 1])
               ])
  -- 4th column

  -- define a forward pass to get the loss
  let forward =
    \(means_clip, conics, colors, opacities, scales, rotations) ->
      (-- calculate the 2d gaussians (algorithm 2 in my paper)
       let gaussians =
         compute2dGaussians means3D
                            colors
                            opacities
                            scales
                            rotations
                            view_matrix
                            tan_fovx
                            tan_fovy
                            H
                            W
       -- replace the screen space means and conic values we just calculate with 
       -- the ones we were given to ensure correct propagation of gradients with vjp       
       let gaussians' = map3 (\(g: Gaussian2D) m c -> (g with mean_clip = m) with conic = c) gaussians means_clip conics
       -- rasterize the gaussians (algorithm 1 in my paper)
       let (radii, pix) = rasterize2dGaussians gaussians' background image_height image_width true
       -- calculate the loss and return
       let l = loss image_height image_width ssim_kernel_size ssim_kernel_sigma pix gt_image lambda
       in (l, radii, pix))
  -- inputs to our forward pass
  let inps = (means2D, conic_2d, colors, opacities, scales, rotations)
  -- perform a full forward and backward pass on our loss calculation
  let ((loss', radii, pix), (dL_means2D, dL_conics, dL_colors, dL_opacities, _, _)) = vjp2 forward inps (1.0, rep 0, rep (rep [0, 0, 0]))
  -- use the chain rule to calculate dL/dm_3 = (dL/dm_2 x dm2/dm3) + (dL/dC x dC/dm3)
  let dL_means3D =
    map (\(i: i64) ->
           [ dmeans2D_means3D[i][0][0] * dL_means2D[i].0 + dmeans2D_means3D[i][1][0] * dL_means2D[i].1 + dconic_mean3D[i][0].0 * dL_conics[i].0 + dconic_mean3D[i][0].1 * dL_conics[i].1 + dconic_mean3D[i][0].2 * dL_conics[i].2
           , dmeans2D_means3D[i][0][1] * dL_means2D[i].0 + dmeans2D_means3D[i][1][1] * dL_means2D[i].1 + dconic_mean3D[i][1].0 * dL_conics[i].0 + dconic_mean3D[i][1].1 * dL_conics[i].1 + dconic_mean3D[i][1].2 * dL_conics[i].2
           , dmeans2D_means3D[i][0][2] * dL_means2D[i].0 + dmeans2D_means3D[i][1][2] * dL_means2D[i].1 + dconic_mean3D[i][2].0 * dL_conics[i].0 + dconic_mean3D[i][2].1 * dL_conics[i].1 + dconic_mean3D[i][2].2 * dL_conics[i].2
           ])
        (iota n)
  -- listify dL_means2D for backward compatibility
  let dL_means2D' = map (\m -> [m.0, m.1, 0]) dL_means2D
  -- calculate dL/dS = dL/dC x dC/dS
  let dscales = map2 transform_tup_3x3 dL_conics dconics_scales
  -- map2 matmul_3x3

  -- calculate dL/dS = dL/dC x dC/dR
  let drotations = map2 transform_tup_4x3 dL_conics dconics_rotations
  in (dL_means3D, dL_means2D', dL_colors, dL_opacities, dscales, drotations, pix, radii, loss')

-- a version of grad that does two full passes over rasterize2DGaussians.
-- Once for dL_means2D and again for everything else
entry grad_naive [n]
                 (background: [3]f32)
                 (means3D: [n][3]f32)
                 (colors: [n][3]f32)
                 (opacities: [n][1]f32)
                 (scales: [n][3]f32)
                 (rotations: [n][4]f32)
                 (view_matrix: [4][4]f32)
                 (tan_fovx: f32)
                 (tan_fovy: f32)
                 (image_height: i64)
                 (image_width: i64)
                 (ssim_kernel_size: i32)
                 (ssim_kernel_sigma: f32)
                 (gt_image: [image_height][image_width][3]f32)
                 (lambda: f32) : -- percentage of our loss that is dssim (the rest is L1)
( [n][3]f32
, [n][3]f32
, [n][3]f32
, [n][1]f32
, [n][3]f32
, [n][4]f32
, [image_height][image_width][3]f32
, [n]i32
, f32
) =
  -- define a unified function that calculates the loss from the
  -- scene parameters
  let forward_naive =
    \(means3D, colors, opacities, scales, rotations) ->
      (-- calulate the radii and pixels
       let (radii, pix) =
         rasterize background
                   means3D
                   colors
                   opacities
                   scales
                   rotations
                   view_matrix
                   tan_fovx
                   tan_fovy
                   image_height
                   image_width
       -- compute the loss
       let l =
         loss image_height
              image_width
              ssim_kernel_size
              ssim_kernel_sigma
              pix
              gt_image
              lambda
       -- return everything
       in (l, radii, pix))
  -- inputs to our forward pass
  let inps = (means3D, colors, opacities, scales, rotations)
  -- perform a full forward and backward pass on our loss calculation
  let ((loss', radii, pix), (dL_means3D, dL_colors, dL_opacities, dscales, drotations)) = vjp2 forward_naive inps (1.0, rep 0, rep (rep [0, 0, 0]))
  let H = i32.i64 image_height
  let W = i32.i64 image_width
  -- define the same forward pass, but
  let forward_naive2 =
    \means2D ->
      (-- calculate the gaussians
       let gaussians =
         compute2dGaussians means3D
                            colors
                            opacities
                            scales
                            rotations
                            view_matrix
                            tan_fovx
                            tan_fovy
                            H
                            W
      
       -- propagate the means2D_clip and conic paramaters we were given
       let gaussians' = map2 (\(g: Gaussian2D) m -> g with mean_clip = m) gaussians means2D
       -- rasterize the gaussians
       let (_, pix) = rasterize2dGaussians gaussians' background image_height image_width false
       -- calculate the loss and return
       in loss image_height image_width ssim_kernel_size ssim_kernel_sigma pix gt_image lambda)
  let means_clip =
    map (\mean ->
           let cmean = transform_point_4x3 mean (transpose view_matrix)
           in (cmean.0 / (cmean.2 * tan_fovx), cmean.1 / (cmean.2 * tan_fovy)))
        -- pix y

        means3D
  let dL_means2D = vjp forward_naive2 means_clip 1.0
  -- now get dmeans2d
  let dL_means2D' = map (\(x, y) -> [x, y, 0]) dL_means2D
  in (dL_means3D, dL_means2D', dL_colors, dL_opacities, dscales, drotations, pix, radii, loss')
