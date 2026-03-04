-- Transform Point 4x3
-- ==
-- entry: transform_point_4x3
-- input {[1.0, 2.0, 3.0] [[0.0, -1.0, 0.0, 0.0],[1.0,  0.0, 0.0, 0.0],[0.0, 0.0, 1.0, 0.0],[10.0, 20.0, 30.0, 1.0]]}
-- output {[8.0,21.0,33.0]}

entry dot [n] (x: [n]f32) (y: [n]f32): f32 = 
    reduce (+) 0 (map2 (*) x y)

-- adapted from https://www.futhark-lang.org/examples/matrix-multiplication.html
def matmul_f32 [n][m][p] (A: [n][m]f32) (B: [m][p]f32) : [n][p]f32 =
  map (\A_row ->
         map (\B_col ->
                reduce (+) 0 (map2 (*) A_row B_col))
             (transpose B))
      A

entry transform_point_4x3 (p: [3]f32) (T: [4][4]f32) : [3]f32 = 
    [
        T[0][0]*p[0] + T[0][1]*p[1] + T[0][2]*p[2] + T[0][3],
        T[1][0]*p[0] + T[1][1]*p[1] + T[1][2]*p[2] + T[1][3],
        T[2][0]*p[0] + T[2][1]*p[1] + T[2][2]*p[2] + T[2][3]
    ]

entry transform_point_4x4  (p: [3]f32) (T: [4][4]f32) : [4]f32 = 
    [
        T[0][0]*p[0] + T[0][1]*p[1] + T[0][2]*p[2] + T[0][3],
        T[1][0]*p[0] + T[1][1]*p[1] + T[1][2]*p[2] + T[1][3],
        T[2][0]*p[0] + T[2][1]*p[1] + T[2][2]*p[2] + T[2][3],
        T[3][0]*p[0] + T[3][1]*p[1] + T[3][2]*p[2] + T[3][3]
    ]

entry matmul_3x3 (A: [3][3]f32) (B: [3][3]f32) : [3][3]f32 =
    [
        [A[0][0]*B[0][0] + A[0][1]*B[1][0] + A[0][2]*B[2][0],
         A[0][0]*B[0][1] + A[0][1]*B[1][1] + A[0][2]*B[2][1],
         A[0][0]*B[0][2] + A[0][1]*B[1][2] + A[0][2]*B[2][2]],

        [A[1][0]*B[0][0] + A[1][1]*B[1][0] + A[1][2]*B[2][0],
         A[1][0]*B[0][1] + A[1][1]*B[1][1] + A[1][2]*B[2][1],
         A[1][0]*B[0][2] + A[1][1]*B[1][2] + A[1][2]*B[2][2]],

        [A[2][0]*B[0][0] + A[2][1]*B[1][0] + A[2][2]*B[2][0],
         A[2][0]*B[0][1] + A[2][1]*B[1][1] + A[2][2]*B[2][1],
         A[2][0]*B[0][2] + A[2][1]*B[1][2] + A[2][2]*B[2][2]]
    ]

-- Transform a **UNIT** quaternion to a rotation matrix 
-- https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation#Using_quaternions_as_rotations
-- TODO: Figure out where we enforce that quaternions are unit
entry quat_to_mat (q: [4]f32) : [3][3]f32= 
    let qr = q[0]
    let qi = q[1]
    let qj = q[2]
    let qk = q[3]
    in [
        [1 - 2*(qj*qj + qk*qk),  2*(qi*qj - qk*qr),      2*(qi*qk + qj*qr)],
        [2*(qi*qj + qk*qr),      1 - 2*(qi*qi + qk*qk),  2*(qj*qk - qi*qr)],
        [2*(qi*qk - qj*qr),      2*(qj*qk + qi*qr),      1 - 2*(qi*qi + qj*qj)]
    ]

-- convert a scale vector to a scale matrix
entry scale_to_mat (s: [3]f32) : [3][3]f32 = 
    [
        [s[0],0,0],
        [0,s[1],0],
        [0,0,s[2]]
    ]

-- convert a coordinate from normalized device coordinates (-1 to 1)
-- to a pixel value
entry ndc_to_pix (v: f32) (s: i32) : f32 = 
    ((v + 1.0) * (f32.i32 s) - 1) * 0.5


entry get_rect_2d (center: [2]f32) (radius: f32) (W: i32) (H: i32) (tilesize: i32): (i32,i32,i32,i32) =
    let xlo = f32.max 0 (f32.floor (center[1] - radius))
    let xhi = f32.min (f32.i32 (W-1)) (f32.ceil  (center[1] + radius))
    let ylo = f32.max 0 (f32.floor (center[0] - radius))
    let yhi = f32.min (f32.i32 (H-1)) (f32.ceil  (center[0] + radius))
    in (
        (i32.f32 ylo)/tilesize,
        (i32.f32 yhi)/tilesize,
        (i32.f32 xlo)/tilesize,
        (i32.f32 xhi)/tilesize)


-- https://www.futhark-lang.org/examples/radix-sort.html
def radix_sort_step [n] (xs: [n]u64) (b: i32): [n]u64 =
  let bits = map (\x -> (i32.u64 (x >> u64.i32 b)) & 1) xs
  let bits_neg = map (1-) bits
  let offs = reduce (+) 0 bits_neg
  let idxs0 = map2 (*) bits_neg (scan (+) 0 bits_neg)
  let idxs1 = map2 (*) bits (map (+offs) (scan (+) 0 bits))
  let idxs2 = map2 (+) idxs0 idxs1
  let idxs  = map (\x->x-1) idxs2
  let xs' = scatter (copy xs) (map i64.i32 idxs) xs
  in xs'

def radix_sort [n] (xs: [n]u64): [n]u64 =
  loop xs for i < 32 do radix_sort_step xs i
        