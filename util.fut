-- Transform Point 4x3
-- ==
-- entry: transform_point_4x3
-- input {[1.0f32, 2.0, 3.0] [[0.0f32, -1.0, 0.0, 0.0],[1.0f32,  0.0, 0.0, 0.0],[0.0f32, 0.0, 1.0, 0.0],[10.0f32, 20.0, 30.0, 1.0]]}
-- output {[8.0f32,21.0f32,33.0f32]}
--
--
-- Binary Search
-- ==
-- entry: binary_seach
-- input {[1,3,5,8,10,15,20,40,44], 39}
-- output {6}

def dot [n] (x: [n]f32) (y: [n]f32): f32 = 
    reduce (+) 0 (map2 (*) x y)

-- adapted from https://www.futhark-lang.org/examples/matrix-multiplication.html
def matmul_f32 [n][m][p] (A: [n][m]f32) (B: [m][p]f32) : [n][p]f32 =
  map (\A_row ->
         map (\B_col ->
                reduce (+) 0 (map2 (*) A_row B_col))
             (transpose B))
      A

def transform_point_4x3 (p: [3]f32) (T: [4][4]f32) : [3]f32 = 
    [
        T[0][0]*p[0] + T[0][1]*p[1] + T[0][2]*p[2] + T[0][3],
        T[1][0]*p[0] + T[1][1]*p[1] + T[1][2]*p[2] + T[1][3],
        T[2][0]*p[0] + T[2][1]*p[1] + T[2][2]*p[2] + T[2][3]
    ]

def transform_point_4x4  (p: [3]f32) (T: [4][4]f32) : [4]f32 = 
    [
        T[0][0]*p[0] + T[0][1]*p[1] + T[0][2]*p[2] + T[0][3],
        T[1][0]*p[0] + T[1][1]*p[1] + T[1][2]*p[2] + T[1][3],
        T[2][0]*p[0] + T[2][1]*p[1] + T[2][2]*p[2] + T[2][3],
        T[3][0]*p[0] + T[3][1]*p[1] + T[3][2]*p[2] + T[3][3]
    ]

def matmul_3x3 (A: [3][3]f32) (B: [3][3]f32) : [3][3]f32 =
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
def quat_to_mat (q: [4]f32) : [3][3]f32= 
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
def scale_to_mat (s: [3]f32) : [3][3]f32 = 
    [
        [s[0],0,0],
        [0,s[1],0],
        [0,0,s[2]]
    ]

-- convert a coordinate from normalized device coordinates (-1 to 1)
-- to a pixel value
def ndc_to_pix (v: f32) (s: i32) : f32 = 
    ((v + 1.0) * (f32.i32 s) - 1) * 0.5


def get_rect_2d (x:f32, y:f32) (radius: f32) (W: i32) (H: i32) (tilesize: i32): (i32,i32,i32,i32) =
    let tiles_x = W / tilesize
    let tiles_y = H / tilesize
    let xlo = i32.min tiles_x (i32.max 0 ((i32.f32 (x - radius))/tilesize))
    let xhi = i32.min tiles_x (i32.max 0 ((i32.f32 (x + radius + f32.i32 tilesize - 1))/tilesize))
    let ylo = i32.min tiles_y (i32.max 0 ((i32.f32 (y - radius))/tilesize))
    let yhi = i32.min tiles_y (i32.max 0 ((i32.f32 (y + radius + f32.i32 tilesize - 1))/tilesize))
    in (ylo, yhi, xlo, xhi)


-- used psuedocode from https://en.wikipedia.org/wiki/Binary_search
def binary_search [n] 't (ls: [n]t) (idx: t) (get_value: t -> u64): i32 =
  let (lo, hi) = (0i32, (i32.i64 n - 1i32))
  let (result, _) = loop (lo, hi) while lo < hi do
    let mid = lo + (hi - lo) / 2
    in if get_value ls[mid] < get_value idx
       then (mid + 1, hi)
       else (lo, mid)
  in result
        