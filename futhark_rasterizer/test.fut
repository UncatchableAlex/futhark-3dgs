-- toy projection function
def f [n] (x:[n][3]f32) : [n](f32, f32) = map (\x' -> (x'[0]/x'[2], x'[1]/(x'[2]**x'[2]))) x

-- toy loss function
def g [n] (y:[n](f32,f32)) : f32 = reduce (+) 0 <| map (\y' -> (10 - y'.0) + (10 - y'.1)) y

-- toy gradient function
def grad [n] (x: [n][3]f32): ([n][3]f32, [n][3]f32) = 
    let (y, df_dx_1) = vjp2 f x (rep (1,0))
    let df_dx_2 = vjp f x (rep (0,1))
    let dg_df = vjp g y 1.0
    let forward = \x -> g (f x)
    let chain = map3 (\(p: (f32,f32)) (T0:[3]f32) (T1:[3]f32) ->
        [
            T0[0]*p.0 + T1[0]*p.1,
            T0[1]*p.0 + T1[1]*p.1,
            T0[2]*p.0 + T1[2]*p.1
        ]) 
        dg_df df_dx_1 df_dx_2

    let ad = vjp forward x 1.0
    in (chain, ad)


    

