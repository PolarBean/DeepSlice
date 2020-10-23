import tensorflow.keras.backend as K

def make_tensor_absolute(ox, oy, oz, xx, yy, zz):
    x = K.sum((xx, ox), axis=0, keepdims=False)
    y = K.sum((yy, oy), axis=0, keepdims=False)
    z = K.sum((zz, oz), axis=0, keepdims=False)
    return x, y, z


def get_tensor_diagonal(qx, qy, qz):
    return K.sqrt(K.sum(K.square((qx, qy, qz)), axis=0))


def convert_tensor(tensor):
    ##define Oxyz
    ox, oy, oz = tensor[:, 0], tensor[:, 1], tensor[:, 2]
    ##define Uxyz
    ux, uy, uz = tensor[:, 3], tensor[:, 4], tensor[:, 5]
    ##define Vxyz
    vx, vy, vz = tensor[:, 6], tensor[:, 7], tensor[:, 8]
    ##define Qxyz
    ##Cheeky use of make_tensor_absolute as the result is still a vector
    wx, wy, wz = make_tensor_absolute(ux, uy, uz, vx, vy, vz)
    return ox, oy, oz, ux, uy, uz, vx, vy, vz, wx, wy, wz


def euclidean_dist_between_tensors(
    T1ox,
    T1oy,
    T1oz,
    T1ux,
    T1uy,
    T1uz,
    T1vx,
    T1vy,
    T1vz,
    T1wx,
    T1wy,
    T1wz,
    T2ox,
    T2oy,
    T2oz,
    T2ux,
    T2uy,
    T2uz,
    T2vx,
    T2vy,
    T2vz,
    T2wx,
    T2wy,
    T2wz,
):

    ##o_squared_error
    ox_squared_dist = (T1ox - T2ox) ** 2
    oy_squared_dist = (T1oy - T2oy) ** 2
    oz_squared_dist = (T1oz - T2oz) ** 2
    ##u_squared_error
    ux_squared_dist = (T1ux - T2ux) ** 2
    uy_squared_dist = (T1uy - T2uy) ** 2
    uz_squared_dist = (T1uz - T2uz) ** 2
    # v_squared_error
    vx_squared_dist = (T1vx - T2vx) ** 2
    vy_squared_dist = (T1vy - T2vy) ** 2
    vz_squared_dist = (T1vz - T2vz) ** 2
    # w_squared_error
    wx_squared_dist = (T1wx - T2wx) ** 2
    wy_squared_dist = (T1wy - T2wy) ** 2
    wz_squared_dist = (T1wz - T2wz) ** 2
    O_loss = K.sqrt(K.sum(
        (ox_squared_dist, oy_squared_dist, oz_squared_dist), axis=0, keepdims=False
    ))
    U_loss = K.sqrt(K.sum(
        (ux_squared_dist, uy_squared_dist, uz_squared_dist), axis=0, keepdims=False
    ))
    V_loss = K.sqrt(K.sum(
        (vx_squared_dist, vy_squared_dist, vz_squared_dist), axis=0, keepdims=False
    ))
    W_loss = K.sqrt(K.sum(
        (wx_squared_dist, wy_squared_dist, wz_squared_dist), axis=0, keepdims=False
    ))
    temp_concat = K.concatenate((O_loss, U_loss, V_loss, W_loss))
    return K.mean(temp_concat)


def Scaled_Finite_Plane_Distance(y_true, y_pred):
    ##Get coordinates from tensor
    (
        true_ox,
        true_oy,
        true_oz,
        true_ux,
        true_uy,
        true_uz,
        true_vx,
        true_vy,
        true_vz,
        true_wx,
        true_wy,
        true_wz,
    ) = convert_tensor(y_true)
    (
        pred_ox,
        pred_oy,
        pred_oz,
        pred_ux,
        pred_uy,
        pred_uz,
        pred_vx,
        pred_vy,
        pred_vz,
        pred_wx,
        pred_wy,
        pred_wz,
    ) = convert_tensor(y_pred)
    ##make Uxyz absolute
    true_abs_ux, true_abs_uy, true_abs_uz = make_tensor_absolute(
        true_ox, true_oy, true_oz, true_ux, true_uy, true_uz
    )
    print(true_abs_ux)
    pred_abs_ux, pred_abs_uy, pred_abs_uz = make_tensor_absolute(
        pred_ox, pred_oy, pred_oz, pred_ux, pred_uy, pred_uz
    )
    ##make Vxyz absolute
    true_abs_vx, true_abs_vy, true_abs_vz = make_tensor_absolute(
        true_ox, true_oy, true_oz, true_vx, true_vy, true_vz
    )
    pred_abs_vx, pred_abs_vy, pred_abs_vz = make_tensor_absolute(
        pred_ox, pred_oy, pred_oz, pred_vx, pred_vy, pred_vz
    )
    ##make Wxyz absolute
    true_abs_wx, true_abs_wy, true_abs_wz = make_tensor_absolute(
        true_ox, true_oy, true_oz, true_wx, true_wy, true_wz
    )
    pred_abs_wx, pred_abs_wy, pred_abs_wz = make_tensor_absolute(
        pred_ox, pred_oy, pred_oz, pred_wx, pred_wy, pred_wz
    )
    ##Get the diagonal length
    true_diagonal = get_tensor_diagonal(true_wx, true_wy, true_wz)
    pred_diagonal = get_tensor_diagonal(pred_wx, pred_wy, pred_wz)
    diagonals = K.concatenate((true_diagonal, pred_diagonal), axis=-1)
    avg_diagonal = K.mean(diagonals, axis=0)
    print(avg_diagonal)
    ##Get the euclidean distance between tensors
    cost = euclidean_dist_between_tensors(
        true_ox,
        true_oy,
        true_oz,
        true_abs_ux,
        true_abs_uy,
        true_abs_uz,
        true_abs_vx,
        true_abs_vy,
        true_abs_vz,
        true_abs_wx,
        true_abs_wy,
        true_abs_wz,
        pred_ox,
        pred_oy,
        pred_oz,
        pred_abs_ux,
        pred_abs_uy,
        pred_abs_uz,
        pred_abs_vx,
        pred_abs_vy,
        pred_abs_vz,
        pred_abs_wx,
        pred_abs_wy,
        pred_abs_wz,
    )
    ##scale_cost by diagonal size
    cost /= avg_diagonal
    return cost * 100
