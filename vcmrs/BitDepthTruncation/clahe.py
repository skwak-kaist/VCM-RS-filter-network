import cv2
import numpy as np
import copy

def clahe_calc_lut_body(src_for_lut, lut, tile_size, tiles_x, clip_limit, lut_scale, hist_size, total_tiles, dtype, shift=0):
    for k in range(0, total_tiles):
        tile_lut = lut[k, :]
        ty, tx = divmod(k, tiles_x)

        tile_x = tx * tile_size[0]
        tile_y = ty * tile_size[1]
        tile_width, tile_height = tile_size

        tileROI = np.s_[tile_y:tile_y + tile_height, tile_x:tile_x + tile_width]
        tile = src_for_lut[tileROI]

        tile_hist = np.zeros(hist_size, dtype=int)

        # calc histogram
        tile_hist = np.zeros(hist_size, dtype=int)
        flat_tile = tile.ravel() >> shift
        np.add.at(tile_hist, flat_tile, 1)

        if clip_limit > 0:
            clipped = np.sum(np.maximum(tile_hist - clip_limit, 0))

            tile_hist = np.minimum(tile_hist, clip_limit)

            redist_batch = clipped // len(tile_hist)
            residual = clipped % len(tile_hist)

            tile_hist += redist_batch

            if residual > 0:
                residual_step = max(len(tile_hist) // residual, 1)
                indices = np.arange(0, len(tile_hist), residual_step)[:residual]
                tile_hist[indices] += 1

        # calc LUT
        cdf = np.cumsum(tile_hist, dtype=np.int32)
        info = np.iinfo(dtype)
        tile_lut[:] = np.clip(np.round(cdf * lut_scale), info.min, info.max).astype(dtype)

    return lut, tile_hist



def process_image_vectorized(src, lut, inv_th, tiles_x, tiles_y, ind1_p, ind2_p, xa1_p, xa_p, shift, dtype):
    dst = np.zeros_like(src, dtype=dtype)
    
    y_indices = np.arange(src.shape[0])
    tyf = np.float32(y_indices * inv_th - 0.5)
    ty1 = np.floor(tyf).astype(int)
    ty1 = np.clip(ty1, 0, tiles_y - 1)
    ty2 = np.clip(ty1 + 1, 0, tiles_y - 1)
    
    ya = tyf - ty1
    ya1 = 1.0 - ya
    
    if dtype == np.uint8:
        min_val, max_val = 0, 255
    elif dtype == np.uint16:
        min_val, max_val = 0, 65535
    
    lut_step = lut.strides[0] // lut.itemsize  

    for y in range(src.shape[0]):
        srcRow = src[y, :]
        src_shifted = srcRow >> shift
    
        ind1_1 = ((ind1_p + src_shifted).astype(int)) // lut_step
        ind2_1 = ((ind2_p + src_shifted).astype(int)) // lut_step
        ind1_2 = ((ind1_p + src_shifted).astype(int)) % lut.shape[1]
        ind2_2 = ((ind2_p + src_shifted).astype(int)) % lut.shape[1]
    
        lut_ty1_ind1 = lut[(ty1[y] * tiles_x + ind1_1), ind1_2]
        lut_ty1_ind2 = lut[(ty1[y] * tiles_x + ind2_1), ind2_2]
        lut_ty2_ind1 = lut[(ty2[y] * tiles_x + ind1_1), ind1_2]
        lut_ty2_ind2 = lut[(ty2[y] * tiles_x + ind2_1), ind2_2]
    
        res = ((lut_ty1_ind1 * xa1_p + lut_ty1_ind2 * xa_p) * ya1[y] +
               (lut_ty2_ind1 * xa1_p + lut_ty2_ind2 * xa_p) * ya[y])
    
        dst[y, :] = (np.clip(np.round(res.squeeze()), min_val, max_val).astype(dtype)) << shift
    
    return dst




def clahe_interpolation_body(src, dst, lut, tile_size, tiles_x, tiles_y, dtype, shift=0):
    src_row, src_cols = src.shape

    tmp = np.zeros(src_cols * 4, dtype=np.float32)
    ind1_p = tmp[:src_cols]
    ind2_p = tmp[src_cols:src_cols * 2]
    xa_p = tmp[src_cols * 2:src_cols * 3]
    xa1_p = tmp[src_cols * 3:src_cols * 4]

    lut_step = lut.strides[0] // lut.itemsize
    inv_tw = np.float64(1.0 / tile_size[0])

    for x in range(src_cols):
        txf = np.float32(x * inv_tw - 0.5)

        tx1 = np.floor(txf).astype(int)
        tx2 = tx1 + 1

        xa_p[x] = txf - tx1
        xa1_p[x] = 1.0 - xa_p[x]

        tx1 = max(tx1, 0)
        tx2 = min(tx2, tiles_x - 1)

        ind1_p[x] = tx1 * lut_step
        ind2_p[x] = tx2 * lut_step



    inv_th = np.float32(1.0 / tile_size[1])
    return process_image_vectorized(src, lut, inv_th, tiles_x, tiles_y, ind1_p, ind2_p, xa1_p, xa_p, shift, dtype)


def clahe_custom(src_, clip_limit_, tileGridSize_, dtype_):
    (tiles_x, tiles_y) = tileGridSize_
    hist_size = 256 if dtype_ == np.uint8 else 65536

    # step 1: divide tiles(output: src_for_lut)
    if src_.shape[1] % tiles_x == 0 and src_.shape[0] % tiles_y == 0:
        tile_size = (src_.shape[1] // tiles_x, src_.shape[0] // tiles_y)
        src_for_lut = src_
    else:
        border_bottom = tiles_y - (src_.shape[0] % tiles_y)
        border_right = tiles_x - (src_.shape[1] % tiles_x)
        src_ext = cv2.copyMakeBorder(
            src_,
            0, border_bottom,
            0, border_right,
            borderType=cv2.BORDER_REFLECT_101
        )
        tile_size = (src_ext.shape[1] // tiles_x, src_ext.shape[0] // tiles_y)
        src_for_lut = src_ext

    tile_size_total = tile_size[0] * tile_size[1]
    lut_scale = np.float32(hist_size - 1) / np.float32(tile_size_total)

    clip_limit = 0
    if clip_limit_ > 0.0:
        clip_limit = int(clip_limit_ * tile_size_total / hist_size)
        clip_limit = max(clip_limit, 1)

    dst = np.zeros_like(src_for_lut)
    lut = np.zeros((tiles_x * tiles_y, hist_size), dtype=dtype_)


    lut, tile_hist = clahe_calc_lut_body(src_for_lut, lut, tile_size, tiles_x, clip_limit, lut_scale, hist_size, tiles_x * tiles_y, dtype_)

    dst = clahe_interpolation_body(src_, dst, lut, tile_size, tiles_x, tiles_y, dtype_)

    return dst
