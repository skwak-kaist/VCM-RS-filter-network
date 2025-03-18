import vcmrs
import sys
import cv2
import numpy as np
import numba


@numba.njit
def InterpolationNumbaInt12tapLuma(img_out, img_in):
  filters_luma_alt = [
    [ 0, 0, 0, 0, 0, 256, 0, 0, 0, 0, 0, 0, ],
    [ 1, -1, 0, 3, -12, 253, 16, -6, 2, 0, 0, 0, ],
    [ 0, 0, -3, 9, -24, 250, 32, -11, 4, -1, 0, 0, ],
    [ 0, 0, -4, 12, -32, 241, 52, -18, 8, -4, 2, -1, ],
    [ 0, 1, -6, 15, -38, 228, 75, -28, 14, -7, 3, -1, ],
    [ 0, 1, -7, 18, -43, 214, 96, -33, 16, -8, 3, -1, ],
    [ 1, 0, -6, 17, -44, 196, 119, -40, 20, -10, 4, -1, ],
    [ 0, 2, -9, 21, -47, 180, 139, -43, 20, -10, 4, -1, ],
    [ -1, 3, -9, 21, -46, 160, 160, -46, 21, -9, 3, -1, ],
    [ -1, 4, -10, 20, -43, 139, 180, -47, 21, -9, 2, 0, ],
    [ -1, 4, -10, 20, -40, 119, 196, -44, 17, -6, 0, 1, ],
    [ -1, 3, -8, 16, -33, 96, 214, -43, 18, -7, 1, 0, ],
    [ -1, 3, -7, 14, -28, 75, 228, -38, 15, -6, 1, 0, ],
    [ -1, 2, -4, 8, -18, 52, 241, -32, 12, -4, 0, 0, ],
    [ 0, 0, -1, 4, -11, 32, 250, -24, 9, -3, 0, 0, ],
    [ 0, 0, 0, 2, -6, 16, 253, -12, 3, 0, -1, 1, ],
  ]
  
  img_out_dy, img_out_dx = img_out.shape[0:2]
  img_in_dy, img_in_dx = img_in.shape[0:2]

  img_in_dx1 = img_in_dx - 1
  img_in_dy1 = img_in_dy - 1

  x_nom = (img_in_dx) << 14
  x_den = img_out_dx

  if x_den == 0:
    x_nom = 0
    x_den = 1

  y_nom = (img_in_dy) << 14
  y_den = img_out_dy

  if y_den == 0:
    y_nom = 0
    y_den = 1

  scaling_factor_x = (x_nom + (x_den >> 1)) // x_den
  scaling_factor_y = (y_nom + (y_den >> 1)) // y_den

  img_tmp = np.zeros( (img_in_dy, img_out_dx), dtype = np.int32 )
  for x in range(img_out_dx):
    xp = (x  * scaling_factor_x + 512) >> 10
    integer = xp >> 4
    frac = xp & 15
    for y in range(img_in_dy):
      sumVal = 0
      for k in [0,1,2,3,4,5,6,7,8,9,10,11]:
        pos = max(0, min(img_in_dx1, integer+k-5))
        sumVal += filters_luma_alt[frac][k] * img_in[y, pos]
      img_tmp[y, x] = sumVal

  for y in range(img_out_dy):
    yp = (y  * scaling_factor_y + 512 ) >> 10
    integer = yp >> 4
    frac = yp & 15
    for x in range(img_out_dx):
      sumVal = 0
      for k in [0,1,2,3,4,5,6,7,8,9,10,11]:
        pos = max(0, min(img_in_dy1, integer+k-5))
        sumVal += filters_luma_alt[frac][k] * img_tmp[pos, x]
      img_out[y, x] =  max(0, min(1023, (sumVal + 32768 ) >> 16))



@numba.njit
def InterpolationNumbaInt6tapChroma(img_out, img_in, scaleX, scaleY):
  filters_chroma_alt = [
    [0, 0, 256, 0, 0, 0, ],
    [ 1, -6, 256, 6, -1, 0, ],
    [ 2, -11, 254, 14, -4, 1, ],
    [ 4, -18, 252, 23, -6, 1, ],
    [ 6, -24, 249, 32, -9, 2, ],
    [ 6, -26, 244, 41, -12, 3, ],
    [ 7, -30, 239, 53, -18, 5, ],
    [ 8, -34, 235, 61, -19, 5, ],
    [ 10, -38, 228, 72, -22, 6, ],
    [ 10, -39, 220, 84, -26, 7, ],
    [ 10, -40, 213, 94, -29, 8, ],
    [ 11, -42, 205, 105, -32, 9, ],
    [ 11, -42, 196, 116, -35, 10, ],
    [ 11, -42, 186, 128, -37, 10, ],
    [ 11, -42, 177, 138, -38, 10, ],
    [ 11, -41, 167, 148, -40, 11, ],
    [ 11, -41, 158, 158, -41, 11, ],
    [ 11, -40, 148, 167, -41, 11, ],
    [ 10, -38, 138, 177, -42, 11, ],
    [ 10, -37, 128, 186, -42, 11, ],
    [ 10, -35, 116, 196, -42, 11, ],
    [ 9, -32, 105, 205, -42, 11, ],
    [ 8, -29, 94, 213, -40, 10, ],
    [ 7, -26, 84, 220, -39, 10, ],
    [ 6, -22, 72, 228, -38, 10, ],
    [ 5, -19, 61, 235, -34, 8, ],
    [ 5, -18, 53, 239, -30, 7, ],
    [ 3, -12, 41, 244, -26, 6, ],
    [ 2, -9, 32, 249, -24, 6, ],
    [ 1, -6, 23, 252, -18, 4, ],
    [ 1, -4, 14, 254, -11, 2, ],
    [ 0, -1, 6, 256, -6, 1, ],
  ]

  img_out_dy, img_out_dx = img_out.shape[0:2]
  img_in_dy, img_in_dx = img_in.shape[0:2]

  img_in_dx1 = img_in_dx - 1
  img_in_dy1 = img_in_dy - 1

  x_nom = (img_in_dx) << 14
  x_den = img_out_dx

  if x_den == 0:
    x_nom = 0
    x_den = 1

  y_nom = (img_in_dy) << 14
  y_den = img_out_dy

  if y_den == 0:
    y_nom = 0
    y_den = 1

  scaling_factor_x = (x_nom + (x_den >> 1)) // x_den
  scaling_factor_y = (y_nom + (y_den >> 1)) // y_den

  img_tmp = np.zeros( (img_in_dy, img_out_dx), dtype = np.int32 )
  pos_shift_x = 9+scaleX
  pos_shift_y = 9+scaleY
  pos_shift_x_offset = (1<<(pos_shift_x-1))
  pos_shift_y_offset = (1<<(pos_shift_y-1))

  for x in range(img_out_dx):
    xp = ((x<<scaleX)  * scaling_factor_x + pos_shift_x_offset) >> pos_shift_x
    integer = xp >> 5
    frac = xp & 31
    for y in range(img_in_dy):
      sumVal = 0
      for k in [0,1,2,3,4,5]:
        pos = max(0, min(img_in_dx1, integer+k-2))
        sumVal += filters_chroma_alt[frac][k] * img_in[y, pos]
      img_tmp[y, x] = sumVal


  for y in range(img_out_dy):
    yp = ((y<<scaleY)  * scaling_factor_y + pos_shift_y_offset) >> pos_shift_y
    integer = yp >> 5
    frac = yp & 31
    for x in range(img_out_dx):
      sumVal = 0
      for k in [0,1,2,3,4,5]:
        pos = max(0, min(img_in_dy1, integer+k-2))
        sumVal += filters_chroma_alt[frac][k] * img_tmp[pos, x]
      img_out[y, x] =  max(0, min(1023, (sumVal + 32768 ) >> 16))