import vcmrs
import sys
import cv2
import numpy as np
import numba

# ix=Clip(0,w_in-2,floor(x))
# iy=Clip(0,h_in-2,floor(y))
# fx=x-ix
# fy=y-iy
# LT_c=in[c,iy,ix]
# RT_c=in[c,iy,ix+1]
# LB_c=in[c,iy+1,ix]
# RB_c=in[c,iy+1,ix+1]
# 
# out[c]= LT_c∙(1-fx)∙(1-fy) + LB_c∙(1-fx)∙fy + RT_c∙(1-fy)∙fx + RB_c∙fx∙fy

#Variable scaleWidth is set equal to ( wOut – 1 ) / ( wIn – 1 ).
#Variable scaleHeight is set equal to ( hOut – 1 ) / (hIn – 1 ).
#Tensor xOut is produced, with i in the range 0 to wOut – 1 inclusive, j in the range 0 to hOut – 1 inclusive, and c in the range 0 to channels – 1 inclusive, as follows:
#	x0In = i / scaleWidth
#	y0In = j / scaleHeight
#	x1In = Min( x0In + 1, wIn – 1 )
#	y1In = Min( y0In + 1, hIn – 1 )
#	xFrac = ( i – ( x0In * scaleWidth ) ) ÷ scaleWidth
#	yFrac = ( j – ( y0In * scaleHEight ) ) ÷ scaleHeight
#	valTL = x[ c, y0In, x0In ]
#	valTR = x[ c, y0In, x1In ]
#	valBL = x[ c, y1In, x0In ]
#	valBR = x[ c, y1In, x1In ]
#	xOut[ c, j, i ] = valTL * ( 1 – xFrac ) * ( 1 – yFrac ) + valTR * xFrac * ( 1 – yFRac ) + valBL * ( 1 – xFrac ) * yFrac + valBR * xFrac * yFrac 
#
#

def BilinearInterpolationCV2(img_out, img_in):

  M = cv2.getRotationMatrix2D( (0,0) ,0, 0 ) 
      
  inp_size_x = img_in.shape[1]
  inp_size_y = img_in.shape[0]
  
  out_size_x = img_out.shape[1]
  out_size_y = img_out.shape[0]
  
  M[0,0] = (inp_size_x-1)/(out_size_x-1) if out_size_x>1 else 0
  M[0,1] = 0
  M[0,2] = 0
  
  M[1,0] = 0
  M[1,1] = (inp_size_y-1)/(out_size_y-1) if out_size_y>1 else 0
  M[1,2] = 0
  
  img_out[ :,: ] = cv2.warpAffine(img_in,M,(out_size_x,out_size_y),flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP )



def BilinearInterpolationNumpyFloat(img_out, img_in):

  img_out_dy, img_out_dx = img_out.shape[0:2]

  img_in_dy, img_in_dx = img_in.shape[0:2]

  x_coords = np.arange(0, img_out_dx, 1, dtype=np.uint32)
  y_coords = np.arange(0, img_out_dy, 1, dtype=np.uint32)
  
  x_nom = img_in_dx-1
  x_den = img_out_dx-1
  if x_den==0:
    x_nom = 0
    x_den = 1
    
  y_nom = img_in_dy-1
  y_den = img_out_dy-1
  if y_den==0:
    y_nom = 0
    y_den = 1
    

  x_coords_int0 = x_coords * x_nom // x_den
  y_coords_int0 = y_coords * y_nom // y_den
  
  x_coords_int1 = np.minimum(x_coords_int0+1, img_in_dx-1)
  y_coords_int1 = np.minimum(y_coords_int0+1, img_in_dy-1)
  
  x_coords_frac = x_coords * x_nom / x_den - x_coords_int0
  y_coords_frac = y_coords * y_nom / y_den - y_coords_int0

  x_coords_frac1 = 1.0-x_coords_frac
  y_coords_frac1 = 1.0-y_coords_frac
  
  y_coords_frac  = y_coords_frac .reshape( (img_out_dy,1) )
  y_coords_frac1 = y_coords_frac1.reshape( (img_out_dy,1) )

  img_LT = img_in[:, x_coords_int0][y_coords_int0]
  img_RT = img_in[:, x_coords_int1][y_coords_int0]
  img_LB = img_in[:, x_coords_int0][y_coords_int1]
  img_RB = img_in[:, x_coords_int1][y_coords_int1]
  
  img_out[:,:,0] = np.round(img_LT[:,:,0]*x_coords_frac1*y_coords_frac1 + img_RT[:,:,0]*x_coords_frac*y_coords_frac1 + img_LB[:,:,0]*x_coords_frac1*y_coords_frac + img_RB[:,:,0]*x_coords_frac*y_coords_frac)
  img_out[:,:,1] = np.round(img_LT[:,:,1]*x_coords_frac1*y_coords_frac1 + img_RT[:,:,1]*x_coords_frac*y_coords_frac1 + img_LB[:,:,1]*x_coords_frac1*y_coords_frac + img_RB[:,:,1]*x_coords_frac*y_coords_frac)
  img_out[:,:,2] = np.round(img_LT[:,:,2]*x_coords_frac1*y_coords_frac1 + img_RT[:,:,2]*x_coords_frac*y_coords_frac1 + img_LB[:,:,2]*x_coords_frac1*y_coords_frac + img_RB[:,:,2]*x_coords_frac*y_coords_frac)
  
  
def _mul3(img, fx, fy):
  return np.multiply( np.multiply(img, fx, dtype=np.uint32), fy, dtype=np.uint32)

def BilinearInterpolationNumpyInt(img_out, img_in, precision = 8):

  img_out_dy, img_out_dx = img_out.shape[0:2]
  img_in_dy, img_in_dx = img_in.shape[0:2]

  x_coords = np.arange(0, img_out_dx, 1, dtype=np.uint32)
  y_coords = np.arange(0, img_out_dy, 1, dtype=np.uint32)
  
  x_nom = img_in_dx-1
  x_den = img_out_dx-1
  if x_den==0:
    x_nom = 0
    x_den = 1
    
  y_nom = img_in_dy-1
  y_den = img_out_dy-1
  if y_den==0:
    y_nom = 0
    y_den = 1
    

  x_coords_int0 = x_coords * x_nom // x_den
  y_coords_int0 = y_coords * y_nom // y_den
  
  x_coords_int1 = np.minimum(x_coords_int0+1, img_in_dx-1)
  y_coords_int1 = np.minimum(y_coords_int0+1, img_in_dy-1)
  
  x_coords_frac = ( x_coords * (x_nom<<precision) // x_den ) & ((1<<precision)-1)
  y_coords_frac = ( y_coords * (y_nom<<precision) // y_den ) & ((1<<precision)-1)

  x_coords_frac1 = (1<<precision) - x_coords_frac
  y_coords_frac1 = (1<<precision) - y_coords_frac
  
  y_coords_frac  = y_coords_frac .reshape( (img_out_dy,1) )
  y_coords_frac1 = y_coords_frac1.reshape( (img_out_dy,1) )

  img_LT = img_in[:, x_coords_int0][y_coords_int0]
  img_RT = img_in[:, x_coords_int1][y_coords_int0]
  img_LB = img_in[:, x_coords_int0][y_coords_int1]
  img_RB = img_in[:, x_coords_int1][y_coords_int1]
  
  precision2 = precision*2
  img_out[:,:,0] = ( _mul3(img_LT[:,:,0],x_coords_frac1,y_coords_frac1) + _mul3(img_RT[:,:,0],x_coords_frac,y_coords_frac1) + _mul3(img_LB[:,:,0],x_coords_frac1,y_coords_frac) + _mul3(img_RB[:,:,0],x_coords_frac,y_coords_frac) + (1<<(precision2-1))  ) >> precision2
  img_out[:,:,1] = ( _mul3(img_LT[:,:,1],x_coords_frac1,y_coords_frac1) + _mul3(img_RT[:,:,1],x_coords_frac,y_coords_frac1) + _mul3(img_LB[:,:,1],x_coords_frac1,y_coords_frac) + _mul3(img_RB[:,:,1],x_coords_frac,y_coords_frac) + (1<<(precision2-1))  ) >> precision2
  img_out[:,:,2] = ( _mul3(img_LT[:,:,2],x_coords_frac1,y_coords_frac1) + _mul3(img_RT[:,:,2],x_coords_frac,y_coords_frac1) + _mul3(img_LB[:,:,2],x_coords_frac1,y_coords_frac) + _mul3(img_RB[:,:,2],x_coords_frac,y_coords_frac) + (1<<(precision2-1))  ) >> precision2


BilinearInterpolationNumbaIntLocalsDict = {
  'precision':  numba.int32,
  'img_out_dx': numba.int64,
  'img_out_dy': numba.int64,
  'img_in_dx':  numba.int64,
  'img_in_dy':  numba.int64,
  'img_in_dx1': numba.int64,
  'img_in_dy1': numba.int64,
  'mask': numba.int64,
  'precision_val1': numba.int64,
  'precision2': numba.int64,
  'offset': numba.int64,
  'x': numba.int64,
  'y': numba.int64,
  'xp': numba.int64,
  'yp': numba.int64,
  'xi': numba.int64,
  'yi': numba.int64,
  'xf': numba.int64,
  'yf': numba.int64,
  'xf1': numba.int64,
  'yf1': numba.int64,
  'xi1': numba.int64,
  'yi1': numba.int64,  
}

#@numba.njit
@numba.jit(nopython=True, fastmath = True, locals=BilinearInterpolationNumbaIntLocalsDict )
def BilinearInterpolationNumbaIntLocals(img_out, img_in, computation_precision=8):
  
  precision = computation_precision
  
  #precision = (63-bit_depth-2)//2
  
  img_out_dy, img_out_dx = img_out.shape[0:2]
  img_in_dy, img_in_dx = img_in.shape[0:2]
  
  img_in_dx1 = img_in_dx-1
  img_in_dy1 = img_in_dy-1
    
  x_nom = img_in_dx1 << precision
  x_den = img_out_dx-1
  
  if x_den==0:
    x_nom = 0
    x_den = 1
  
  y_nom = img_in_dy1 << precision
  y_den = img_out_dy-1
  
  if y_den==0:
    y_nom = 0
    y_den = 1
  
  mask = (1<<precision)-1
  precision_val1 = np.int64(1)<<precision
  precision2 = precision*2
  
  offset = np.int64(1)<<(precision*2-1)
   
  for y in range(img_out_dy):
    for x in range(img_out_dx):
    
      xp = x*x_nom//x_den
      yp = y*y_nom//y_den
      
      xi = xp>>precision
      yi = yp>>precision
      
      xf = xp&mask
      yf = yp&mask
      
      xf1 = precision_val1 - xf
      yf1 = precision_val1 - yf
      
      xi1 = min(xi+1, img_in_dx1)
      yi1 = min(yi+1, img_in_dy1)
      
      #img_out[y, x] = ( img_in[yi, xi]*xf1*yf1 + img_in[yi, xi1]*xf*yf1 + img_in[yi1, xi]*xf1*yf + img_in[yi1, xi1]*xf*yf + offset ) >> precision2
      
      img_out[y, x, 0] = ( img_in[yi, xi, 0]*xf1*yf1 + img_in[yi, xi1, 0]*xf*yf1 + img_in[yi1, xi, 0]*xf1*yf + img_in[yi1, xi1, 0]*xf*yf + offset ) >> precision2
      img_out[y, x, 1] = ( img_in[yi, xi, 1]*xf1*yf1 + img_in[yi, xi1, 1]*xf*yf1 + img_in[yi1, xi, 1]*xf1*yf + img_in[yi1, xi1, 1]*xf*yf + offset ) >> precision2
      img_out[y, x, 2] = ( img_in[yi, xi, 2]*xf1*yf1 + img_in[yi, xi1, 2]*xf*yf1 + img_in[yi1, xi, 2]*xf1*yf + img_in[yi1, xi1, 2]*xf*yf + offset ) >> precision2
      

@numba.njit
def BilinearInterpolationNumbaInt(img_out, img_in, precision = 8): # argument prec_type = np.int64  makes it slower ???
  
  prec_type = np.int64
  
  #precision = (63-bit_depth-2)//2
  
  img_out_dy, img_out_dx = img_out.shape[0:2]
  img_in_dy, img_in_dx = img_in.shape[0:2]
  
  img_in_dx1 = prec_type(img_in_dx-1)
  img_in_dy1 = prec_type(img_in_dy-1)
    
  x_nom = (prec_type(img_in_dx-1) << precision)
  x_den = prec_type(img_out_dx-1)
  
  if x_den==0:
    x_nom = prec_type(0)
    x_den = prec_type(1)
  
  y_nom = (prec_type(img_in_dy-1) << precision)
  y_den = prec_type(img_out_dy-1)
  
  if y_den==0:
    y_nom = prec_type(0)
    y_den = prec_type(1)
  
  mask = prec_type((1<<precision)-1)
  precision_val1 = prec_type(1)<<precision
  precision2 = prec_type(precision*2)
  
  offset = prec_type(1<<(precision*2-1))
  
  x_nom = prec_type(x_nom)
  x_den = prec_type(x_den)
  
  y_nom = prec_type(y_nom)
  y_den = prec_type(y_den)
  
  precision = np.int32(precision)
    
  for y in range(img_out_dy):
    for x in range(img_out_dx):
    
      xp = prec_type(x)*x_nom//x_den
      yp = prec_type(y)*y_nom//y_den
      
      xi = prec_type(xp>>precision)
      yi = prec_type(yp>>precision)
      
      xf = prec_type(xp&mask)
      yf = prec_type(yp&mask)
      
      xf1 = prec_type(precision_val1-xf)
      yf1 = prec_type(precision_val1-yf)
      
      xi1 = prec_type(np.minimum(xi+1, img_in_dx1))
      yi1 = prec_type(np.minimum(yi+1, img_in_dy1))
      
      #img_out[y, x] = ( img_in[yi, xi]*xf1*yf1 + img_in[yi, xi1]*xf*yf1 + img_in[yi1, xi]*xf1*yf + img_in[yi1, xi1]*xf*yf + offset ) >> precision2
      
      img_out[y, x, 0] = ( prec_type(img_in[yi, xi, 0])*xf1*yf1 + prec_type(img_in[yi, xi1, 0])*xf*yf1 + prec_type(img_in[yi1, xi, 0])*xf1*yf + prec_type(img_in[yi1, xi1, 0])*xf*yf + offset ) >> precision2
      img_out[y, x, 1] = ( prec_type(img_in[yi, xi, 1])*xf1*yf1 + prec_type(img_in[yi, xi1, 1])*xf*yf1 + prec_type(img_in[yi1, xi, 1])*xf1*yf + prec_type(img_in[yi1, xi1, 1])*xf*yf + offset ) >> precision2
      img_out[y, x, 2] = ( prec_type(img_in[yi, xi, 2])*xf1*yf1 + prec_type(img_in[yi, xi1, 2])*xf*yf1 + prec_type(img_in[yi1, xi, 2])*xf1*yf + prec_type(img_in[yi1, xi1, 2])*xf*yf + offset ) >> precision2
      

@numba.njit
def BilinearInterpolationNumbaIntOneComp(img_out, img_in, precision = 8): # argument prec_type = np.int64  makes it slower ???
  
  prec_type = np.int64
  
  #precision = (63-bit_depth-2)//2
  
  img_out_dy, img_out_dx = img_out.shape[0:2]
  img_in_dy, img_in_dx = img_in.shape[0:2]
  
  img_in_dx1 = prec_type(img_in_dx-1)
  img_in_dy1 = prec_type(img_in_dy-1)
    
  x_nom = (prec_type(img_in_dx-1) << precision)
  x_den = prec_type(img_out_dx-1)
  
  if x_den==0:
    x_nom = prec_type(0)
    x_den = prec_type(1)
  
  y_nom = (prec_type(img_in_dy-1) << precision)
  y_den = prec_type(img_out_dy-1)
  
  if y_den==0:
    y_nom = prec_type(0)
    y_den = prec_type(1)
  
  mask = prec_type((1<<precision)-1)
  precision_val1 = prec_type(1)<<precision
  precision2 = prec_type(precision*2)
  
  offset = prec_type(1<<(precision*2-1))
  
  x_nom = prec_type(x_nom)
  x_den = prec_type(x_den)
  
  y_nom = prec_type(y_nom)
  y_den = prec_type(y_den)
  
  precision = np.int32(precision)
    
  for y in range(img_out_dy):
    for x in range(img_out_dx):
    
      xp = prec_type(x)*x_nom//x_den
      yp = prec_type(y)*y_nom//y_den
      
      xi = prec_type(xp>>precision)
      yi = prec_type(yp>>precision)
      
      xf = prec_type(xp&mask)
      yf = prec_type(yp&mask)
      
      xf1 = prec_type(precision_val1-xf)
      yf1 = prec_type(precision_val1-yf)
      
      xi1 = prec_type(np.minimum(xi+1, img_in_dx1))
      yi1 = prec_type(np.minimum(yi+1, img_in_dy1))
      
      #img_out[y, x] = ( img_in[yi, xi]*xf1*yf1 + img_in[yi, xi1]*xf*yf1 + img_in[yi1, xi]*xf1*yf + img_in[yi1, xi1]*xf*yf + offset ) >> precision2
      
      img_out[y, x] = ( prec_type(img_in[yi, xi])*xf1*yf1 + prec_type(img_in[yi, xi1])*xf*yf1 + prec_type(img_in[yi1, xi])*xf1*yf + prec_type(img_in[yi1, xi1])*xf*yf + offset ) >> precision2


@numba.njit
#@numba.jit(nopython=True, fastmath = True)
def BilinearInterpolationNumbaFloat(img_out, img_in):
 
  img_out_dy, img_out_dx = img_out.shape[0:2]
  img_in_dy, img_in_dx = img_in.shape[0:2]
  
  img_in_dx1 = img_in_dx-1
  img_in_dy1 = img_in_dy-1
    
  x_nom = img_in_dx-1
  x_den = img_out_dx-1
  
  if x_den==0:
    x_nom = 0
    x_den = 1
  
  y_nom = img_in_dy-1
  y_den = img_out_dy-1
  
  if y_den==0:
    y_nom = 0
    y_den = 1
  
  x_scale = x_nom/x_den
  y_scale = y_nom/y_den
  
  for y in range(img_out_dy):
    for x in range(img_out_dx):
    
      xp = x * x_scale
      yp = y * y_scale
      
      xi = int(np.floor(xp))
      yi = int(np.floor(yp))
      
      xf = xp - xi
      yf = yp - yi
      
      xf1 = 1.0-xf
      yf1 = 1.0-yf
      
      xi1 = min(xi+1, img_in_dx1)
      yi1 = min(yi+1, img_in_dy1)
      
      img_out[y, x, 0] = int( img_in[yi, xi, 0]*xf1*yf1 + img_in[yi, xi1, 0]*xf*yf1 + img_in[yi1, xi, 0]*xf1*yf + img_in[yi1, xi1, 0]*xf*yf + 0.5 )
      img_out[y, x, 1] = int( img_in[yi, xi, 1]*xf1*yf1 + img_in[yi, xi1, 1]*xf*yf1 + img_in[yi1, xi, 1]*xf1*yf + img_in[yi1, xi1, 1]*xf*yf + 0.5 )
      img_out[y, x, 2] = int( img_in[yi, xi, 2]*xf1*yf1 + img_in[yi, xi1, 2]*xf*yf1 + img_in[yi1, xi, 2]*xf1*yf + img_in[yi1, xi1, 2]*xf*yf + 0.5 )
      




@numba.njit
def BilinearInterpolationNumbaIntHorizontal(img_out, img_in, precision=8):

  img_out_dy, img_out_dx = img_out.shape[0:2]
  img_in_dy, img_in_dx = img_in.shape[0:2]
  
  img_in_dx1 = img_in_dx-1
  img_in_dy1 = img_in_dy-1
    
  x_nom = (img_in_dx-1) << precision
  x_den = img_out_dx-1
  
  if x_den==0:
    x_nom = 0
    x_den = 1
  
  mask = (1<<precision)-1  
  precision_val1 = 1<<precision
  
  offset = 1<<(precision-1)
  
  for y in range(img_out_dy):
    for x in range(img_out_dx):
    
      xp = x*x_nom//x_den
      xi = xp>>precision
      xf = xp&mask
      xf1 = precision_val1-xf
      xi1 = min(xi+1, img_in_dx1)
      img_out[y, x, 0] = ( img_in[y, xi, 0]*xf1 + img_in[y, xi1, 0]*xf + offset ) >> precision
      img_out[y, x, 1] = ( img_in[y, xi, 1]*xf1 + img_in[y, xi1, 1]*xf + offset ) >> precision
      img_out[y, x, 2] = ( img_in[y, xi, 2]*xf1 + img_in[y, xi1, 2]*xf + offset ) >> precision
      

@numba.njit
def BilinearInterpolationNumbaIntVertical(img_out, img_in, precision=8):

  img_out_dy, img_out_dx = img_out.shape[0:2]
  img_in_dy, img_in_dx = img_in.shape[0:2]
  
  img_in_dx1 = img_in_dx-1
  img_in_dy1 = img_in_dy-1
  
  y_nom = (img_in_dy-1) << precision
  y_den = img_out_dy-1
  
  if y_den==0:
    y_nom = 0
    y_den = 1
  
  mask = (1<<precision)-1  
  precision_val1 = 1<<precision
  
  offset = 1<<(precision-1)
  
  for y in range(img_out_dy):
    for x in range(img_out_dx):
    
      yp = y*y_nom//y_den
      yi = yp>>precision
      yf = yp&mask
      yf1 = precision_val1-yf
      yi1 = min(yi+1, img_in_dy1)
      
      img_out[y, x, 0] = ( img_in[yi, x, 0]*yf1 + img_in[yi1, x, 0]*yf + offset ) >> precision
      img_out[y, x, 1] = ( img_in[yi, x, 1]*yf1 + img_in[yi1, x, 1]*yf + offset ) >> precision
      img_out[y, x, 2] = ( img_in[yi, x, 2]*yf1 + img_in[yi1, x, 2]*yf + offset ) >> precision
      
def BilinearInterpolationNumbaIntHorizontalVerticalAllocTmp(img_out, img_in):
  img_out_dy, img_out_dx = img_out.shape[0:2]
  img_in_dy, img_in_dx = img_in.shape[0:2]
  return np.zeros( (img_in_dy, img_out_dx, 3), dtype = img_in.dtype )

def BilinearInterpolationNumbaIntHorizontalVertical(img_out, img_in, img_tmp=None, precision=8):
  
  if img_tmp is None:
    img_tmp = BilinearInterpolationNumbaIntHorizontalVerticalAllocTmp(img_out, img_in)
    
  BilinearInterpolationNumbaIntHorizontal(img_tmp, img_in, precision)
  BilinearInterpolationNumbaIntVertical(img_out, img_tmp, precision)
