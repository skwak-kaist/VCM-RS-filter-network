import os
import sys
import numpy as np
#from scipy.stats import pearsonr # removed and implemented manually for better numerical reproducibility

import ast

def load_descriptor_from_file(descriptor_file):
  with open(descriptor_file, 'r') as file:
    content = file.read()
    content = content.replace(" ","")
    content = content.replace("\r","")
    content = content.replace("\n","")
    content = content.replace("\t","")
    return ast.literal_eval(content)

def save_descriptor_to_file(descriptor_file, content):
  with open(descriptor_file, 'w', newline="\n") as file:
    file.write("{\n")
    for frame in content.keys():
      objr = str(content[frame]).replace(" ","")
      file.write(f" {frame}: {objr},\n")
    file.write("}\n")

def determine_format_characteristics(width, height, chroma_format, bit_depth):
  if chroma_format == "420":
    y_size = width * height
    uv_size = (width // 2) * (height // 2)
    uv_shape = (height // 2, width // 2)
  elif chroma_format == "422":
    y_size = width * height
    uv_size = (width // 2) * height
    uv_shape = (height, width // 2)
  elif chroma_format == "444":
    y_size = width * height
    uv_size = width * height
    uv_shape = (height, width)
  else:
    raise ValueError("Unsupported chroma format: {}".format(chroma_format))
    
  bytes_per_pixel = 2 if bit_depth > 8 else 1
  
  frame_size_bytes = (y_size+uv_size*2)*bytes_per_pixel
  
  return y_size, uv_size, uv_shape, bytes_per_pixel, frame_size_bytes


def yuv_select_frames(frames_list, input_fname, output_fname, width, height, chroma_format, bit_depth):
  
  y_size, uv_size, uv_shape, bytes_per_pixel, frame_size_bytes = determine_format_characteristics(width, height, chroma_format, bit_depth)
  
  input_file = open(input_fname, 'rb')
  output_file = open(output_fname, 'wb')

  for frame_index in frames_list:
    input_file.seek( frame_index*frame_size_bytes )
    
    y_buffer = input_file.read(y_size * bytes_per_pixel)
    if len(y_buffer) == 0:
      break
    
    y = np.frombuffer(y_buffer,                                   dtype=np.uint16 if bytes_per_pixel == 2 else np.uint8).reshape((height, width)).copy()
    u = np.frombuffer(input_file.read(uv_size * bytes_per_pixel), dtype=np.uint16 if bytes_per_pixel == 2 else np.uint8).reshape(uv_shape).copy()
    v = np.frombuffer(input_file.read(uv_size * bytes_per_pixel), dtype=np.uint16 if bytes_per_pixel == 2 else np.uint8).reshape(uv_shape).copy()
    
    output_file.write(y.tobytes())
    output_file.write(u.tobytes())
    output_file.write(v.tobytes())
    
  input_file.close()
  output_file.close()
  return 
  
  
def yuv_chroma_clear(num_frames, input_fname, input_frame_start, output_fname, width, height, chroma_format, bit_depth, graying_specification):
  
  y_size, uv_size, uv_shape, bytes_per_pixel, frame_size_bytes = determine_format_characteristics(width, height, chroma_format, bit_depth)
    
  input_file = open(input_fname, 'rb')
  output_file = open(output_fname, 'wb')

  input_file.seek( input_frame_start*frame_size_bytes )
  frame_index = 0
  while True:
    y_buffer = input_file.read(y_size * bytes_per_pixel)
    if len(y_buffer) == 0:
      break
    y = np.frombuffer(y_buffer,                                   dtype=np.uint16 if bytes_per_pixel == 2 else np.uint8).reshape((height, width)).copy()
    u = np.frombuffer(input_file.read(uv_size * bytes_per_pixel), dtype=np.uint16 if bytes_per_pixel == 2 else np.uint8).reshape(uv_shape).copy()
    v = np.frombuffer(input_file.read(uv_size * bytes_per_pixel), dtype=np.uint16 if bytes_per_pixel == 2 else np.uint8).reshape(uv_shape).copy()

    try:
      gr = graying_specification[frame_index]
    except:
      gr = bool(graying_specification)
      
    if gr:
      clear_value = (1<<(bit_depth-1))
      u[:,:] = clear_value
      v[:,:] = clear_value
      
    output_file.write(y.tobytes())
    output_file.write(u.tobytes())
    output_file.write(v.tobytes())
    
    frame_index += 1
    if (num_frames>0) and (frame_index>=num_frames): break
    
    
  input_file.close()
  output_file.close()
  return frame_index

def cov_f(a,b):
  a_mean = np.mean(np.array(a, dtype=np.float64))
  b_mean = np.mean(np.array(b, dtype=np.float64))
  a = a-a_mean
  b = b-b_mean
  return np.float64( np.mean( a.astype(np.float64)*b.astype(np.float64) ) )

def pearsonr_f(a,b):
  return cov_f(a,b)/np.sqrt( cov_f(a,a)*cov_f(b,b) ), None

def cov(a, b):
  # cov(a,b) = E[ab]-E[a]*E[b]
  N = np.uint64(a.size)
  N2 = N*N
  Sab = np.sum(np.multiply(a, b, dtype=np.uint32), dtype=np.uint64)
  Sa = np.sum(a, dtype=np.uint64)
  Sb = np.sum(b, dtype=np.uint64)
  SaSb = Sa*Sb
  
  # return Sab/N - SaSb/N2
  # avoid summing small floaring point numbers with big ones: calculate integer part separately from fractional part:  a/b = a//b + (a%b)/b
  SabN = Sab//N
  SaSbN2 = SaSb//N2
  if SabN>SaSbN2:
    return np.float64(SabN - SaSbN2)  + ( np.float64(Sab%N)/N - np.float64(SaSb%N2)/N2 ) 
  else:
    return -np.float64(SaSbN2 - SabN) + ( np.float64(Sab%N)/N - np.float64(SaSb%N2)/N2 )

# implemented with integer-only arithmetic for better numerical stability and reproducibility
def pearsonr(a,b):
  return cov(a,b)/np.sqrt( cov(a,a)*cov(b,b) ), None

def calculate_psnr_f(img1, img2, max_value=255):
    mse = np.mean( (np.array(img1, dtype=np.float32) - np.array(img2, dtype=np.float32)) ** 2)
    if mse == 0: return 1000
    return 20 * np.log10( max_value / (np.sqrt(mse)))

# implemented with integer-only arithmetic for better numerical stability and reproducibility
def calculate_psnr(a, b, max_value=255):
    # SUM( (a-b)**2 ) = SUM(a**2) -2*SUM(a*b) + SUM(b*2)
    N = np.uint64(a.size)
    Sab = np.sum(np.multiply(a, b, dtype=np.uint32), dtype=np.uint64)
    Saa = np.sum(np.multiply(a, a, dtype=np.uint32), dtype=np.uint64)
    Sbb = np.sum(np.multiply(b, b, dtype=np.uint32), dtype=np.uint64)
    S = Saa + Sbb - 2*Sab
    #mse = S/N
    # avoid summing small floaring point numbers with big ones: calculate integer part separately from fractional part:  a/b = a//b + (a%b)/b
    mse = np.float64(S//N) + np.float64(S%N)/N  
    if mse == 0: return 1000
    return 20 * np.log10( max_value // np.sqrt(mse))

def calculate_chroma_similarity(num_frames, input1_fname, input2_fname, width, height, chroma_format, bit_depth):

  y_size, uv_size, uv_shape, bytes_per_pixel, frame_size_bytes = determine_format_characteristics(width, height, chroma_format, bit_depth)
        
  input1_file = open(input1_fname, 'rb')
  input2_file = open(input2_fname, 'rb')
  
  similarities = []
  
  frame_index = 0
  while True:
    # read Y, U, V
    y_buffer1 = input1_file.read(y_size * bytes_per_pixel)
    if len(y_buffer1) == 0:  # check if at the end of the file
      break
    y_buffer2 = input2_file.read(y_size * bytes_per_pixel)
    if len(y_buffer2) == 0:  # check if at the end of the file
      break
    
    uv1 = np.frombuffer(input1_file.read(uv_size * bytes_per_pixel * 2), dtype=np.uint16 if bytes_per_pixel == 2 else np.uint8)
    uv2 = np.frombuffer(input2_file.read(uv_size * bytes_per_pixel * 2), dtype=np.uint16 if bytes_per_pixel == 2 else np.uint8)
    
    correlation, _ = pearsonr(uv1, uv2)
    psnr = calculate_psnr(uv1, uv2, (1<<bit_depth)-1 )
    if psnr<27.5: correlation = 0
    similarities.append( correlation )

    frame_index += 1
    if (num_frames>0) and (frame_index>=num_frames): break
    
  input1_file.close()
  input2_file.close()
  return similarities

