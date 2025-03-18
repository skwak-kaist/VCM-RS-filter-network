#include <sadl/model.h>

template<typename T> bool sadl::layers::Conv2D<T>::dump(std::ostream &file)
{
  int32_t x = m_strides.size();
  file.write((const char *) &x, sizeof(int32_t));
  file.write((const char *) m_strides.begin(), m_strides.size() * sizeof(int32_t));
  x = m_pads.size();
  file.write((const char *) &x, sizeof(int32_t));
  file.write((const char *) m_pads.begin(), m_pads.size() * sizeof(int32_t));
  file.write((const char *) &m_groups, sizeof(m_groups));
  file.write((const char *) &m_q, sizeof(m_q));
  return true;
}

template<typename T> bool sadl::layers::Conv2DTranspose<T>::dump(std::ostream &file)
{
  int32_t x = m_strides.size();
  file.write((const char *) &x, sizeof(int32_t));
  file.write((const char *) m_strides.begin(), m_strides.size() * sizeof(int32_t));
  x = m_pads.size();
  file.write((const char *) &x, sizeof(int32_t));
  file.write((const char *) m_pads.begin(), m_pads.size() * sizeof(int32_t));
  x = m_out_pads.size();
  file.write((const char *) &x, sizeof(int32_t));
  file.write((const char *) m_out_pads.begin(), m_out_pads.size() * sizeof(int32_t));
  file.write((const char *) &m_q, sizeof(m_q));
  return true;
}

template<typename T> bool sadl::layers::Slice<T>::dump(std::ostream &file)
{
  file.write((const char *) &m_start_h, sizeof(m_start_h));
  file.write((const char *) &m_end_h, sizeof(m_end_h));
  file.write((const char *) &m_start_w, sizeof(m_start_w));
  file.write((const char *) &m_end_w, sizeof(m_end_w));
  file.write((const char *) &m_start_c, sizeof(m_start_c));
  file.write((const char *) &m_end_c, sizeof(m_end_c));
  return true;
}

template<typename T> bool sadl::layers::MatMul<T>::dump(std::ostream &file)
{
  file.write((const char *) &m_q, sizeof(m_q));
  return true;
}

template<typename T> bool sadl::layers::Mul<T>::dump(std::ostream &file)
{
  file.write((const char *) &m_q, sizeof(m_q));
  return true;
}

template<typename T> bool sadl::layers::Placeholder<T>::dump(std::ostream &file)
{
  int32_t x = m_dims.size();
  file.write((const char *) &x, sizeof(x));
  file.write((const char *) m_dims.begin(), sizeof(int) * x);
  file.write((const char *) &m_q, sizeof(m_q));
  return true;
}

template<typename T> bool sadl::layers::MaxPool<T>::dump(std::ostream &file)
{
  int32_t x = m_strides.size();
  file.write((const char *) &x, sizeof(int32_t));
  file.write((const char *) m_strides.begin(), m_strides.size() * sizeof(int32_t));
  x = m_kernel.size();
  file.write((const char *) &x, sizeof(int32_t));
  file.write((const char *) m_kernel.begin(), m_kernel.size() * sizeof(int32_t));
  x = m_pads.size();
  file.write((const char *) &x, sizeof(int32_t));
  file.write((const char *) m_pads.begin(), m_pads.size() * sizeof(int32_t));
  return true;
}

template<typename T> bool sadl::layers::Flatten<T>::dump(std::ostream &file)
{
  int32_t x = m_axis;
  file.write((const char *) &x, sizeof(int32_t));
  return true;
}

template<typename T> bool sadl::layers::Const<T>::dump(std::ostream &file)
{
  // load values
  int32_t x = m_out.dims().size();
  file.write((const char *) &x, sizeof(x));
  file.write((const char *) m_out.dims().begin(), x * sizeof(int));
  if (std::is_same<T, int16_t>::value)
  {
    x = TensorInternalType::Int16;
  }
  else if (std::is_same<T, int32_t>::value)
  {
    x = TensorInternalType::Int32;
  }
  else if (std::is_same<T, float>::value)
  {
    x = TensorInternalType::Float;
  }
  else
  {
    std::cerr << "[ERROR] to do" << std::endl;
    exit(-1);
  }
  file.write((const char *) &x, sizeof(x));

  if (!std::is_same<T, float>::value)
    file.write((const char *) &m_out.quantizer, sizeof(m_out.quantizer));
  file.write((const char *) m_out.data(), m_out.size() * sizeof(T));
  return true;
}

template<typename T> bool sadl::layers::GridSample<T>::dump(std::ostream &file)
{
  file.write((const char *) &m_align_corners, sizeof(m_align_corners));
  file.write((const char *) &m_mode, sizeof(m_mode));
  file.write((const char *) &m_padding_mode, sizeof(m_padding_mode));
  return true;
}

template<typename T> bool sadl::layers::Resize<T>::dump(std::ostream &file)
{
  file.write((const char *) &m_input_label, sizeof(m_input_label));
  file.write((const char *) &m_coordinate_transformation_mode, sizeof(m_coordinate_transformation_mode));
  file.write((const char *) &m_mode, sizeof(m_mode));
  file.write((const char *) &m_nearest_mode, sizeof(m_nearest_mode));
  return true;
}

template<typename T> bool sadl::layers::Compare<T>::dump(std::ostream &file)
{
  file.write((const char *) &m_mode, sizeof(m_mode));
  return true;
}

template<typename T> bool sadl::layers::Where<T>::dump(std::ostream &file)
{
  return true;
}

template<typename T> bool sadl::layers::Layer<T>::dump(std::ostream &file)
{
  // std::cout<<"todo? "<<opName(op_)<<std::endl;
  return true;
}

template<typename T> bool sadl::Model<T>::dump(std::ostream &file)
{
  if (!file)
  {
    std::cerr << "The file is not open." << std::endl;
    return false;
  }

  char magic[9] = "SADL0004";
  file.write(magic, 8);
  int32_t x = 0;
  if (std::is_same<T, float>::value)
    x = layers::TensorInternalType::Float;
  else if (std::is_same<T, int32_t>::value)
    x = layers::TensorInternalType::Int32;
  else if (std::is_same<T, int16_t>::value)
    x = layers::TensorInternalType::Int16;
  else
  {
    std::cerr << "[ERROR] to do Model::dump" << std::endl;
    exit(-1);
  }
  file.write((const char *) &x, sizeof(int32_t));

  int32_t nb_layers = (int) m_data.size();
  file.write((const char *) &nb_layers, sizeof(int32_t));
  int32_t nb = (int) m_ids_input.size();
  file.write((const char *) &nb, sizeof(int32_t));
  file.write((const char *) m_ids_input.data(), sizeof(int32_t) * nb);
  nb = (int) m_ids_output.size();
  file.write((const char *) &nb, sizeof(int32_t));
  file.write((const char *) m_ids_output.data(), sizeof(int32_t) * nb);

  for (int k = 0; k < nb_layers; ++k)
  {
    // save header
    int32_t x = m_data[k].layer->id();
    file.write((const char *) &x, sizeof(int32_t));
    x = m_data[k].layer->op();
    file.write((const char *) &x, sizeof(int32_t));
    // savePrefix
    int32_t L = (int) m_data[k].layer->m_name.size();
    file.write((const char *) &L, sizeof(int32_t));
    file.write((const char *) m_data[k].layer->m_name.c_str(), m_data[k].layer->m_name.size());
    L = (int) m_data[k].layer->m_inputs_id.size();
    file.write((const char *) &L, sizeof(int32_t));
    file.write((const char *) m_data[k].layer->m_inputs_id.data(), m_data[k].layer->m_inputs_id.size() * sizeof(int32_t));
    m_data[k].layer->dump(file);
  }
  return true;
}
