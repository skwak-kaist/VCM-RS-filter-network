#include <sadl/model.h>
#include <iostream>
#include <fstream>

using namespace std;

template<typename T> bool copy(const sadl::layers::Layer<float> &layer, sadl::layers::Layer<T> &layerQ)
{
  // from loadPrefix
  layerQ.m_name      = layer.m_name;
  layerQ.m_inputs_id = layer.m_inputs_id;
  // WARNING: SHOULD BE SYNC BY HAND WITH NEW LAYERS
  // IF LOADINTERNAL IMPLEMENTED FOR A LAYER
  switch (layerQ.op())
  {
  case sadl::layers::OperationType::Add:
    break;
  case sadl::layers::OperationType::BiasAdd:
    break;
  case sadl::layers::OperationType::Concat:
    break;
  case sadl::layers::OperationType::Const:
    layerQ.m_out.resize(layer.m_out.dims());
    for (int k = 0; k < layer.m_out.size(); ++k)
      layerQ.m_out[k] = static_cast<T>(layer.m_out[k]);
    break;
  case sadl::layers::OperationType::Conv2D:
    dynamic_cast<sadl::layers::Conv2D<T> &>(layerQ).m_strides = dynamic_cast<const sadl::layers::Conv2D<float> &>(layer).m_strides;
    dynamic_cast<sadl::layers::Conv2D<T> &>(layerQ).m_pads    = dynamic_cast<const sadl::layers::Conv2D<float> &>(layer).m_pads;
    dynamic_cast<sadl::layers::Conv2D<T> &>(layerQ).m_groups  = dynamic_cast<const sadl::layers::Conv2D<float> &>(layer).m_groups;
    break;
  case sadl::layers::OperationType::Conv2DTranspose:
    dynamic_cast<sadl::layers::Conv2DTranspose<T> &>(layerQ).m_strides  = dynamic_cast<const sadl::layers::Conv2DTranspose<float> &>(layer).m_strides;
    dynamic_cast<sadl::layers::Conv2DTranspose<T> &>(layerQ).m_pads     = dynamic_cast<const sadl::layers::Conv2DTranspose<float> &>(layer).m_pads;
    dynamic_cast<sadl::layers::Conv2DTranspose<T> &>(layerQ).m_out_pads = dynamic_cast<const sadl::layers::Conv2DTranspose<float> &>(layer).m_out_pads;
    break;
  case sadl::layers::OperationType::Copy:
    break;
  case sadl::layers::OperationType::Identity:
    break;
  case sadl::layers::OperationType::LeakyRelu:
    break;
  case sadl::layers::OperationType::MatMul:
    break;
  case sadl::layers::OperationType::MaxPool:
    dynamic_cast<sadl::layers::MaxPool<T> &>(layerQ).m_kernel  = dynamic_cast<const sadl::layers::MaxPool<float> &>(layer).m_kernel;
    dynamic_cast<sadl::layers::MaxPool<T> &>(layerQ).m_strides = dynamic_cast<const sadl::layers::MaxPool<float> &>(layer).m_strides;
    dynamic_cast<sadl::layers::MaxPool<T> &>(layerQ).m_pads    = dynamic_cast<const sadl::layers::MaxPool<float> &>(layer).m_pads;
    break;
  case sadl::layers::OperationType::Maximum:
    break;
  case sadl::layers::OperationType::Mul:
    break;
  case sadl::layers::OperationType::Placeholder: /* do not copy q */;
    break;
  case sadl::layers::OperationType::Relu:
    break;
  case sadl::layers::OperationType::Reshape:
    break;
  case sadl::layers::OperationType::OperationTypeCount:
    break;
  case sadl::layers::OperationType::Transpose:
    dynamic_cast<sadl::layers::Transpose<T> &>(layerQ).m_perm = dynamic_cast<const sadl::layers::Transpose<float> &>(layer).m_perm;
    break;
  case sadl::layers::OperationType::Flatten:
    dynamic_cast<sadl::layers::Flatten<T> &>(layerQ).m_axis = dynamic_cast<const sadl::layers::Flatten<float> &>(layer).m_axis;
    dynamic_cast<sadl::layers::Flatten<T> &>(layerQ).m_dim  = dynamic_cast<const sadl::layers::Flatten<float> &>(layer).m_dim;
    break;
  case sadl::layers::OperationType::Shape:
    break;
  case sadl::layers::OperationType::Expand:
    break;
  case sadl::layers::OperationType::Slice:
    dynamic_cast<sadl::layers::Slice<T> &>(layerQ).m_start_h = dynamic_cast<const sadl::layers::Slice<float> &>(layer).m_start_h;
    dynamic_cast<sadl::layers::Slice<T> &>(layerQ).m_end_h   = dynamic_cast<const sadl::layers::Slice<float> &>(layer).m_end_h;
    dynamic_cast<sadl::layers::Slice<T> &>(layerQ).m_start_w = dynamic_cast<const sadl::layers::Slice<float> &>(layer).m_start_w;
    dynamic_cast<sadl::layers::Slice<T> &>(layerQ).m_end_w   = dynamic_cast<const sadl::layers::Slice<float> &>(layer).m_end_w;
    dynamic_cast<sadl::layers::Slice<T> &>(layerQ).m_start_c = dynamic_cast<const sadl::layers::Slice<float> &>(layer).m_start_c;
    dynamic_cast<sadl::layers::Slice<T> &>(layerQ).m_end_c   = dynamic_cast<const sadl::layers::Slice<float> &>(layer).m_end_c;
    break;
  case sadl::layers::OperationType::PReLU:
    break;
  case sadl::layers::OperationType::ScatterND:
    break;
  case sadl::layers::OperationType::GridSample:
    dynamic_cast<sadl::layers::GridSample<T> &>(layerQ).m_align_corners = dynamic_cast<const sadl::layers::GridSample<float> &>(layer).m_align_corners;
    dynamic_cast<sadl::layers::GridSample<T> &>(layerQ).m_mode          = dynamic_cast<const sadl::layers::GridSample<float> &>(layer).m_mode;
    dynamic_cast<sadl::layers::GridSample<T> &>(layerQ).m_padding_mode  = dynamic_cast<const sadl::layers::GridSample<float> &>(layer).m_padding_mode;
    break;
  case sadl::layers::OperationType::Resize:
    dynamic_cast<sadl::layers::Resize<T> &>(layerQ).m_input_label = dynamic_cast<const sadl::layers::Resize<float> &>(layer).m_input_label;
    dynamic_cast<sadl::layers::Resize<T> &>(layerQ).m_coordinate_transformation_mode =
      dynamic_cast<const sadl::layers::Resize<float> &>(layer).m_coordinate_transformation_mode;
    dynamic_cast<sadl::layers::Resize<T> &>(layerQ).m_mode            = dynamic_cast<const sadl::layers::Resize<float> &>(layer).m_mode;
    dynamic_cast<sadl::layers::Resize<T> &>(layerQ).m_nearest_mode    = dynamic_cast<const sadl::layers::Resize<float> &>(layer).m_nearest_mode;
    break;
  case sadl::layers::OperationType::Compare:
    dynamic_cast<sadl::layers::Compare<T> &>(layerQ).m_mode            = dynamic_cast<const sadl::layers::Compare<float> &>(layer).m_mode;
    break;
  case sadl::layers::OperationType::Where:
    break;
  case sadl::layers::OperationType::Minimum:
    break;
    // no default to get warning
  }

  return true;
}

template<typename T> bool copy(const sadl::Model<float> &model, sadl::Model<T> &modelQ)
{
  modelQ.m_version = sadl::Version::sadl03;
  modelQ.m_data.clear();
  modelQ.m_data.resize(model.m_data.size());
  modelQ.m_ids_input  = model.m_ids_input;
  modelQ.m_ids_output = model.m_ids_output;
  int nb_layers       = (int) modelQ.m_data.size();
  for (int k = 0; k < nb_layers; ++k)
  {
    modelQ.m_data[k].layer = sadl::createLayer<T>(model.m_data[k].layer->id(), model.m_data[k].layer->op());
    modelQ.m_data[k].inputs.clear();
    copy(*model.m_data[k].layer, *modelQ.m_data[k].layer);
  }
  return true;
}
