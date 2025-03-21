/* The copyright in this software is being made available under the BSD
 * License, included below. This software may be subject to other third party
 * and contributor rights, including patent rights, and no such rights are
 * granted under this license.
 *
 * Copyright (c) 2010-2024, ITU/ISO/IEC
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *  * Neither the name of the ITU/ISO/IEC nor the names of its contributors may
 *    be used to endorse or promote products derived from this software without
 *    specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS
 * BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
 * THE POSSIBILITY OF SUCH DAMAGE.
 */
#pragma once
#include "layer.h"

namespace sadl
{
namespace layers
{
template<typename T> class Reshape : public Layer<T>
{
public:
  using Layer<T>::Layer;
  using Layer<T>::m_out;   // to avoid this->
  using Layer<T>::m_initDone;

  virtual bool apply(std::vector<Tensor<T> *> &in) override;
  virtual bool init(const std::vector<Tensor<T> *> &in) override;
  virtual bool mutateInput() const override { return true; }

protected:
  virtual bool loadInternal(std::istream &file, Version v) override;
};

// assume data in in[0] and shape in in[1]
template<typename T> bool Reshape<T>::apply(std::vector<Tensor<T> *> &in)
{
  assert(in.size() == 2);
  // second layer is reshape prms
  // assert(in[1]->dims().size()==2);
  assert(in[0]->size() == m_out.size());
  // resize done at init
  swapData(*in[0], m_out);

  return true;
}

template<typename T> bool Reshape<T>::init(const std::vector<Tensor<T> *> &in)
{
  if (in.size() != 2)
    return false;
  SADL_DBG(std::cout << "  - " << in[0]->dims() << ' ' << in[1]->dims() << std::endl);
  // second layer is always reshape prms: value as int inside the tensor
  if (in[1]->dims().size() != 1)
    return false;
  Dimensions dim;
  dim.resize((int) in[1]->size());
  if (!std::is_same<float, T>::value && in[1]->quantizer != 0)
  {
    std::cerr << "[ERROR] quantizer on reshape dimensions data layer" << std::endl;
    return false;
  }
  int cnt=0;
  for (int k = 0; k < in[1]->size(); ++k)
  {
    if ((*in[1]) (k) == -1) ++cnt;
  }
  if (cnt>=2) // to be  checked
  {
    std::cerr << "[ERROR] more than one -1" << std::endl;
    return false;
  }
  int pos=0;  
  for (int k = 0; k < in[1]->size(); ++k)
  {
    if ((*in[1]) (k) == -1)
    { 
      dim[k] = 1; 
      pos = k;
    }
    else
    {
      dim[k] = (int) ((*in[1]) (k));
    }
  }
  //
  if (cnt==1) {
    assert(dim.nbElements()!=0);
    dim[pos]=(int)(in[0]->dims().nbElements()/dim.nbElements());
  }
  if (dim.nbElements() != in[0]->dims().nbElements())
  {
    std::cerr << "[ERROR] reshape incompatible sizes " << dim << ' ' << in[0]->dims() << std::endl;
    std::cerr << "[ERROR] ";
    for (int k = 0; k < in[1]->dims()[0]; ++k)
      std::cerr << (*in[1]) (k) << ' ';
    std::cerr << std::endl;

    return false;
  }
  SADL_DBG(std::cout << "  - new shape: " << dim << std::endl);
  m_out.resize(dim);
  m_initDone = true;
  return true;
}

template<typename T> bool Reshape<T>::loadInternal(std::istream &, Version) { return true; }

}   // namespace layers
}   // namespace sadl
