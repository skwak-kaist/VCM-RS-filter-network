/* The copyright in this software is being made available under the BSD
 * License, included below. This software may be subject to other third party
 * and contributor rights, including patent rights, and no such rights are
 * granted under this license.
 *
 * Copyright (c) 2010-2023, ITU/ISO/IEC
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
template<typename T> class Mul : public Layer<T>
{
public:
  using Layer<T>::Layer;
  using Layer<T>::m_out;   // to avoid this->
  using Layer<T>::m_initDone;

  virtual bool apply(std::vector<Tensor<T> *> &in) override;
  virtual bool init(const std::vector<Tensor<T> *> &in) override;
  virtual bool mutateInput() const override { return true; }

protected:
  virtual bool          loadInternal(std::istream &file, Version v) override;
  int                   m_q = 0;
  template<int NN> bool apply_same_dim(std::vector<Tensor<T> *> &in);
  template<int NN> bool apply_singleton(std::vector<Tensor<T> *> &in);
  template<int NN> bool apply_dim2(std::vector<Tensor<T> *> &in);
  template<int NN> bool apply_dim3(std::vector<Tensor<T> *> &in);
  template<int NN> bool apply_dim4(std::vector<Tensor<T> *> &in);
#if __AVX2__
  bool apply_singleton_simd8(std::vector<Tensor<T> *> &in);
  bool apply_singleton_simd16(std::vector<Tensor<T> *> &in);
#endif
  DUMP_MODEL_EXT;
};

template<typename T> bool Mul<T>::apply(std::vector<Tensor<T> *> &in)
{
  assert(in.size() == 2);
  if (in[0] == in[1])
  {
    std::cerr << "  input aliasing" << std::endl;
    return false;
  }
  swap(*in[0], m_out);
  m_out.border_skip = std::max(m_out.border_skip, in[1]->border_skip);

  m_out.quantizer -= m_q;   // q0-q
  assert(m_out.quantizer >= 0);
  assert(in[1]->quantizer + m_q >= 0);

  const int last = in[0]->dims().back();
  if (last % 16 == 0)
  {
    constexpr int NN = 16;
    if (in[0]->dims() == in[1]->dims())
    {   // product wise
      return apply_same_dim<NN>(in);
    }
    else if (in[1]->size() == 1)
    {   // broadcast single element
#if __AVX2__
      return apply_singleton_simd16(in);
#endif
      return apply_singleton<NN>(in);
    }
    else if (in[0]->dims().size() == 2)
    {
      return apply_dim2<NN>(in);
    }
    else if (in[0]->dims().size() == 3)
    {
      return apply_dim3<NN>(in);
    }
    else if (in[0]->dims().size() == 4)
    {
      return apply_dim4<NN>(in);
    }
  }
  else if (last % 8 == 0)
  {
    constexpr int NN = 8;
    if (in[0]->dims() == in[1]->dims())
    {   // product wise
      return apply_same_dim<NN>(in);
    }
    else if (in[1]->size() == 1)
    {   // broadcast single element
#if __AVX2__
      return apply_singleton_simd8(in);
#endif
      return apply_singleton<NN>(in);
    }
    else if (in[0]->dims().size() == 2)
    {
      return apply_dim2<NN>(in);
    }
    else if (in[0]->dims().size() == 3)
    {
      return apply_dim3<NN>(in);
    }
    else if (in[0]->dims().size() == 4)
    {
      return apply_dim4<NN>(in);
    }
  }
  else
  {
    constexpr int NN = 1;
    if (in[0]->dims() == in[1]->dims())
    {   // product wise
      return apply_same_dim<NN>(in);
    }
    else if (in[1]->size() == 1)
    {   // broadcast single element
      return apply_singleton<NN>(in);
    }
    else if (in[0]->dims().size() == 2)
    {
      return apply_dim2<NN>(in);
    }
    else if (in[0]->dims().size() == 3)
    {
      return apply_dim3<NN>(in);
    }
    else if (in[0]->dims().size() == 4)
    {
      return apply_dim4<NN>(in);
    }
  }

  return false;
}

template<typename T> template<int NN> bool Mul<T>::apply_same_dim(std::vector<Tensor<T> *> &in)
{
  const int shift = in[1]->quantizer + m_q;
#if __AVX2__ && DEBUG_SIMD
  std::cout << "\n[WARN] generic version mul sameDim (but likely vectorized) " << in[0]->dims() << ' ' << in[1]->dims() << " "
            << in[0]->dims().nbElements() / 1000 << " kMAC" << std::endl;
#endif   // SIMD
  const auto &B = *in[1];
  const auto  N = (m_out.size() / NN) * NN;
  for (int k = 0; k < N; ++k)
  {
    typename ComputationType<T>::type x = m_out[k];
    x *= B[k];
    COUNTERS_MAC(B[k]);
    ComputationType<T>::quantize(x, shift);
    COUNTERS(x);
    SATURATE(x);
    m_out[k] = (T) x;
  }
  return true;
}

template<typename T> template<int NN> bool Mul<T>::apply_singleton(std::vector<Tensor<T> *> &in)
{
  const int        shift = in[1]->quantizer + m_q;
  const Tensor<T> &B     = *in[1];
#if __AVX2__ && DEBUG_SIMD
  std::cout << "[WARN] generic version mul singleton (but likely vectorized) " << in[0]->dims() << ' ' << in[1]->dims() << std::endl;
#endif   // SIMD
  const T    value{ B[0] };
  const auto N = (m_out.size() / NN) * NN;
  for (int k = 0; k < N; ++k)
  {
    typename ComputationType<T>::type x = m_out[k];
    x *= value;
    COUNTERS_MAC(value);
    ComputationType<T>::quantize(x, shift);
    COUNTERS(x);
    SATURATE(x);
    m_out[k] = (T) x;
  }
  return true;
}

template<typename T> template<int NN> bool Mul<T>::apply_dim2(std::vector<Tensor<T> *> &in)
{
  const int shift = in[1]->quantizer + m_q;

#if __AVX2__ && DEBUG_SIMD
  std::cout << "[WARN] generic version mul singleton (but likely vectorized) " << in[0]->dims() << ' ' << in[1]->dims() << std::endl;
#endif   // SIMD

  const Tensor<T> &B = *in[1];
  const int        N = in[0]->dims()[0];
  const int        H = (in[0]->dims()[1] / NN) * NN;
  for (int n = 0; n < N; ++n)
    for (int i = 0; i < H; ++i)
    {
      typename ComputationType<T>::type x = m_out(n, i);
      x *= B[i];
      COUNTERS_MAC(B[i]);
      ComputationType<T>::quantize(x, shift);
      COUNTERS(x);
      SATURATE(x);
      m_out(n, i) = (T) x;
    }
  return true;
}

template<typename T> template<int NN> bool Mul<T>::apply_dim3(std::vector<Tensor<T> *> &in)
{
  const int shift = in[1]->quantizer + m_q;

#if __AVX2__ && DEBUG_SIMD
  std::cout << "[WARN] generic version mul singleton " << in[0]->dims() << ' ' << in[1]->dims() << std::endl;
#endif   // SIMD

  const Tensor<T> &B = *in[1];
  const int        N = in[0]->dims()[0];
  const int        H = in[0]->dims()[1];
  const int        W = (in[0]->dims()[2] / NN) * NN;
  for (int n = 0; n < N; ++n)
    for (int i = 0; i < H; ++i)
      for (int j = 0; j < W; ++j)
      {
        typename ComputationType<T>::type x = m_out(n, i, j);
        x *= B[j];
        COUNTERS_MAC(B[j]);
        ComputationType<T>::quantize(x, shift);
        COUNTERS(x);
        SATURATE(x);
        m_out(n, i, j) = (T) x;
      }
  return true;
}

template<typename T> template<int NN> bool Mul<T>::apply_dim4(std::vector<Tensor<T> *> &in)
{
  const int shift = in[1]->quantizer + m_q;

#if __AVX2__ && DEBUG_SIMD
  std::cout << "[WARN] generic version mul singleton" << in[0]->dims() << ' ' << in[1]->dims() << std::endl;
#endif   // SIMD
  assert(in[0]->dims()[0] == 1);

  const Tensor<T> &B = *in[1];
  const int        N = in[0]->dims()[0];
  const int        H = in[0]->dims()[1];
  const int        W = in[0]->dims()[2];
  const int        K = (in[0]->dims()[3] / NN) * NN;
  for (int n = 0; n < N; ++n)
    for (int i = 0; i < H; ++i)
      for (int j = 0; j < W; ++j)
        for (int k = 0; k < K; ++k)
        {
          typename ComputationType<T>::type x = m_out(n, i, j, k);
          x *= B[k];
          COUNTERS_MAC(B[k]);
          ComputationType<T>::quantize(x, shift);
          COUNTERS(x);
          SATURATE(x);
          m_out(n, i, j, k) = (T) x;
        }

  return true;
}

#if __AVX2__
template<> inline bool Mul<float>::apply_singleton_simd8(std::vector<Tensor<float> *> &in)
{
  using T                = float;
  const Tensor<T> &B     = *in[1];
  const __m256     value = _mm256_set1_ps(B[0]);
  for (int k = 0; k < m_out.size(); k += 8)
  {
    float *aptr = m_out.data() + k;
    __m256 a    = _mm256_load_ps(aptr);
    __m256 v    = _mm256_mul_ps(a, value);
    _mm256_store_ps(aptr, v);
  }
  return true;
}
template<> inline bool Mul<float>::apply_singleton_simd16(std::vector<Tensor<float> *> &in) { return apply_singleton_simd8(in); }

template<typename T> bool Mul<T>::apply_singleton_simd8(std::vector<Tensor<T> *> &in) { return apply_singleton<8>(in); }

template<typename T> bool Mul<T>::apply_singleton_simd16(std::vector<Tensor<T> *> &in) { return apply_singleton<16>(in); }
#endif
// data in in[0]
// bias in in[1]
// assume data shape [N,W,H,D]
// assume bias shape [D]
template<typename T> bool Mul<T>::init(const std::vector<Tensor<T> *> &in)
{
  SADL_DBG(std::cout << "  - " << in[0]->dims() << ' ' << in[1]->dims() << std::endl);
  if (in.size() != 2)
    return false;

  // cases:
  // same dim: element wise
  // if B as only one element-> bradcast to all A element
  // B has dim [n] or [1,n] and A[...,n]
  /*
  If the bias a single dimension dimension and it
  is not a singleton, the last dimension of the input
  tensor has to be equal to the bias dimension.
  */

  if (in[1]->size() == 1)
  {
    // ok
  }
  else if (in[1]->dims().size() == 1 || (in[1]->dims().size() == 2 && in[1]->dims()[0] == 1))
  {
    if (in[0]->dims().back() != in[1]->dims().back())
      return false;
  }
  else if (in[0]->dims().size() >= 2 && in[1]->size() == in[0]->dims().back())
  {
  }
  else
  {
    if (!(in[0]->dims() == in[1]->dims()))
      return false;
  }
  m_out.resize(in[0]->dims());
  m_initDone = true;
  return true;
}

template<typename T> bool Mul<T>::loadInternal(std::istream &file, Version v)
{
  file.read((char *) &m_q, sizeof(m_q));
  SADL_DBG(std::cout << "  - q: " << m_q << std::endl);

  return true;
}

}   // namespace layers
}   // namespace sadl
