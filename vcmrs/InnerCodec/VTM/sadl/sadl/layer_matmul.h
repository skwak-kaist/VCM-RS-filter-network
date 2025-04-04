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
#if __AVX2__ || __SSE4_2__
#include <immintrin.h>
#endif
#include "simd_utils.h"

namespace sadl
{
namespace layers
{
template<typename T> class MatMul : public Layer<T>
{
public:
  using Layer<T>::Layer;
  using Layer<T>::m_out;   // to avoid this->
  using Layer<T>::m_initDone;

  virtual bool apply(std::vector<Tensor<T> *> &in) override;
  virtual bool init(const std::vector<Tensor<T> *> &in) override;

protected:
  virtual bool          loadInternal(std::istream &file, Version v) override;
  template<int NN> bool apply_dim2(std::vector<Tensor<T> *> &in);
  template<int NN> bool apply_dim3(std::vector<Tensor<T> *> &in);
  template<int NN> bool apply_dim4(std::vector<Tensor<T> *> &in);

#if __AVX2__
  bool apply_dim2_simd8(std::vector<Tensor<T> *> &in) { return apply_dim2<8>(in); }
  bool apply_dim2_simd16(std::vector<Tensor<T> *> &in) { return apply_dim2_simd8(in); }
#endif

#if SPARSE_SUPPORT
  bool apply_sparse_matmul(std::vector<Tensor<T> *> &in);
#if __AVX2__
  bool apply_sparse_matmul_simd16(std::vector<Tensor<T> *> &in);
#endif
#if __SSE4_2__
  bool apply_sparse_matmul_simd8(std::vector<Tensor<T> *> &in);
#endif
#endif

  int m_q = 0;
  DUMP_MODEL_EXT;
};

template<typename T> bool MatMul<T>::apply(std::vector<Tensor<T> *> &in)
{
  assert(in.size() == 2);
#if __AVX2__
#define MULT8_DIM2 apply_dim2_simd8
#define MULT16_DIM2 apply_dim2_simd16
#else
#define MULT8_DIM2 apply_dim2<8>
#define MULT16_DIM2 apply_dim2<16>
#endif

#if SPARSE_SUPPORT
#if __AVX2__
#define SPARSE_MATMULT apply_sparse_matmul_simd16
#elif __SSE4_2__
#define SPARSE_MATMULT apply_sparse_matmul_simd8
#else
#define SPARSE_MATMULT apply_sparse_matmul
#endif
#endif

  const Tensor<T> &A{ *in[0] };
  const Tensor<T> &B{ *in[1] };
  m_out.quantizer = A.quantizer - m_q;
  assert(m_out.quantizer >= 0);
  assert(in[1]->quantizer + m_q >= 0);
  int dum = A.dims().size();
  // cases:
  // A: always a tensor
  // B: tensor or const
  // 1- A [x] B[x] || A [x,y] B[y,z] || A [x,y,z] B[x,z,t]
  // 2- A [1,x] B[x] || A [1,x,y] B[y,z] || A [1,x,y,z] B[x,z,t]
  if (A.dims().size() - 1 == B.dims().size())
    dum--;
  const int H{ A.dims().back() };   // to be changed if SIMD for more than dim1 and dim2
  switch (dum)
  {
  case 2:
#if SPARSE_SUPPORT
    if (in[1]->isSparse())
      return SPARSE_MATMULT(in);
    else
#endif
      if (H % 16 == 0)
      return MULT16_DIM2(in);
    else if (H % 8 == 0)
      return MULT8_DIM2(in);
    else
      return apply_dim2<1>(in);
    break;
  case 3:
    return apply_dim3<1>(in);
    break;
  case 4:
    return apply_dim4<1>(in);
    break;
  default:
    std::cerr << "Logical error MatMul::apply(std::vector<Tensor<T> *> &in)" << A.dims() << ' ' << B.dims() << std::endl;
    return false;
  }
}

#if __AVX2__
template<> inline bool MatMul<float>::apply_dim2_simd8(std::vector<Tensor<float> *> &in)
{
  using T = float;
  const Tensor<T> &A{ *in[0] };
  const Tensor<T> &B{ *in[1] };
  const int        last = A.dims().size() - 1;
  const int        N{ A.dims()[last - 1] };
  const int        H{ A.dims()[last] };
  const int        R{ B.dims()[1] };
#if DEBUG_SIMD
  if (H >= 16)
  {
    std::cout << "\n[WARN] suboptimal SIMD version matmul dim2 " << A.dims() << ' ' << B.dims() << "(H=" << H << ") " << (N * R * H) / 1000 << " kMAC"
              << std::endl;
  }
#endif
  assert(H % 8 == 0);
  constexpr int idx_start{ 0 };
  const int     idx_end{ H };
  for (int b = 0; b < N; ++b)
  {
    float *optr = m_out.data() + R * b;
    for (int t = 0; t < R; ++t)
    {
      __m256 s = _mm256_setzero_ps();
      {
        const float *aptr = A.data() + b * H + idx_start;
        const float *bptr = B.data() + t * H + idx_start;   // T * i + t  (i, t); => B[t*H+i] if transposed
        for (int i = idx_start; i < idx_end; i += 8, aptr += 8, bptr += 8)
        {
          __m256 a = _mm256_load_ps(aptr);
          __m256 b = _mm256_load_ps(bptr);
#if __FMA__
          s = _mm256_fmadd_ps(a, b, s);
#else
          s = _mm256_add_ps(s, _mm256_mul_ps(a, b));
#endif
        }
      }
      optr[t] = sum8_float(s);
    }
  }
  return true;
}

#if __AVX512F__
template<> inline bool MatMul<float>::apply_dim2_simd16(std::vector<Tensor<float> *> &in)
{
  const Tensor<float> &A{ *in[0] };
  const Tensor<float> &B{ *in[1] };
  const int            last = A.dims().size() - 1;
  const int            N{ A.dims()[last - 1] };
  const int            H{ A.dims()[last] };
  const int            R{ B.dims()[1] };
  assert(H % 16 == 0);
  constexpr int idx_start{ 0 };
  const int     idx_end{ H };
  for (int b = 0; b < N; ++b)
  {
    float *optr = m_out.data() + R * b;
    for (int t = 0; t < R; ++t)
    {
      __m512 s = _mm512_setzero_ps();
      {
        const float *aptr = A.data() + b * H + idx_start;
        const float *bptr = B.data() + t * H + idx_start;
        for (int i = idx_start; i < idx_end; i += 16, aptr += 16, bptr += 16)
        {
          __m512 a = _mm512_load_ps(aptr);
          __m512 b = _mm512_load_ps(bptr);
#if __FMA__
          s = _mm512_fmadd_ps(a, b, s);
#else
          s = _mm512_add_ps(s, _mm512_mul_ps(a, b));
#endif
        }
      }
      optr[t] = sum16_float(s);
    }
  }
  return true;
}
#endif
#endif

template<typename T> template<int NN> bool MatMul<T>::apply_dim2(std::vector<Tensor<T> *> &in)
{
  const Tensor<T> &A{ *in[0] };
  const Tensor<T> &B{ *in[1] };
  const int        shift{ in[1]->quantizer + m_q };
  const int        last = A.dims().size() - 1;
  const int        N{ A.dims()[last - 1] };
  const int        H{ (A.dims()[last] / NN) * NN };
  const int        R{ B.dims().back() };
#if __AVX2__ && DEBUG_SIMD
  std::cout << "\n[WARN] generic version matmul dim2 " << A.dims() << ' ' << B.dims() << "(H=" << H << ") " << (N * R * H) / 1000 << " kMAC" << std::endl;
#endif   // SIMD
  constexpr int idx_start{ 0 };
  const int     idx_end{ H };
  if (A.dims().size() == 2)
  {
    for (int b = 0; b < N; ++b)
    {
      const T *aptr = A.data() + H * b;   // A(b,i)   => A[H*b]
      for (int t = 0; t < R; ++t)
      {
        typename ComputationType<T>::type x    = 0;
        const T *                         bptr = B.data() + t * H;   // T * i + t  (i, t); => B[t*H+i] if transposed
        {
          for (int i = idx_start; i < idx_end; ++i)
          {
            x += (typename ComputationType<T>::type) aptr[i] * bptr[i];   // A(b,i)*B(i, t);
            COUNTERS_MAC(bptr[i]);
          }
        }
        ComputationType<T>::quantize(x, shift);
        COUNTERS(x);
        SATURATE(x);
        m_out(b, t) = (T) x;
      }
    }
  }
  else
  {
    for (int b = 0; b < N; ++b)
    {
      const T *aptr = A.data() + H * b;   // A(0,b,i)  => A[H*b]
      for (int t = 0; t < R; ++t)
      {
        typename ComputationType<T>::type x    = 0;
        const T *                         bptr = B.data() + t * H;   // T * i + t  (i, t); => B[t*H+i] if transposed
        {
          for (int i = idx_start; i < idx_end; ++i)
          {
            x += (typename ComputationType<T>::type) aptr[i] * bptr[i];   // A(0,b,i)*B(i, t);
            COUNTERS_MAC(bptr[i]);
          }
        }
        ComputationType<T>::quantize(x, shift);
        COUNTERS(x);
        SATURATE(x);
        m_out(0, b, t) = (T) x;
      }
    }
  }
  return true;
}

template<typename T> template<int NN> bool MatMul<T>::apply_dim3(std::vector<Tensor<T> *> &in)
{
  const Tensor<T> &A{ *in[0] };
  const Tensor<T> &B{ *in[1] };
  const int        shift{ in[1]->quantizer + m_q };
  const int        last = A.dims().size() - 1;
  const int        N{ A.dims()[last - 2] };
  const int        H{ A.dims()[last - 1] };
  const int        R{ B.dims().back() };
  const int        W{ (A.dims()[last] / NN) * NN };
  (void) W;
#if __AVX2__ && DEBUG_SIMD
  std::cout << "\n[WARN] generic version matmul dim3 " << A.dims() << ' ' << B.dims() << "(H=" << H << ") " << (N * R * H * W) / 1000 << " kMAC" << std::endl;
#endif   // SIMD
  constexpr int idx_start{ 0 };
  const int     idx_end{ W };
  if (A.dims().size() == 3)
  {
    for (int b = 0; b < N; ++b)
    {
      for (int i = 0; i < H; ++i)
      {
        for (int t = 0; t < R; ++t)
        {
          typename ComputationType<T>::type x = 0;
          {
            for (int j = idx_start; j < idx_end; ++j)
            {
              x += (typename ComputationType<T>::type) A(b, i, j) * B(b, j, t);
              COUNTERS_MAC(B(b, j, t));
            }
          }
          ComputationType<T>::quantize(x, shift);
          COUNTERS(x);
          SATURATE(x);
          m_out(b, i, t) = (T) x;
        }
      }
    }
  }
  else
  {   // size==4
    for (int b = 0; b < N; ++b)
    {
      for (int i = 0; i < H; ++i)
      {
        for (int t = 0; t < R; ++t)
        {
          typename ComputationType<T>::type x = 0;
          {
            for (int j = idx_start; j < idx_end; ++j)
            {
              x += (typename ComputationType<T>::type) A(0, b, i, j) * B(b, j, t);
              COUNTERS_MAC(B(b, j, t));
            }
          }
          ComputationType<T>::quantize(x, shift);
          COUNTERS(x);
          SATURATE(x);
          m_out(0, b, i, t) = (T) x;
        }
      }
    }
  }
  return true;
}
template<typename T> template<int NN> bool MatMul<T>::apply_dim4(std::vector<Tensor<T> *> &in)
{
  const Tensor<T> &A{ *in[0] };
  const Tensor<T> &B{ *in[1] };
  int        shift{ in[1]->quantizer + m_q };
  const int        last = A.dims().size() - 1;
  const int        N{ A.dims()[last - 2] };
  const int        H{ A.dims()[last - 1] };
  const int        R{ B.dims().back() };
  const int        W{ (A.dims()[last] / NN) * NN };
  (void) W;
#if __AVX2__ && DEBUG_SIMD
  std::cout << "\n[WARN] generic version matmul dim4 " << A.dims() << ' ' << B.dims() << "(H=" << H << ") " << (N * R * H * W) / 1000 << " kMAC" << std::endl;
#endif   // SIMD
  constexpr int idx_start{ 0 };
  const int     idx_end{ W };

  for (int b = 0; b < N; ++b)
  {
    for (int i = 0; i < H; ++i)
    {
      for (int t = 0; t < R; ++t)
      {
        typename ComputationType<T>::type x = 0;
        {
          for (int j = idx_start; j < idx_end; ++j)
          {
            x += (typename ComputationType<T>::type) A(0, b, i, j) * B(0, b, j, t);
            COUNTERS_MAC(B(0, b, j, t));
          }
        }
        ComputationType<T>::quantize(x, shift);
        COUNTERS(x);
        SATURATE(x);
        m_out(0, b, i, t) = (T) x;
      }
    }
  }
  
  return true;
}


#if SPARSE_SUPPORT
template<typename T> bool MatMul<T>::apply_sparse_matmul(std::vector<Tensor<T> *> &in)
{
  const Tensor<T> &A{ *in[0] };
  const Tensor<T> &B{ *in[1] };
  const int        shift{ in[1]->quantizer + m_q };
  const int        last = A.dims().size() - 1;
  const int        N{ A.dims()[last - 1] };
  const int        H{ A.dims()[last] };

  m_out.fill(0);

  uint32_t offset_data = 0;
  uint16_t i           = 0;

  if (A.dims().size() == 2)
  {
    for (int b = 0; b < N; ++b)
    {
      const T *aptr = A.data() + H * b;   // A(b,i)   => A[H*b]

      for (const auto &nb_nonzero: B.getNbNonzerosCol())
      {
        typename ComputationType<T>::type x = 0;

        for (auto k = 0; k < nb_nonzero; ++k, ++offset_data)
        {
          uint16_t j = B.getIndices()[offset_data];
          x += (typename ComputationType<T>::type) aptr[j] * B.getDataSparse()[offset_data];
          COUNTERS_MAC(B.getDataSparse()[offset_data]);
        }

        ComputationType<T>::quantize(x, shift);
        COUNTERS(x);
        SATURATE(x);
        m_out(b, i) = (T) x;
        i++;
      }
    }
  }
  else
  {
    for (int b = 0; b < N; ++b)
    {
      const T *aptr = A.data() + H * b;   // A(b,i)   => A[H*b]

      for (const auto &nb_nonzero: B.getNbNonzerosCol())
      {
        typename ComputationType<T>::type x = 0;

        for (uint16_t k = 0; k < nb_nonzero; ++k, ++offset_data)
        {
          uint16_t j = B.getIndices()[offset_data];
          x += (typename ComputationType<T>::type) aptr[j] * B.getDataSparse()[offset_data];
          COUNTERS_MAC(B.getDataSparse()[offset_data]);
        }

        ComputationType<T>::quantize(x, shift);
        COUNTERS(x);
        SATURATE(x);
        m_out(0, b, i) = (T) x;
        i++;
      }
    }
  }

  return true;
}

#if __SSE4_2__
template<> inline bool MatMul<int16_t>::apply_sparse_matmul_simd8(std::vector<Tensor<int16_t> *> &in)
{
  using T = int16_t;

  const Tensor<int16_t> &A{ *in[0] };
  const Tensor<int16_t> &B{ *in[1] };
  const int              shift{ in[1]->quantizer + m_q };
  const int              last = A.dims().size() - 1;
  const int              N{ A.dims()[last - 1] };
  const int              H{ A.dims()[last] };

  m_out.fill(0);

  int t = 0;

  for (int b = 0; b < N; ++b)
  {
    const int16_t *aptr = A.data() + H * b;   // A(b,i)   => A[H*b]
    const int16_t *bptr = B.getDataSparse().data();
    const auto *   idx  = B.getIndices().data();

    for (const auto &nb_nonzero: B.getNbNonzerosCol())
    {
      __m128i s = _mm_setzero_si128();

      for (int j = 0; j < nb_nonzero; j += 8)
      {
        int16_t eA[8];

        for (int k = 0; k < 8; ++k)
        {
          eA[k] = *(aptr + *idx);
          idx++;
        }

        __m128i a  = _mm_loadu_si128((__m128i *) eA);
        __m128i b  = _mm_loadu_si128((const __m128i *) bptr);
        __m128i ab = _mm_madd_epi16(a, b);   // res in si32

        s = _mm_add_epi32(s, ab);

        bptr += 8;
      }

      __m128i hi64  = _mm_unpackhi_epi64(s, s);   // 3-operand non-destructive AVX lets us save a byte without needing a movdqa
      __m128i sum64 = _mm_add_epi32(hi64, s);
      __m128i hi32  = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1));   // Swap the low two elements
      __m128i sum32 = _mm_add_epi32(sum64, hi32);

      typename ComputationType<int16_t>::type z = _mm_cvtsi128_si32(sum32);

      ComputationType<int16_t>::quantize(z, shift);
      SATURATE(z);
      m_out[t] = z;

      t++;
    }
  }
  return true;
}

template<typename T> bool MatMul<T>::apply_sparse_matmul_simd8(std::vector<Tensor<T> *> &in) { return apply_sparse_matmul(in); }
#endif

#if __AVX2__
template<> inline bool MatMul<int16_t>::apply_sparse_matmul_simd16(std::vector<Tensor<int16_t> *> &in)
{
  using T = int16_t;

  const Tensor<int16_t> &A{ *in[0] };
  const Tensor<int16_t> &B{ *in[1] };
  const int              shift{ in[1]->quantizer + m_q };
  const int              last = A.dims().size() - 1;
  const int              N{ A.dims()[last - 1] };
  const int              H{ A.dims()[last] };

  m_out.fill(0);

  int t = 0;

  for (int b = 0; b < N; ++b)
  {
    const int16_t *aptr = A.data() + H * b;   // A(b,i)   => A[H*b]
    const int16_t *bptr = B.getDataSparse().data();
    const auto *   idx  = B.getIndices().data();

    for (auto nb_nonzero: B.getNbNonzerosCol())
    {
      __m256i s = _mm256_setzero_si256();

      for (int j = 0; j < nb_nonzero; j += 16)
      {
        int16_t eA[16];

        for (int k = 0; k < 16; ++k)
        {
          eA[k] = *(aptr + *idx);
          idx++;
        }

        __m256i a  = _mm256_loadu_si256((__m256i *) eA);
        __m256i b  = _mm256_loadu_si256((const __m256i *) bptr);
        __m256i ab = _mm256_madd_epi16(a, b);

        s = _mm256_add_epi32(s, ab);

        bptr += 16;
      }

      typename ComputationType<int16_t>::type z = sum32_int16(s);
      ComputationType<int16_t>::quantize(z, shift);
      SATURATE(z);
      m_out[t] = z;

      t++;
    }
  }

  return true;
}

template<typename T> bool MatMul<T>::apply_sparse_matmul_simd16(std::vector<Tensor<T> *> &in) { return apply_sparse_matmul(in); }
#endif
#endif

template<typename T> bool MatMul<T>::init(const std::vector<Tensor<T> *> &in)
{
  // output[..., i, j] = sum_k (a[..., i, k] * b[..., k, j]), for all indices i, j.
  SADL_DBG(std::cout << "  - input matmul: " << in[0]->dims() << ' ' << in[1]->dims() << std::endl);

  if (in.size() != 2)
  {
    return false;
  }
  // cases:
  // A: always a tensor
  // B: const (because assumed transposed)
  // 1- A [x,y] B[y,z] || A [x,y,z] B[x,z,t] || A [1,x,y,z] B[1,x,z,t]
  // 2- A [1,x,y] B[y,z] || A [1,x,y,z] B[x,z,t]
  if (in[1]->dims().size() < 2 || in[1]->dims().size() > 4)
  {
    return false;
  }
  if (in[0]->dims().size() != in[1]->dims().size() && !(in[0]->dims().size() - 1 == in[1]->dims().size() && in[0]->dims()[0] == 1))
  {
    return false;
  }
  Dimensions dim  = in[0]->dims();
  const int  last = in[0]->dims().size() - 1;

  if (in[0]->dims().size() - 1 == in[1]->dims().size())
  {
    for (int k = 1; k < last - 1; ++k)
    {
      if (in[0]->dims()[k] != in[1]->dims()[k - 1])
      {
        return false;
      }
    }
    if (in[0]->dims()[last] != in[1]->dims()[last - 2])
    {
      return false;
    }
  }
  else
  {
#if DEBUG_MODEL
    if (in[0]->dims()[0] != 1)
      std::cout << "[WARN] suspicious operation (likely second input not a Const)" << std::endl;
#endif
    // Excluding the last two dimensions, the dimension
    // of index i in the first input Tensor<T> must be equal
    // to the dimension of index i in the second input
    // Tensor<T>.
    for (int k = 0; k < last - 1; ++k)
    {
      if (in[0]->dims()[k] != in[1]->dims()[k])
      {
        return false;
      }
    }
    if (in[0]->dims()[last] != in[1]->dims()[last - 1])
    {
      return false;
    }
  }
  dim[last] = in[1]->dims().back();
  m_out.resize(dim);
  SADL_DBG(std::cout << "  - output matmul: " << m_out.dims() << std::endl);
  m_initDone = true;
  return true;
}

template<typename T> bool MatMul<T>::loadInternal(std::istream &file, Version v)
{
  file.read((char *) &m_q, sizeof(m_q));
  SADL_DBG(std::cout << "  - q: " << m_q << std::endl);
  return true;
}

}   // namespace layers
}   // namespace sadl
