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
#include <cmath>
#include "layer.h"
#if __AVX2__
#include <immintrin.h>
#endif

namespace sadl
{
namespace layers
{
template<typename T> template<int s_h, int s_w> void Conv2D<T>::conv2d_ixj_s_peel(const Tensor<T> &A, const Tensor<T> &kernel)
{
  constexpr int im_nb = 0;
  const int     shift = kernel.quantizer + m_q;
  const int     ihalf_size{ kernel.dims()[0] / 2 };
  const int     jhalf_size{ kernel.dims()[1] / 2 };
  int           in_H{ A.dims()[1] };
  int           in_W{ A.dims()[2] };
  const int     in_D{ A.dims()[3] };
  const int     nb_filters{ kernel.dims()[2] };
  const int     top{ m_pads[0] };
  const int     left{ m_pads[1] };
  int           start_h{ ihalf_size - top };
  int           start_w{ jhalf_size - left };

  const int cout_by_g = nb_filters / m_groups;
  const int cin_by_g  = in_D / m_groups;
  for (int filter_nb = 0; filter_nb < nb_filters; ++filter_nb)
  {
    int offset = (filter_nb / cout_by_g) * cin_by_g;

    // corners
    {
      int  im_i;
      int  im_j;
      auto loop_with_cond = [&, filter_nb, shift](int i0, int i1, int j0, int j1)
      {
        typename ComputationType<T>::type x = 0;
        for (int filter_i = i0; filter_i <= i1; ++filter_i)
        {
          for (int filter_j = j0; filter_j <= j1; ++filter_j)
          {
            for (int filter_d = 0; filter_d < in_D / m_groups; ++filter_d)
            {
              int ii = im_i + filter_i;
              int jj = im_j + filter_j;
              int ki = ihalf_size + filter_i;
              int kj = jhalf_size + filter_j;
              x += (typename ComputationType<T>::type) A(im_nb, ii, jj, offset + filter_d) * kernel(ki, kj, filter_nb, filter_d);
              COUNTERS_MAC(kernel(ki, kj, filter_nb, filter_d));
            }
          }
        }
        ComputationType<T>::quantize(x, shift);
        COUNTERS(x);
        SATURATE(x);
        COUNTERS_MAC_NOP(in_D / m_groups*(kernel.dims()[0]*kernel.dims()[1]-(i1-i0+1)*(j1-j0+1)));
        m_out(im_nb, im_i / s_h, im_j / s_w, filter_nb) = static_cast<T>(x);
      };

      im_j = start_w;
      if (im_j < in_W)
      {   // left side
        im_i = start_h;
        if (im_i < in_H)
        {   // top left corner
          loop_with_cond(-start_h, ihalf_size, -start_w, jhalf_size);
        }
        im_i = ((in_H - ihalf_size - start_h) / s_h) * s_h + start_h;
        if (im_i > 0 && im_i < in_H && im_i != start_h)
        {   // bottom left corner
          const int end_i = (im_i + 1 < in_H) ? 1 : 0;
          loop_with_cond(-ihalf_size, end_i, -start_w, jhalf_size);
        }
      }

      im_j            = ((in_W - jhalf_size - start_w) / s_w) * s_w + start_w;
      const int end_j = (im_j + 1 < in_W) ? 1 : 0;
      if (im_j > 0 && im_j < in_W && im_j != start_w)
      {   // rihgt side
        im_i = start_h;
        if (im_i < in_H)
        {   // top right corner
          loop_with_cond(-start_h, ihalf_size, -jhalf_size, end_j);
        }

        im_i = ((in_H - ihalf_size - start_h) / s_h) * s_h + start_h;
        if (im_i > 0 && im_i < in_H && im_i != start_h)
        {   // bottom right corner
          const int end_i = (im_i + 1 < in_H) ? 1 : 0;
          loop_with_cond(-ihalf_size, end_i, -jhalf_size, end_j);
        }
      }
    }

    // vertical borders
    {
      for (int im_i = start_h + s_h; im_i < in_H - ihalf_size; im_i += s_h)
      {
        int im_j = start_w;   // can be only 0 or 1
        if (im_j < in_W)
        {   // left side
          typename ComputationType<T>::type x = 0;
          for (int filter_i = -ihalf_size; filter_i <= ihalf_size; ++filter_i)
          {
            for (int filter_j = -start_w; filter_j <= jhalf_size; ++filter_j)
            {
              for (int filter_d = 0; filter_d < in_D / m_groups; ++filter_d)
              {
                int ii = im_i + filter_i;
                int jj = im_j + filter_j;
                int ki = ihalf_size + filter_i;
                int kj = jhalf_size + filter_j;
                x += (typename ComputationType<T>::type) A(im_nb, ii, jj, offset + filter_d) * kernel(ki, kj, filter_nb, filter_d);
                COUNTERS_MAC(kernel(ki, kj, filter_nb, filter_d));
              }
            }
          }
          ComputationType<T>::quantize(x, shift);
          COUNTERS(x);
          SATURATE(x);
          COUNTERS_MAC_NOP(in_D / m_groups*(kernel.dims()[0]*kernel.dims()[1]-(ihalf_size+ihalf_size+1)*(jhalf_size+start_w+1)));
          m_out(im_nb, im_i / s_h, im_j / s_w, filter_nb) = static_cast<T>(x);
        }

        im_j = ((in_W - jhalf_size - start_w) / s_w) * s_w + start_w;
        if (im_j > 0 && im_j < in_W && im_j != start_w)
        {   // rihgt side
          typename ComputationType<T>::type x          = 0;
          const int                         end_filter = (im_j + 1) < in_W ? 1 : 0;
          for (int filter_i = -ihalf_size; filter_i <= ihalf_size; ++filter_i)
          {
            for (int filter_j = -jhalf_size; filter_j <= end_filter; ++filter_j)
            {
              for (int filter_d = 0; filter_d < in_D / m_groups; ++filter_d)
              {
                int ii = im_i + filter_i;
                int jj = im_j + filter_j;
                int ki = ihalf_size + filter_i;
                int kj = jhalf_size + filter_j;
                x += (typename ComputationType<T>::type) A(im_nb, ii, jj, offset + filter_d) * kernel(ki, kj, filter_nb, filter_d);
                COUNTERS_MAC(kernel(ki, kj, filter_nb, filter_d));
              }
            }
          }
          ComputationType<T>::quantize(x, shift);
          COUNTERS(x);
          SATURATE(x);
          COUNTERS_MAC_NOP(in_D / m_groups*(kernel.dims()[0]*kernel.dims()[1]-(ihalf_size+ihalf_size+1)*(jhalf_size+end_filter+1)));
          m_out(im_nb, im_i / s_h, im_j / s_w, filter_nb) = static_cast<T>(x);
        }
      }
    }
    {
      // horizontal borders
      for (int im_j = s_w + start_w; im_j < in_W - jhalf_size; im_j += s_w)
      {
        int im_i = start_h;   // 0 or 1 -> adapt filter start
        if (im_i < in_H)
        {   // top line
          typename ComputationType<T>::type x = 0;
          for (int filter_i = -start_h; filter_i <= ihalf_size; ++filter_i)
          {
            for (int filter_j = -jhalf_size; filter_j <= jhalf_size; ++filter_j)
            {
              for (int filter_d = 0; filter_d < in_D / m_groups; ++filter_d)
              {
                int ii = im_i + filter_i;
                int jj = im_j + filter_j;
                int ki = ihalf_size + filter_i;
                int kj = jhalf_size + filter_j;
                x += (typename ComputationType<T>::type) A(im_nb, ii, jj, offset + filter_d) * kernel(ki, kj, filter_nb, filter_d);
                COUNTERS_MAC(kernel(ki, kj, filter_nb, filter_d));
              }
            }
          }

          ComputationType<T>::quantize(x, shift);
          COUNTERS(x);
          SATURATE(x);
          COUNTERS_MAC_NOP(in_D / m_groups*(kernel.dims()[0]*kernel.dims()[1]-(ihalf_size+start_h+1)*(jhalf_size+jhalf_size+1)));
          m_out(im_nb, im_i / s_h, im_j / s_w, filter_nb) = static_cast<T>(x);
        }
        im_i = ((in_H - ihalf_size - start_h) / s_h) * s_h + start_h;
        if (im_i > 0 && im_i < in_H && im_i != start_h)
        {   // bottom line
          typename ComputationType<T>::type x          = 0;
          const int                         end_filter = (im_i + 1) < in_H ? 1 : 0;
          for (int filter_i = -ihalf_size; filter_i <= end_filter; ++filter_i)
          {
            for (int filter_j = -jhalf_size; filter_j <= jhalf_size; ++filter_j)
            {
              for (int filter_d = 0; filter_d < in_D / m_groups; ++filter_d)
              {
                int ii = im_i + filter_i;
                int jj = im_j + filter_j;
                int ki = ihalf_size + filter_i;
                int kj = jhalf_size + filter_j;
                x += (typename ComputationType<T>::type) A(im_nb, ii, jj, offset + filter_d) * kernel(ki, kj, filter_nb, filter_d);
                COUNTERS_MAC(kernel(ki, kj, filter_nb, filter_d));
              }
            }
          }
          ComputationType<T>::quantize(x, shift);
          COUNTERS(x);
          SATURATE(x);
          COUNTERS_MAC_NOP(in_D / m_groups*(kernel.dims()[0]*kernel.dims()[1]-(ihalf_size+end_filter+1)*(jhalf_size+jhalf_size+1)));
          m_out(im_nb, im_i / s_h, im_j / s_w, filter_nb) = static_cast<T>(x);
        }
      }
    }
  }   // filter_nb
}

template<typename T> template<int s_h, int s_w> void Conv2D<T>::conv2d_ixj_s_core(const Tensor<T> &A, const Tensor<T> &kernel)
{
  constexpr int im_nb = 0;
  const int     ihalf_size{ kernel.dims()[0] / 2 };
  const int     jhalf_size{ kernel.dims()[1] / 2 };
  const int     in_D{ A.dims()[3] };
  const int     nb_filters{ kernel.dims()[2] };
  const int     shift     = kernel.quantizer + m_q;
  const int     cout_by_g = nb_filters / m_groups;
  const int     cin_by_g  = in_D / m_groups;
  const int     top{ m_pads[0] };
  const int     left{ m_pads[1] };
  int           in_H{ A.dims()[1] };
  int           in_W{ A.dims()[2] };
  int           start_h{ ihalf_size - top };
  int           start_w{ jhalf_size - left };
#if DEBUG_SIMD && __AVX2__
  std::cout << "\n[WARN] generic version conv (stride known) " << kernel.dims()[0] << "x" << kernel.dims()[1] << "g" << m_groups << " inD=" << in_D << " outD=" << nb_filters
            << " s=[" << s_w << ' ' << s_h << "]  " << in_H << 'x' << in_W << " "
            << "?? kMAC" << std::endl;
#endif
  assert(start_h + s_h - ihalf_size >= 0);
  assert(start_w + s_w - jhalf_size >= 0);
  for (int im_i = start_h + s_h; im_i < in_H - ihalf_size; im_i += s_h)
  {
    for (int im_j = start_w + s_w; im_j < in_W - jhalf_size; im_j += s_w)
    {
      for (int filter = 0; filter < nb_filters; ++filter)
      {
        int                               offset = (filter / cout_by_g) * cin_by_g;
        typename ComputationType<T>::type x      = 0;
        for (int filter_i = -ihalf_size; filter_i <= ihalf_size; ++filter_i)
        {
          for (int filter_j = -jhalf_size; filter_j <= jhalf_size; ++filter_j)
          {
            for (int filter_d = 0; filter_d < in_D / m_groups; ++filter_d)
            {
              int ii = im_i + filter_i;
              int jj = im_j + filter_j;
              int ki = ihalf_size + filter_i;
              int kj = jhalf_size + filter_j;
              x += (typename ComputationType<T>::type) A(im_nb, ii, jj, offset + filter_d) * kernel(ki, kj, filter, filter_d);
              COUNTERS_MAC(kernel(ki, kj, filter, filter_d));
            }
          }
        }
        ComputationType<T>::quantize(x, shift);
        COUNTERS(x);
        SATURATE(x);
        m_out(im_nb, im_i / s_h, im_j / s_w, filter) = static_cast<T>(x);
      }
    }
  }
}


template<typename T> template<int in_D, int ihalf_size, int jhalf_size> void Conv2D<T>::conv2d_ixj_s11_g1_d_core(const Tensor<T> &A, const Tensor<T> &kernel)
{
  constexpr int im_nb = 0;
  constexpr int s_h = 1;
  constexpr int s_w = 1;
  const int     nb_filters{ kernel.dims()[2] };
  const int     shift     = kernel.quantizer + m_q;
  const int     top{ m_pads[0] };
  const int     left{ m_pads[1] };
  int           in_H{ A.dims()[1] };
  int           in_W{ A.dims()[2] };
  int           start_h{ ihalf_size - top };
  int           start_w{ jhalf_size - left };
#if DEBUG_SIMD && __AVX2__
  std::cout << "\n[WARN] generic version conv2d_ixj_s11_g1_d_core (s=1, g=1, known kernel)" << kernel.dims()[0] << "x" << kernel.dims()[1] << "g" << m_groups << " inD=" << in_D << " outD=" << nb_filters
            << " s=[" << s_w << ' ' << s_h << "]  " << in_H << 'x' << in_W << " "
            << "?? kMAC" << std::endl;
#endif
  assert(start_h + s_h - ihalf_size >= 0);
  assert(start_w + s_w - jhalf_size >= 0);
  for (int im_i = start_h + s_h; im_i < in_H - ihalf_size; im_i += s_h)
  {
    for (int im_j = start_w + s_w; im_j < in_W - jhalf_size; im_j += s_w)
    {
      for (int filter = 0; filter < nb_filters; ++filter)
      {
        typename ComputationType<T>::type x      = 0;
        for (int filter_d = 0; filter_d < in_D ; ++filter_d)
        { // fixed
        for (int filter_i = -ihalf_size; filter_i <= ihalf_size; ++filter_i)
        {   // fixed
          for (int filter_j = -jhalf_size; filter_j <= jhalf_size; ++filter_j)
          {   // fixed
              int ii = im_i + filter_i;
              int jj = im_j + filter_j;
              int ki = ihalf_size + filter_i;
              int kj = jhalf_size + filter_j;
              x += (typename ComputationType<T>::type) A(im_nb, ii, jj, filter_d) * kernel(ki, kj, filter, filter_d);
              COUNTERS_MAC(kernel(ki, kj, filter, filter_d));
            }
          }
        }
        ComputationType<T>::quantize(x, shift);
        COUNTERS(x);
        SATURATE(x);
        m_out(im_nb, im_i / s_h, im_j / s_w, filter) = static_cast<T>(x);
      }
    }
  }
}

#if __AVX512F__
template<> template<int in_D, int ihalf_size, int jhalf_size> void Conv2D<float>::simd16_conv2d_ixj_s11_g1_d_core(const Tensor<float> &A, const Tensor<float> &kernel)
{
  static_assert(in_D % 16 == 0, "Should be used with mod16 filters.");
  constexpr int im_nb = 0;
  constexpr int s_h = 1;
  constexpr int s_w = 1;
  const int     nb_filters{ kernel.dims()[2] };
  const int     top{ m_pads[0] };
  const int     left{ m_pads[1] };
  int           in_H{ A.dims()[1] };
  int           in_W{ A.dims()[2] };
  int           start_h{ ihalf_size - top };
  int           start_w{ jhalf_size - left };
  assert(start_h + s_h - ihalf_size >= 0);
  assert(start_w + s_w - jhalf_size >= 0);

  for (int im_i = start_h + s_h; im_i < in_H - ihalf_size; im_i += s_h)
  {
    for (int im_j = start_w + s_w; im_j < in_W - jhalf_size; im_j += s_w)
    {
      for (int filter = 0; filter < nb_filters; ++filter)
      {
        __m512 s = _mm512_setzero_ps();
        for (int filter_i = -ihalf_size; filter_i <= ihalf_size; ++filter_i)
        {   // fixed
          int ii = im_i + filter_i;
          int ki = ihalf_size + filter_i;
          for (int filter_j = -jhalf_size; filter_j <= jhalf_size; ++filter_j)
          {   // fixed
              int jj = im_j + filter_j;
              int kj = jhalf_size + filter_j;
              for (int filter_d = 0; filter_d < in_D; filter_d += 16)
              {
                const float *kptr = kernel.addr(ki, kj, filter, filter_d);
                const __m512 k0   = _mm512_load_ps(kptr);
                const float *aptr = A.addr(im_nb, ii, jj, filter_d);
#if __FMA__
                s = _mm512_fmadd_ps(k0, _mm512_load_ps(aptr), s);
#else
                const __m512 m0 = _mm512_mul_ps(k0, _mm512_load_ps(aptr));
                s               = _mm512_add_ps(s, m0);
#endif
              }
          }
        }
        m_out(im_nb, im_i / s_h, im_j / s_w, filter) = sum16_float(s);
      }
    }
  }
}
#endif

#if __AVX512BW__
template<> template<int in_D, int ihalf_size, int jhalf_size> void Conv2D<int16_t>::simd32_conv2d_ixj_s11_g1_d_core(const Tensor<int16_t> &A, const Tensor<int16_t> &kernel)
{
  static_assert(in_D % 32 == 0, "Should be used with mod32 filters.");
  using T=int16_t;
  constexpr int im_nb = 0;
  constexpr int s_h = 1;
  constexpr int s_w = 1;
  const int     nb_filters{ kernel.dims()[2] };
  const int     shift     = kernel.quantizer + m_q;
  const int     top{ m_pads[0] };
  const int     left{ m_pads[1] };
  int           in_H{ A.dims()[1] };
  int           in_W{ A.dims()[2] };
  int           start_h{ ihalf_size - top };
  int           start_w{ jhalf_size - left };

  assert(start_h + s_h - ihalf_size >= 0);
  assert(start_w + s_w - jhalf_size >= 0);
  for (int im_i = start_h + s_h; im_i < in_H - ihalf_size; im_i += s_h)
  {
    for (int im_j = start_w + s_w; im_j < in_W - jhalf_size; im_j += s_w)
    {
      for (int filter = 0; filter < nb_filters; ++filter)
      {
        __m512i s = _mm512_setzero_si512();
        for (int filter_i = -ihalf_size; filter_i <= ihalf_size; ++filter_i)
        {   // fixed
          for (int filter_j = -jhalf_size; filter_j <= jhalf_size; ++filter_j)
          {   // fixed
            for (int filter_d = 0; filter_d < in_D; filter_d += 32)
            {
              const int      ii   = im_i + filter_i;
              const int      jj   = im_j + filter_j;
              const int      ki   = ihalf_size + filter_i;
              const int      kj   = jhalf_size + filter_j;
              const __m512i *kptr = (const __m512i *) kernel.addr(ki, kj, filter, filter_d);
              const __m512i  k0   = _mm512_load_si512(kptr);
              const __m512i *aptr = (const __m512i *) A.addr(im_nb, ii, jj, filter_d);
              const __m512i  v0   = _mm512_load_si512(aptr);

              const __m512i mad0 = _mm512_madd_epi16(k0, v0);   // res in si32
              s                  = _mm512_add_epi32(s, mad0);
            }
          }
        }
        typename ComputationType<T>::type z = (_mm512_reduce_add_epi32(s) >> shift);
        COUNTERS(z);
        SATURATE(z);
        m_out(im_nb, im_i / s_h, im_j / s_w, filter) = z;
      }
    }
  }
}
#endif
#if __AVX2__
template<> template<int in_D, int ihalf_size, int jhalf_size> void Conv2D<int16_t>::simd16_conv2d_ixj_s11_g1_d_core(const Tensor<int16_t> &A, const Tensor<int16_t> &kernel)
{
  static_assert(in_D % 16 == 0, "Should be used with mod16 filters.");
  using T=int16_t;
  constexpr int im_nb = 0;
  constexpr int s_h = 1;
  constexpr int s_w = 1;
  const int     nb_filters{ kernel.dims()[2] };
  const int     shift     = kernel.quantizer + m_q;
  const int     top{ m_pads[0] };
  const int     left{ m_pads[1] };
  int           in_H{ A.dims()[1] };
  int           in_W{ A.dims()[2] };
  int           start_h{ ihalf_size - top };
  int           start_w{ jhalf_size - left };
#if DEBUG_SIMD && __AVX512BW__
  if (in_D % 32 == 0 || in_D>=32)
    std::cout << "\n[WARN] avx2 (instead of avx512) version simd16_conv2d_ixj_s11_g1_d_core (s=1, g=1, known kernel)" << kernel.dims()[0] << "x" << kernel.dims()[1] << "g" << m_groups << " inD=" << in_D << " outD=" << nb_filters
              << " s=[" << s_w << ' ' << s_h << "]  " << in_H << 'x' << in_W << " "
              << "?? kMAC" << std::endl;
#endif

  assert(start_h + s_h - ihalf_size >= 0);
  assert(start_w + s_w - jhalf_size >= 0);
  for (int im_i = start_h + s_h; im_i < in_H - ihalf_size; im_i += s_h)
  {
    for (int im_j = start_w + s_w; im_j < in_W - jhalf_size; im_j += s_w)
    {
      for (int filter = 0; filter < nb_filters; ++filter)
      {
        __m256i s = _mm256_setzero_si256();
        for (int filter_i = -ihalf_size; filter_i <= ihalf_size; ++filter_i)
        {   // fixed
          for (int filter_j = -jhalf_size; filter_j <= jhalf_size; ++filter_j)
          {   // fixed
            for (int filter_d = 0; filter_d < in_D; filter_d += 16)
            {
              const int      ii   = im_i + filter_i;
              const int      jj   = im_j + filter_j;
              const int      ki   = ihalf_size + filter_i;
              const int      kj   = jhalf_size + filter_j;
              const __m256i *kptr = (const __m256i *) kernel.addr(ki, kj, filter, filter_d);
              const __m256i  k0   = _mm256_load_si256(kptr);
              const __m256i *aptr = (const __m256i *) A.addr(im_nb, ii, jj, filter_d);
              const __m256i  v0   = _mm256_load_si256(aptr);

              const __m256i mad0 = _mm256_madd_epi16(k0, v0);   // res in si32
              s                  = _mm256_add_epi32(s, mad0);
            }
          }
        }
        typename ComputationType<T>::type z = (sum32_int16(s) >> shift);
        COUNTERS(z);
        SATURATE(z);
        m_out(im_nb, im_i / s_h, im_j / s_w, filter) = z;
      }
    }
  }
}
#endif
template<typename T> template<int in_D, int ihalf_size, int jhalf_size> void Conv2D<T>::conv2d_ixj_s11_gD_d_core(const Tensor<T> &A, const Tensor<T> &kernel)
{
  constexpr int nb_filters = in_D;
  constexpr int s_h        = 1;
  constexpr int s_w        = 1;
  constexpr int im_nb      = 0;
  const int     shift      = kernel.quantizer + m_q;
  const int     top{ m_pads[0] };
  const int     left{ m_pads[1] };
  int           in_H{ A.dims()[1] };
  int           in_W{ A.dims()[2] };
  int           start_h{ ihalf_size - top };
  int           start_w{ jhalf_size - left };
#if DEBUG_SIMD && __AVX2__
  std::cout << "\n[WARN] partially generic version conv2d_ixj_s11_gD_d_core " << kernel.dims()[0] << "x" << kernel.dims()[1] << "g" << m_groups << " inD=" << in_D
            << " outD=" << nb_filters << " s=[" << s_w << ' ' << s_h << "]  " << in_H << 'x' << in_W << " "
            << "?? kMAC" << std::endl;
#endif

  for (int im_i = start_h + top; im_i < in_H - ihalf_size; im_i += s_h)
  {
    for (int im_j = start_w + left; im_j < in_W - jhalf_size; im_j += s_w)
    {
      for (int filter = 0; filter < nb_filters; ++filter)
      { // fixed
        typename ComputationType<T>::type x = 0;

        for (int filter_i = -ihalf_size; filter_i <= ihalf_size; ++filter_i)
        {   // fixed
          for (int filter_j = -jhalf_size; filter_j <= jhalf_size; ++filter_j)
          {   // fixed
            int ii = im_i + filter_i;
            int jj = im_j + filter_j;
            int ki = ihalf_size + filter_i;
            int kj = jhalf_size + filter_j;
            x += (typename ComputationType<T>::type) A(im_nb, ii, jj, filter) * kernel(ki, kj, filter, 0);
            COUNTERS_MAC(kernel(ki, kj, filter, 0));
          }
        }

        ComputationType<T>::quantize(x, shift);
        COUNTERS(x);
        SATURATE(x);
        m_out(im_nb, im_i / s_h, im_j / s_w, filter) = static_cast<T>(x);
      }
    }
  }
}

template<typename T> template<int s_h, int s_w> void Conv2D<T>::conv2d_ixj_s_core_dispatch(const Tensor<T> &A, const Tensor<T> &kernel)
{
  const int in_D{ A.dims()[3] };
  const int nb_filters{ kernel.dims()[2] };

  // grouped conv with stride 1 and inD==outD
#if __AVX2__
#define CONV_MOD32 simd32_conv2d_ixj_s11_g1_d_core
#define CONV_MOD16 simd16_conv2d_ixj_s11_g1_d_core
#else
#define CONV_MOD32 conv2d_ixj_s11_g1_d_core
#define CONV_MOD16 conv2d_ixj_s11_g1_d_core
#endif
  if (in_D == m_groups && in_D == nb_filters && s_h == 1 && s_w == 1)
  {
    if (kernel.dims()[0] / 2 == 0 && kernel.dims()[1] / 2 == 1)
    {
      constexpr int ki = 0;
      constexpr int kj = 1;
      switch (in_D)
      {
      case 8:
        conv2d_ixj_s11_gD_d_core<8, ki, kj>(A, kernel);
        return;
        break;
      case 16:
        conv2d_ixj_s11_gD_d_core<16, ki, kj>(A, kernel);
        return;
        break;
      case 24:
        conv2d_ixj_s11_gD_d_core<24, ki, kj>(A, kernel);
        return;
        break;
      default:   // do default
        break;
      }
    }
    else if (kernel.dims()[0] / 2 == 1 && kernel.dims()[1] / 2 == 0)
    {
      constexpr int ki = 1;
      constexpr int kj = 0;
      switch (in_D)
      {
      case 8:
        conv2d_ixj_s11_gD_d_core<8, ki, kj>(A, kernel);
        return;
        break;
      case 16:
        conv2d_ixj_s11_gD_d_core<16, ki, kj>(A, kernel);
        return;
        break;
      case 24:
        conv2d_ixj_s11_gD_d_core<24, ki, kj>(A, kernel);
        return;
        break;
      default:   // do default
        break;
      }
    }
  }

  // no grouped conv with stride 1
  else if (m_groups == 1 && s_h == 1 && s_w == 1)
  {
    if (kernel.dims()[0] / 2 == 0 && kernel.dims()[1] / 2 == 1)
    {
      constexpr int ki = 0;
      constexpr int kj = 1;
      switch (in_D)
      {
      case 16:
        CONV_MOD16<16, ki, kj>(A, kernel);
        return;
        break;
      case 32:
        CONV_MOD32<32, ki, kj>(A, kernel);
        return;
        break;
      case 64:
        CONV_MOD32<64, ki, kj>(A, kernel);
        return;
        break;
      default:   // do default
        break;
      }
    }
    else if (kernel.dims()[0] / 2 == 1 && kernel.dims()[1] / 2 == 0)
    {
      constexpr int ki = 1;
      constexpr int kj = 0;
      switch (in_D)
      {
      case 16:
        CONV_MOD16<16, ki, kj>(A, kernel);
        return;
        break;
      case 32:
        CONV_MOD32<32, ki, kj>(A, kernel);
        return;
        break;
      case 64:
        CONV_MOD32<64, ki, kj>(A, kernel);
        return;
        break;
      default:   // do default
        break;
      }
    }
  }
#undef CONV_MOD16
  conv2d_ixj_s_core<s_h, s_w>(A, kernel);
}

}   // namespace layers
}   // namespace sadl
