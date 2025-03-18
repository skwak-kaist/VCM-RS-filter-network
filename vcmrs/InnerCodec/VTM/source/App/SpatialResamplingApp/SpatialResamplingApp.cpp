/* The copyright in this software is being made available under the BSD
 * License, included below. This software may be subject to other third party
 * and contributor rights, including patent rights, and no such rights are
 * granted under this license.
 *
 * Copyright (c) 2010-2021, ITU/ISO/IEC
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

/** \file     SpatialResamplingApp.cpp
    \brief    SpatialResampling application
*/

//! \ingroup SpatialResamplingApp
//! \{

#include <cstdio>
#include <cctype>
#include <vector>
#include <utility>
#include <fstream>
#include <sstream>
#include <ios>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include "../../Lib/CommonLib/CommonDef.h"
#include "SpatialResamplingApp.h"
#include "../../Lib/Utilities/VideoIOYuv.h"
#include "../../Lib/CommonLib/Picture.h"

//! \ingroup SpatialResamplingApp
//! \{

#define DEBUGMODE 0

SpatialResamplingApp::SpatialResamplingApp(std::string& inputYuvFileName, std::string& outputYuvFileName,
                               std::string& inputVideoInfoFileName, SRVideoInfoParam& videoInfoParam)
{
  m_bd[ChannelType::LUMA]   = (int) videoInfoParam.bitDepth;
  m_bd[ChannelType::CHROMA] = (int) videoInfoParam.bitDepth;
  m_cf                      = ChromaFormat::_420;   // 420
}

SpatialResamplingApp::~SpatialResamplingApp() {}

template<typename T> std::vector<T> readLine(std::string line)
{
  std::vector<T>     output;
  std::string        buf;
  std::istringstream iss(line);
  while (std::getline(iss, buf, ','))
  {
    output.push_back(static_cast<T>(std::stoi(buf)));
  }
  return output;
}

void readParametersFile(std::string inputVideoInfoFileName, std::vector<int>& dec_width, std::vector<int>& dec_height,
                        std::vector<int>& upscaled_width, std::vector<int>& upscaled_height, std::vector<Window>& dec_window,
                        std::vector<Window>& up_window, int startIdx, int endIdx)
{
  std::ifstream fp;
  std::string   buf;
  std::string   line;
  fp.open(inputVideoInfoFileName);
  if (!fp.is_open())
  {
    std::cerr << "Error: cannot open " << inputVideoInfoFileName << std::endl;
    return;
  }
  std::getline(fp, line);
  dec_width = readLine<int>(line);
  std::getline(fp, line);
  dec_height = readLine<int>(line);
  std::getline(fp, line);
  upscaled_width = readLine<int>(line);
  std::getline(fp, line);
  upscaled_height = readLine<int>(line);
  
  Window tmp;
  std::getline(fp, line);
  std::vector<int> left_offsets = readLine<int>(line);
  std::getline(fp, line);
  std::vector<int> top_offsets = readLine<int>(line);
  std::getline(fp, line);
  std::vector<int>right_offsets = readLine<int>(line);
  std::getline(fp, line);
  std::vector<int> bottom_offsets = readLine<int>(line);
  for (int i = startIdx; i < endIdx; i++) 
  {
    tmp.setWindow(left_offsets[i], right_offsets[i], top_offsets[i],  bottom_offsets[i]);
    dec_window.push_back(tmp);
  }
  std::getline(fp, line);
   left_offsets = readLine<int>(line);
  std::getline(fp, line);
  top_offsets = readLine<int>(line);
  std::getline(fp, line);
  right_offsets = readLine<int>(line);
  std::getline(fp, line);
  bottom_offsets = readLine<int>(line);
  for (int i = startIdx; i < endIdx; i++)
  {
    tmp.setWindow(left_offsets[i], right_offsets[i], top_offsets[i], bottom_offsets[i]);
    up_window.push_back(tmp);
  }
}

void SpatialResamplingApp::exeSR(std::string& inputYuvFileName, std::string& outputYuvFileName,
                                          std::string& inputVideoInfoFileName, SRVideoInfoParam& videoInfoParam)

{
  const int                     seqLength = videoInfoParam.endFrIdx - videoInfoParam.startFrIdx;
  std::vector<int>              dec_width(seqLength);
  std::vector<int>              dec_height(seqLength);
  std::vector<int>              upscaled_width(seqLength);
  std::vector<int>              upscaled_height(seqLength);
  std::vector<Window>           dec_window;
  std::vector<Window>           up_window;
  int                           scaleX      = SPS::getWinUnitX(m_cf);
  int                           scaleY      = SPS::getWinUnitY(m_cf);

  PelStorage outBuf;
  PelStorage buf;
  PelStorage dummy;

  int pad[2] = { 0, 0 };

  readParametersFile(inputVideoInfoFileName, dec_width, dec_height, upscaled_width, upscaled_height, dec_window,
                     up_window, videoInfoParam.startFrIdx, videoInfoParam.endFrIdx);

  VideoIOYuv inputYuvFrame;
  inputYuvFrame.open(inputYuvFileName, false, m_bd, m_bd, m_bd);
  VideoIOYuv outputYuvFrame;
  outputYuvFrame.open(outputYuvFileName, true, m_bd, m_bd, m_bd);

  const std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();

  for (int frCtr = 0; frCtr < seqLength; frCtr++)
  {

    int inWidth =
      dec_width[frCtr] - scaleX * (dec_window[frCtr].getWindowLeftOffset() + dec_window[frCtr].getWindowRightOffset());
    int inHeight =
      dec_height[frCtr] - scaleY * (dec_window[frCtr].getWindowTopOffset() + dec_window[frCtr].getWindowBottomOffset());
    int outWidth = upscaled_width[frCtr]
                   - scaleX * (up_window[frCtr].getWindowLeftOffset() + up_window[frCtr].getWindowRightOffset());
    int outHeight = upscaled_height[frCtr]
                    - scaleY * (up_window[frCtr].getWindowTopOffset() + up_window[frCtr].getWindowBottomOffset());

    const int        xScale = static_cast<int>(((inWidth << ScalingRatio::BITS) + (outWidth >> 1)) / outWidth);
    const int        yScale       = static_cast<int>(((inHeight << ScalingRatio::BITS) + (outHeight >> 1)) / outHeight);
    ScalingRatio scalingRatio         = { xScale, yScale };
    buf.create(m_cf, Area(0, 0, dec_width[frCtr], dec_height[frCtr]));
    dummy.create(m_cf, Area(0, 0, dec_width[frCtr], dec_height[frCtr]));
    outBuf.create(m_cf, Area(0, 0, outWidth, outHeight));
    if (!inputYuvFrame.read(buf, dummy, IPCOLOURSPACE_UNCHANGED, pad))
    {
      std::cerr << frCtr << " th frame cannot be read" << std::endl;
      return;
    }
    if (!(dec_width[frCtr] == upscaled_width[frCtr] && dec_height[frCtr] == upscaled_height[frCtr]))
    {
      std::cout << frCtr << "," << xScale << "," << yScale << "," << inWidth << "x" << inHeight << "," << outWidth
                << "x" << outHeight << "," << dec_width[frCtr] << "x" << dec_height[frCtr] << std::endl;
      Picture::rescalePicture(scalingRatio, buf, dec_window[frCtr], outBuf, up_window[frCtr], m_cf, m_bd, false, false,
                              true, false, true, videoInfoParam.filterIdx);

      if (!outputYuvFrame.write(upscaled_width[frCtr], upscaled_height[frCtr], outBuf, IPCOLOURSPACE_UNCHANGED, false,
                                up_window[frCtr].getWindowLeftOffset() * scaleX, up_window[frCtr].getWindowRightOffset() * scaleX,
                                up_window[frCtr].getWindowTopOffset() * scaleY, up_window[frCtr].getWindowBottomOffset() * scaleY,
                                ChromaFormat::UNDEFINED))
      {
        std::cerr << frCtr << " th frame cannot be written" << std::endl;
        return;
      }
    }
    else
    {
      if (!outputYuvFrame.write(dec_width[frCtr], dec_height[frCtr], buf, IPCOLOURSPACE_UNCHANGED, false,
                                up_window[frCtr].getWindowLeftOffset() * scaleX, up_window[frCtr].getWindowRightOffset() * scaleX,
                                up_window[frCtr].getWindowTopOffset() * scaleY, up_window[frCtr].getWindowBottomOffset() * scaleY,
                                ChromaFormat::UNDEFINED))
      {
        std::cerr << frCtr << " th frame cannot be written" << std::endl;
        return;
      }
    }
    buf.destroy();
    dummy.destroy();
    outBuf.destroy();
  }

  const std::chrono::steady_clock::time_point stop = std::chrono::steady_clock::now();
  const std::chrono::steady_clock::duration duration = stop - start;
  std::cout
    << "Processing Time of SRApp: " << std::fixed << std::setprecision(5)
    << std::chrono::duration_cast<std::chrono::microseconds>(duration).count() / 1000000.0
    << std::endl;

  inputYuvFrame.close();
  outputYuvFrame.close();
}

//! \}
