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

 /** \file     SpatialResamplingMain.cpp
     \brief    Frame splitter main function and command line handling
 */

#include <cstdio>
#include <cctype>
#include <cstring>
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include <ios>
#include <algorithm>
#include "../../Lib/Utilities/program_options_lite.h"
#include "SpatialResamplingApp.h"


namespace po = df::program_options_lite;

 //! \ingroup FrameSplitterNNApp
 //! \{


/**
  - Parse command line parameters
*/
bool parseCmdLine(int argc, char* argv[], std::string& inputYuvFileName, std::string& outputYuvFileName,
                  std::string& inputSRSizeInfoFileName, SRVideoInfoParam& videoInfoParam)
{
  bool doHelp = false;

  po::Options opts;
  opts.addOptions()
    ("-help",                            doHelp,                                             false, "This help text")
    ("i",                                inputYuvFileName,                            std::string(""), "Input YUV file name")
    ("p",                                inputSRSizeInfoFileName,                          std::string(""), "Video info file name")
    ("o",                                outputYuvFileName,                    std::string(""), "Output YUV file name")
    ("w",                                videoInfoParam.widthAfterSR,                          (int) 0, "Width of output pictures")
    ("h",                                videoInfoParam.heightAfterSR,                          (int)0, "Height of output pictures")
    ("d",                                videoInfoParam.bitDepth,                          (int)  10, "BitDepth of input/output pictures")
    ("s",                                videoInfoParam.startFrIdx,                    (int) 0, "Start farme index")
    ("e",                                videoInfoParam.endFrIdx,                    (int)0, "End frame index")
    ("f",                                videoInfoParam.filterIdx,                    (int)1, "FilterIdx. 0:VVC RPR, 1:Alternative, 2:ECM, 3:Bilinear")
    ;

  po::setDefaults(opts);
  po::ErrorReporter err;
  const std::list<const char*>& argvUnhandled = po::scanArgv(opts, argc, (const char**) argv, err);

  if (argc == 1 || doHelp)
  {
    /* argc == 1: no options have been specified */
    po::doHelp(std::cout, opts);

    return false;
  }

  for (std::list<const char*>::const_iterator it = argvUnhandled.begin(); it != argvUnhandled.end(); it++)
  {
    std::cerr << "Unhandled argument ignored: `" << *it << "'" << std::endl;
  }

  if (videoInfoParam.widthAfterSR <= 0 || videoInfoParam.heightAfterSR <= 0)
  {
    std::cerr << "invalid parameters'" << std::endl;
  }

  return true;
}


/**
  - Subpicture merge main()
 */
int main(int argc, char* argv[])
{
  std::string inputYUVFileName;
  std::string      inputSRSizeInfoFileName;
  std::string outputYUVFileName;
  SRVideoInfoParam videoInfoParam;

  if (!parseCmdLine(argc, argv, inputYUVFileName, inputSRSizeInfoFileName, outputYUVFileName, videoInfoParam))
  {
    return 1;
  }

  SpatialResamplingApp* spatialResamplingApp =
    new SpatialResamplingApp(inputYUVFileName, inputSRSizeInfoFileName, outputYUVFileName, videoInfoParam);

  spatialResamplingApp->exeSR(inputYUVFileName, inputSRSizeInfoFileName, outputYUVFileName, videoInfoParam);

  delete spatialResamplingApp;

  return 0;
}

//! \}
