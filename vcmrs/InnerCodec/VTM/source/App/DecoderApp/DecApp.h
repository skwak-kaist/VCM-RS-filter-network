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

/** \file     TAppDecLib.h
    \brief    Decoder application class (header)
*/

#ifndef __DECAPP__
#define __DECAPP__

#pragma once

#include "Utilities/VideoIOYuv.h"
#include "CommonLib/Picture.h"
#include "DecoderLib/DecLib.h"
#include "DecAppCfg.h"

//! \ingroup DecoderApp
//! \{

// ====================================================================================================================
// Class definition
// ====================================================================================================================

/// decoder application class
class DecApp : public DecAppCfg
{
private:
  // class interface
  DecLib          m_cDecLib;                     ///< decoder class

#if NNVC_DUMP_DATA
  std::ofstream                            m_jsonFile;
  int                                      m_dumpDataCnt=-1; // counter for data dump
#if NNVC_USE_REC_AFTER_ALF
  std::unordered_map<int, VideoIOYuv>     m_cVideoIOYuvReconAfterAlfFile;        ///< reconstruction YUV
#endif
#if NNVC_USE_REC_AFTER_UPSAMPLING
  std::unordered_map<int, VideoIOYuv>     m_cVideoIOYuvReconAfterUpsamplingFile;        ///< reconstruction YUV
#endif
#if NNVC_USE_REC_BEFORE_DBF
  std::unordered_map<int, VideoIOYuv>     m_cVideoIOYuvReconBeforeDbfFile;        ///< reconstruction YUV
#endif
#if NNVC_USE_REC_AFTER_DBF
   std::unordered_map<int, VideoIOYuv>     m_cVideoIOYuvReconAfterDbfFile;        ///< reconstruction YUV
#endif
#if NNVC_USE_PRED
  std::unordered_map<int, VideoIOYuv>      m_cVideoIOYuvPredFile;         ///< prediction
#endif
#if NNVC_USE_PARTITION_AS_CU_AVERAGE
  std::unordered_map<int, VideoIOYuv>      m_cVideoIOYuvCUAverageFile;    ///< partition
#endif
#if NNVC_USE_BS
  std::unordered_map<int, VideoIOYuv>      m_cVideoIOYuvBsMapFile;        ///< bs map
#endif
#if NNVC_USE_QP
  std::ofstream                            m_qpFile;    ///< qp slice
#endif
#if NNVC_USE_SLICETYPE
  std::ofstream                            m_sliceTypeFile;    ///< slice type
#endif
#if JVET_AC0089_NNVC_USE_BPM_INFO
  std::unordered_map<int, VideoIOYuv> m_cVideoIOYuvBpmFile;   ///< Block prediction mode
#endif
#endif

  std::unordered_map<int, VideoIOYuv>      m_cVideoIOYuvReconFile;        ///< reconstruction YUV class
  std::unordered_map<int, VideoIOYuv>      m_videoIOYuvSEIFGSFile;       ///< reconstruction YUV with FGS class
  std::unordered_map<int, VideoIOYuv>      m_cVideoIOYuvSEICTIFile;       ///< reconstruction YUV with CTI class

#if JVET_Z0120_SII_SEI_PROCESSING
  bool                                    m_ShutterFilterEnable;          ///< enable Post-processing with Shutter Interval SEI
  VideoIOYuv                              m_cTVideoIOYuvSIIPostFile;      ///< post-filtered YUV class
  int                                     m_SII_BlendingRatio;

  struct IdrSiiInfo
  {
    SEIShutterIntervalInfo m_siiInfo;
    uint32_t               m_picPoc;
    bool                   m_isValidSii;
  };

  std::map<uint32_t, IdrSiiInfo> m_activeSiiInfo;

#endif

  // for output control
  int             m_iPOCLastDisplay;              ///< last POC in display order
  std::ofstream   m_seiMessageFileStream;         ///< Used for outputing SEI messages.

  std::ofstream   m_oplFileStream;                ///< Used to output log file for confomance testing

  bool            m_newCLVS[MAX_NUM_LAYER_IDS];   ///< used to record a new CLVSS

  SEIAnnotatedRegions::AnnotatedRegionHeader                 m_arHeader; ///< AR header
  std::map<uint32_t, SEIAnnotatedRegions::AnnotatedRegionObject> m_arObjects; ///< AR object pool
  std::map<uint32_t, std::string>                                m_arLabels; ///< AR label pool

private:
  bool  xIsNaluWithinTargetDecLayerIdSet( const InputNALUnit* nalu ) const; ///< check whether given Nalu is within targetDecLayerIdSet
  bool  xIsNaluWithinTargetOutputLayerIdSet( const InputNALUnit* nalu ) const; ///< check whether given Nalu is within targetOutputLayerIdSet

public:
  DecApp();
  virtual ~DecApp         ()  {}

  uint32_t  decode            (); ///< main decoding function
#if JVET_Z0120_SII_SEI_PROCESSING
  bool  getShutterFilterFlag()        const { return m_ShutterFilterEnable; }
  void  setShutterFilterFlag(bool value) { m_ShutterFilterEnable = value; }
  int   getBlendingRatio()             const { return m_SII_BlendingRatio; }
  void  setBlendingRatio(int value) { m_SII_BlendingRatio = value; }
#endif

private:
  void  xCreateDecLib     (); ///< create internal classes
  void  xDestroyDecLib    (); ///< destroy internal classes
  void  xWriteOutput      ( PicList* pcListPic , uint32_t tId); ///< write YUV to file
  void  xFlushOutput( PicList* pcListPic, const int layerId = NOT_VALID ); ///< flush all remaining decoded pictures to file

  // check if next NAL unit will be the first NAL unit from a new picture
  bool isNewPicture(std::ifstream *bitstreamFile, class InputByteStream *bytestream);

  // check if next NAL unit will be the first NAL unit from a new access unit
  bool isNewAccessUnit(bool newPicture, std::ifstream *bitstreamFile, class InputByteStream *bytestream);

  void  writeLineToOutputLog(Picture * pcPic);
  void xOutputAnnotatedRegions(PicList* pcListPic);
};

//! \}

#endif // __DECAPP__

