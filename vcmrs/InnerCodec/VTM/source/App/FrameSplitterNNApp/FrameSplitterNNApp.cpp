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

 /** \file     FrameSplitterNNApp.cpp
     \brief    Frame splitter NN application
 */

#define NN_SEI 1

#include <cstdio>
#include <cctype>
#include <vector>
#include <utility>
#include <fstream>
#include <sstream>
#include <ios>
#include <algorithm>
#include "CommonDef.h"
#include "VLCReader.h"
#include "SEIread.h"
#include "AnnexBread.h"
#include "NALread.h"
#include "Slice.h"
#include "VLCWriter.h"
#include "NALwrite.h"
#include "AnnexBwrite.h"
#include "FrameSplitterNNApp.h"
#if ZJU_BIT_STRUCT
#include "../FrameMixerEncApp/VCMBitStruct.h"
#include "../FrameMixerEncApp/VCMBitStruct.cpp"
#endif


 //! \ingroup FrameSplitterNNApp
 //! \{


struct Subpicture {
  bool                                 firstPic;
  int                                  width;
  int                                  height;
  int                                  topLeftCornerX;
  int                                  topLeftCornerY;
  std::ifstream                        *fp;
  InputByteStream                      *bs;
  bool                                 firstSliceInPicture;
  bool                                 NNPfaSEIEnabled;
  int                                  NNPfaId;
  std::vector<InputNALUnit>            nalus;
  std::vector<AnnexBStats>             stats;
  int                                  prevTid0Poc;
  bool                                 dciPresent;
  DCI                                  dci;
  ParameterSetManager                  psManager;
  std::vector<int>                     vpsIds;
  std::vector<int>                     spsIds;
  std::vector<int>                     ppsIds;
  std::vector<std::pair<int, ApsType>> apsIds;
  PicHeader                            picHeader;
  std::vector<Slice>                   slices;
  std::vector<OutputBitstream>         sliceData;
};


FrameSplitterNNApp::FrameSplitterNNApp(std::vector<SubpicParams> &subpicParams, std::string &outBaseFileName, std::string &outConfigBaseFileName, std::string &outInterMachineAdapterConfigBaseFileName
#if ZJU_BIT_STRUCT
, std::string &outRestorationDataFileName
, std::string &outCodedVideoDataFileName
, std::string &outTemporalRestorationDataFileName
#endif
) :
  m_outBaseFileName(outBaseFileName),
  m_outConfigBaseFileName(outConfigBaseFileName),
  m_outInterMachineAdapterConfigBaseFileName(outInterMachineAdapterConfigBaseFileName),
  m_prevPicPOC(std::numeric_limits<int>::max()),
  m_picWidth(0),
  m_picHeight(0)
#if ZJU_BIT_STRUCT
  , m_outRestorationDataFileName(outRestorationDataFileName)
  , m_outCodedVideoDataFileName(outCodedVideoDataFileName)
  , m_outTemporalRestorationDataFileName(outTemporalRestorationDataFileName)
#endif
{
  m_subpics = new std::vector<Subpicture>;
  m_subpics->resize(subpicParams.size());
  for (int i = 0; i < (int)subpicParams.size(); i++)
  {
    Subpicture &subpic = m_subpics->at(i);
    subpic.width          = subpicParams[i].width;
    subpic.height         = subpicParams[i].height;
    subpic.topLeftCornerX = subpicParams[i].topLeftCornerX;
    subpic.topLeftCornerY = subpicParams[i].topLeftCornerY;
    subpic.fp             = &subpicParams[i].fp;
  }
}

FrameSplitterNNApp::~FrameSplitterNNApp()
{
  delete m_subpics;
}


/**
 - lookahead through next NAL units to determine if current NAL unit is the first NAL unit in a new picture
 */
bool FrameSplitterNNApp::isNewPicture(std::ifstream *bitstreamFile, InputByteStream *bytestream, bool firstSliceInPicture)
{
  bool ret = false;
  bool finished = false;

  // cannot be a new picture if there haven"t been any slices yet
  if(firstSliceInPicture)
  {
    return false;
  }

  // save stream position for backup
#if RExt__DECODER_DEBUG_STATISTICS
  CodingStatistics::CodingStatisticsData* backupStats = new CodingStatistics::CodingStatisticsData(CodingStatistics::GetStatistics());
  std::streampos location = bitstreamFile->tellg() - std::streampos(bytestream->GetNumBufferedBytes());
#else
  std::streampos location = bitstreamFile->tellg();
#endif

  // look ahead until picture start location is determined
  while (!finished && !!(*bitstreamFile))
  {
    AnnexBStats stats = AnnexBStats();
    InputNALUnit nalu;
    byteStreamNALUnit(*bytestream, nalu.getBitstream().getFifo(), stats);
    if (nalu.getBitstream().getFifo().empty())
    {
      msg( ERROR, "Warning: Attempt to decode an empty NAL unit\n");
    }
    else
    {
      // get next NAL unit type
      read(nalu);
      switch( nalu.m_nalUnitType ) {

        // NUT that indicate the start of a new picture
        case NAL_UNIT_ACCESS_UNIT_DELIMITER:
        case NAL_UNIT_OPI:
        case NAL_UNIT_DCI:
        case NAL_UNIT_VPS:
        case NAL_UNIT_SPS:
        case NAL_UNIT_PPS:
        case NAL_UNIT_PH:
          ret = true;
          finished = true;
          break;
        
        // NUT that are not the start of a new picture
        case NAL_UNIT_CODED_SLICE_TRAIL:
        case NAL_UNIT_CODED_SLICE_STSA:
        case NAL_UNIT_CODED_SLICE_RASL:
        case NAL_UNIT_CODED_SLICE_RADL:
        case NAL_UNIT_RESERVED_VCL_4:
        case NAL_UNIT_RESERVED_VCL_5:
        case NAL_UNIT_RESERVED_VCL_6:
        case NAL_UNIT_CODED_SLICE_IDR_W_RADL:
        case NAL_UNIT_CODED_SLICE_IDR_N_LP:
        case NAL_UNIT_CODED_SLICE_CRA:
        case NAL_UNIT_CODED_SLICE_GDR:
        case NAL_UNIT_RESERVED_IRAP_VCL_11:
          ret = checkPictureHeaderInSliceHeaderFlag(nalu);
          finished = true;
          break;

          // NUT that are not the start of a new picture
        case NAL_UNIT_EOS:
        case NAL_UNIT_EOB:
        case NAL_UNIT_SUFFIX_APS:
        case NAL_UNIT_SUFFIX_SEI:
        case NAL_UNIT_FD:
          ret = false;
          finished = true;
          break;
        
        // NUT that might indicate the start of a new picture - keep looking
        case NAL_UNIT_PREFIX_APS:
        case NAL_UNIT_PREFIX_SEI:
        case NAL_UNIT_RESERVED_NVCL_26:
        case NAL_UNIT_RESERVED_NVCL_27:
        case NAL_UNIT_UNSPECIFIED_28:
        case NAL_UNIT_UNSPECIFIED_29:
        case NAL_UNIT_UNSPECIFIED_30:
        case NAL_UNIT_UNSPECIFIED_31:
        default:
          break;
      }
    }
  }
  
  // restore previous stream location - minus 3 due to the need for the annexB parser to read three extra bytes
#if RExt__DECODER_DEBUG_BIT_STATISTICS
  bitstreamFile->clear();
  bitstreamFile->seekg(location);
  bytestream->reset();
  CodingStatistics::SetStatistics(*backupStats);
  delete backupStats;
#else
  bitstreamFile->clear();
  bitstreamFile->seekg(location-std::streamoff(3));
  bytestream->reset();
#endif

  // return TRUE if next NAL unit is the start of a new picture
  return ret;
}

/**
  - Parse DCI
*/
bool FrameSplitterNNApp::parseDCI(HLSyntaxReader &hlsReader, DCI &dci)
{
  hlsReader.parseDCI(&dci);
  msg( INFO, "  DCI");
  return true;
}

/**
  - Parse VPS and store it in parameter set manager
*/
int FrameSplitterNNApp::parseVPS(HLSyntaxReader &hlsReader, ParameterSetManager &psManager)
{
  VPS *vps = new VPS;
  hlsReader.parseVPS(vps);
  int vpsId = vps->getVPSId();
  psManager.storeVPS(vps, hlsReader.getBitstream()->getFifo());
  msg( INFO, "  VPS%i", vpsId);
  return vpsId;
}

/**
  - Parse SPS and store it in parameter set manager
*/
int FrameSplitterNNApp::parseSPS(HLSyntaxReader &hlsReader, ParameterSetManager &psManager)
{
  SPS *sps = new SPS;
  hlsReader.parseSPS(sps);
  int spsId = sps->getSPSId();
  psManager.storeSPS(sps, hlsReader.getBitstream()->getFifo());
  msg( INFO, "  SPS%i", spsId);
  return spsId;
}

/**
  - Parse PPS and store it in parameter set manager
*/
int FrameSplitterNNApp::parsePPS(HLSyntaxReader &hlsReader, ParameterSetManager &psManager)
{
  PPS *pps = new PPS;
  hlsReader.parsePPS(pps);
  int ppsId = pps->getPPSId();
  psManager.storePPS(pps, hlsReader.getBitstream()->getFifo());
  msg( INFO, "  PPS%i", ppsId);
  return ppsId;
}

/**
  - Parse APS and store it in parameter set manager
*/
void FrameSplitterNNApp::parseAPS(HLSyntaxReader &hlsReader, ParameterSetManager &psManager, int &apsId, int &apsType)
{
  APS *aps = new APS;
  hlsReader.parseAPS(aps);
  apsId = aps->getAPSId();
  apsType = (int)aps->getAPSType();
  psManager.storeAPS(aps, hlsReader.getBitstream()->getFifo());
  msg( INFO, "  APS%i", apsId);
}

/**
  - Parse picture header
*/
void FrameSplitterNNApp::parsePictureHeader(HLSyntaxReader &hlsReader, PicHeader &picHeader, ParameterSetManager &psManager)
{
  hlsReader.parsePictureHeader(&picHeader, &psManager, true);
  picHeader.setValid();
  msg( INFO, "  PH");
}

/**
  - Parse slice header and store slice data
*/
void FrameSplitterNNApp::parseSliceHeader(HLSyntaxReader &hlsReader, InputNALUnit &nalu, Slice &slice, PicHeader &picHeader, OutputBitstream &sliceData, ParameterSetManager &psManager, int prevTid0Poc)
{
  slice.initSlice();
  slice.setNalUnitType(nalu.m_nalUnitType);
  slice.setTLayer(nalu.m_temporalId);
  slice.setPicHeader(&picHeader);
  hlsReader.parseSliceHeader(&slice, &picHeader, &psManager, prevTid0Poc, m_prevPicPOC);
  slice.setPPS(psManager.getPPS(picHeader.getPPSId()));
  slice.setSPS(psManager.getSPS(picHeader.getSPSId()));

  InputBitstream &inBs = nalu.getBitstream();
  CHECK(inBs.getNumBitsLeft() & 7, "Slicedata must be byte aligned");
  int numDataBytes = inBs.getNumBitsLeft() / 8;
  for (int i = 0; i < numDataBytes; i++ )
  {
    sliceData.write(inBs.readByte(), 8);
  }

  msg( INFO, "  VCL%i", slice.getPOC());
}

/**
  - Parse slice header and store slice data
*/
void FrameSplitterNNApp::parseSliceHeaderNN(HLSyntaxReader &hlsReader, InputNALUnit &nalu, Slice &slice, PicHeader &picHeader, OutputBitstream &sliceData, ParameterSetManager &psManager, int prevTid0Poc)
{
  slice.initSlice();
  slice.setNalUnitType(nalu.m_nalUnitType);
  slice.setTLayer(nalu.m_temporalId);
  slice.setPicHeader(&picHeader);
  hlsReader.parseSliceHeaderNN(&slice, &picHeader, &psManager, prevTid0Poc, m_prevPicPOC);
  slice.setPPS(psManager.getPPS(picHeader.getPPSId()));
  slice.setSPS(psManager.getSPS(picHeader.getSPSId()));

  InputBitstream &inBs = nalu.getBitstream();
  CHECK(inBs.getNumBitsLeft() & 7, "Slicedata must be byte aligned");
  int numDataBytes = inBs.getNumBitsLeft() / 8;
  for (int i = 0; i < numDataBytes; i++ )
  {
    sliceData.write(inBs.readByte(), 8);
  }

  msg( INFO, "  VCL%i", slice.getPOC());
}

void FrameSplitterNNApp::parseSEIMessage(Subpicture &subpic, InputNALUnit &nalu)
{
  SEIReader *seiReader = new SEIReader;
  HRD *hrd = new HRD;
  SEIMessages seiMessages;
  std::ostringstream oss;
  std::ostream *os = &oss;

  seiReader->parseSEImessage(&(nalu.getBitstream()), seiMessages, nalu.m_nalUnitType, nalu.m_nuhLayerId, nalu.m_temporalId, subpic.psManager.getVPS(subpic.psManager.getActiveSPS()->getVPSId()), subpic.psManager.getActiveSPS(), *hrd, os);

  SEIMessages nnpfaSEI = getSeisByType(seiMessages, SEI::PayloadType::NEURAL_NETWORK_POST_FILTER_ACTIVATION);
  if (!nnpfaSEI.empty())
  {
    SEINeuralNetworkPostFilterActivation *sei = dynamic_cast<SEINeuralNetworkPostFilterActivation*>(*(nnpfaSEI.begin()));
    subpic.NNPfaId = sei->m_targetId;
    subpic.NNPfaSEIEnabled = true;
  }

  delete hrd;
  delete seiReader;
}

/**
  - Decode NAL unit if it is parameter set or picture header, or decode slice header of VLC NAL unit
 */
void FrameSplitterNNApp::decodeNalu(Subpicture &subpic, InputNALUnit &nalu)
{
  HLSyntaxReader hlsReader;
  hlsReader.setBitstream(&nalu.getBitstream());
  int apsId;
  int apsType;

  switch (nalu.m_nalUnitType)
  {
  case NAL_UNIT_DCI:
    subpic.dciPresent = parseDCI(hlsReader, subpic.dci);
    break;
  case NAL_UNIT_VPS:
    subpic.vpsIds.push_back(parseVPS(hlsReader, subpic.psManager));
    break;
  case NAL_UNIT_SPS:
    subpic.spsIds.push_back(parseSPS(hlsReader, subpic.psManager));
    break;
  case NAL_UNIT_PPS:
    subpic.ppsIds.push_back(parsePPS(hlsReader, subpic.psManager));
    if (subpic.firstPic)
    {
      subpic.psManager.activatePPS(subpic.ppsIds.back(), true);  // Avoid potential crash due to non-active pps before slice header is parsed
    }
    break;
  case NAL_UNIT_PREFIX_APS:
    parseAPS(hlsReader, subpic.psManager, apsId, apsType);
    subpic.apsIds.push_back(std::pair<int, ApsType>(apsId, (ApsType)apsType));
    break;
  case NAL_UNIT_PH:
    parsePictureHeader(hlsReader, subpic.picHeader, subpic.psManager);
  break;
  default:
    if (nalu.isVcl())
    {
      subpic.slices.emplace_back();
      subpic.sliceData.emplace_back();
      if (nalu.m_nalUnitType == NAL_UNIT_CODED_SLICE_NN_IRAP)
      {
        parseSliceHeaderNN(hlsReader, nalu, subpic.slices.back(), subpic.picHeader, subpic.sliceData.back(), subpic.psManager, subpic.prevTid0Poc);
      }
      else
      {
        parseSliceHeader(hlsReader, nalu, subpic.slices.back(), subpic.picHeader, subpic.sliceData.back(), subpic.psManager, subpic.prevTid0Poc);
      }
      if (subpic.slices.size() == 1) // Is first slice of pic?
      {
        subpic.psManager.activatePPS(subpic.slices.begin()->getPPS()->getPPSId(), subpic.slices.begin()->isIRAP());
      }
    }
    else if (nalu.isSei())
    {
      msg( INFO, "  SEI");
      parseSEIMessage(subpic, nalu);
    }
    else
    {
      msg( INFO, "  NNN");  // Any other NAL unit that is not handled above
    }
    break;
  }
}


/**
  - Parse NAL units of one subpicture
 */
void FrameSplitterNNApp::parseSubpic(Subpicture &subpic, bool &morePictures)
{
  subpic.nalus.clear();
  subpic.stats.clear();
  subpic.dciPresent = false;
  subpic.vpsIds.clear();
  subpic.spsIds.clear();
  subpic.ppsIds.clear();
  subpic.apsIds.clear();
  subpic.picHeader.initPicHeader();
  subpic.slices.clear();
  subpic.sliceData.clear();
  subpic.firstSliceInPicture = true;
  subpic.NNPfaSEIEnabled = false;
  subpic.NNPfaId = 0;

  bool eof = false;

  while (!eof && !isNewPicture(subpic.fp, subpic.bs, subpic.firstSliceInPicture))
  {
    subpic.nalus.emplace_back();  // Add new nalu
    subpic.stats.emplace_back();  // Add new stats
    InputNALUnit &nalu = subpic.nalus.back();
    AnnexBStats &stats = subpic.stats.back();
    nalu.m_nalUnitType = NAL_UNIT_INVALID;

    // find next NAL unit in stream
    eof = byteStreamNALUnit(*subpic.bs, nalu.getBitstream().getFifo(), stats);

    if (eof)
    {
      morePictures = false;
    }

    if (nalu.getBitstream().getFifo().empty())
    {
      subpic.nalus.pop_back();  // Remove empty nalu
      subpic.stats.pop_back();
      msg( ERROR, "Warning: Attempt to decode an empty NAL unit\n");
      continue;
    }

    read(nalu);  // Convert nalu payload to RBSP and parse nalu header
    decodeNalu(subpic, nalu);

    if (nalu.isVcl())
    {
      subpic.firstSliceInPicture = false;
    }
  }
}


/**
  - Create merged stream VPSes
*/
void FrameSplitterNNApp::generateMergedStreamVPSes(std::vector<VPS*> &vpsList)
{
  for (auto vpsId : m_subpics->at(0).vpsIds)
  {
    // Create new SPS based on the SPS from the first subpicture 
    vpsList.push_back(new VPS(*m_subpics->at(0).psManager.getVPS(vpsId)));
    VPS &vps = *vpsList.back();

    for (int i = 0; i < vps.getNumOutputLayerSets(); i++)
    {
      vps.setOlsDpbPicWidth(i, m_picWidth);
      vps.setOlsDpbPicHeight(i, m_picHeight);
    }
  }
}


/**
  - Create merged stream SPSes with subpicture information
*/
void FrameSplitterNNApp::generateMergedStreamSPSes(std::vector<SPS*> &spsList)
{
  for (auto &subpic : *m_subpics)
  {
    for (auto spsId : subpic.spsIds)
    {
      CHECK(subpic.psManager.getSPS(spsId)->getSubPicInfoPresentFlag(), "Input streams containing subpictures not supported")
    }
  }

  for (auto spsId : m_subpics->at(0).spsIds)
  {
    // Create new SPS based on the SPS from the first subpicture 
    spsList.push_back(new SPS(*m_subpics->at(0).psManager.getSPS(spsId)));
    SPS &sps = *spsList.back();

    sps.setMaxPicWidthInLumaSamples(m_picWidth);
    sps.setMaxPicHeightInLumaSamples(m_picHeight);
  }

}


/**
  - Create merged stream PPSes based on the first subpicture PPSes
*/
void FrameSplitterNNApp::generateMergedStreamPPSes(ParameterSetManager &, std::vector<PPS*> &)
{
  return;
}

/**
  - Configure slice headers of all subpicture for merged stream
*/
void FrameSplitterNNApp::updateSliceHeadersForMergedStream(ParameterSetManager &psManager)
{
  for (auto &subpic : *m_subpics)
  {
    for (auto &slice : subpic.slices)
    {
      // Update slice headers to use new SPSes and PPSes
      int ppsId = slice.getPPS()->getPPSId();
      int spsId = slice.getSPS()->getSPSId();
      CHECK(!psManager.getSPS(spsId), "Invaldi SPS");
      CHECK(!psManager.getSPS(ppsId), "Invaldi PPS");
      slice.setSPS(psManager.getSPS(spsId));
      slice.setPPS(psManager.getPPS(ppsId));
    }
  }
}

/**
  - Copy input NAL unit to ouput NAL unit
*/
void FrameSplitterNNApp::copyInputNaluToOutputNalu(OutputNALUnit &outNalu, InputNALUnit &inNalu)
{
  // Copy nal header info
  outNalu = inNalu;

  // Copy payload
  std::vector<uint8_t> &inFifo = inNalu.getBitstream().getFifo();
  std::vector<uint8_t> &outFifo = outNalu.m_bitstream.getFifo();
  outFifo = std::vector<uint8_t>(inFifo.begin() + 2, inFifo.end());
}

/**
  - Copy NAL unit with NAL unit type naluType to access unit
*/
void FrameSplitterNNApp::copyNalUnitsToAccessUnit(AccessUnit &accessUnit, std::vector<InputNALUnit> &nalus, int naluType)
{
  for (auto &inNalu : nalus)
  {
    if (inNalu.m_nalUnitType == (NalUnitType)naluType)
    {
      if (!(naluType == NAL_UNIT_SUFFIX_SEI && SEI::PayloadType(inNalu.getBitstream().getFifo().at(2)) == SEI::PayloadType::DECODED_PICTURE_HASH))  // Don"t copy decoded_picture_hash SEI
      {
        OutputNALUnit outNalu((NalUnitType)naluType);
        copyInputNaluToOutputNalu(outNalu, inNalu);
        accessUnit.push_back(new NALUnitEBSP(outNalu));
      }
    }
  }
}


/**
  - Write NAL units for one picture
 */
void FrameSplitterNNApp::writeOnePic()
{
  AccessUnit accessUnit;

  for (auto& subpic : *m_subpics)
  {
    for (auto& nalu : subpic.nalus)
    {
      OutputNALUnit outNalu((NalUnitType)nalu.m_nalUnitType);
      copyInputNaluToOutputNalu(outNalu, nalu);
      accessUnit.push_back(new NALUnitEBSP(outNalu));
    }
  }

  writeAnnexBAccessUnit(m_outputStream, accessUnit);
}


/**
  - Create file name
*/
void FrameSplitterNNApp::generateFilename(std::string &generatedFilename, std::string &baseFilename, int n, const char* suffix)
{
  generatedFilename = baseFilename;

  std::string picNumString = std::to_string(n);
  for (auto i = picNumString.length(); i < 6; i++)
  {
    generatedFilename.append("0");
  }
  generatedFilename.append(picNumString);

  generatedFilename.append(suffix);
}


/**
  - Merge subpicture bitstreams into one bitstream
 */
void FrameSplitterNNApp::splitFrames()
{
  ParameterSetManager psManager;  // Parameter sets for merged stream
  int picNum = 0;
  int intraPicNum = 0;

  // msg( INFO, "Output picture size is %ix%i\n", m_picWidth, m_picHeight);

  for (auto &subpic : *m_subpics)
  {
    subpic.firstPic = true;
    subpic.bs = new InputByteStream(*(subpic.fp));
    subpic.prevTid0Poc = 0;
    subpic.psManager.storeVPS(new VPS, std::vector<uint8_t>());  // Create VPS with default values (VTM slice header parser needs this)
  }

  bool morePictures = true;
  while (morePictures)
  {
    msg( INFO, "Picture %i\n", picNum);
    
    // int subPicNum = 0;

    for (auto &subpic : *m_subpics)
    {
      // msg( INFO, " Subpicture %i\n", subPicNum);
      parseSubpic(subpic, morePictures);
      // subPicNum++;
      msg( INFO, "\n");
      if (picNum == 0)
      { 
        const SPS* sps = m_subpics->at(0).slices.at(0).getSPS();
        int width = sps->getMaxPicWidthInLumaSamples();
        int height = sps->getMaxPicHeightInLumaSamples();
        if (sps->getConformanceWindow().getWindowEnabledFlag()) {
          const int subWidthC  = SPS::getWinUnitX(sps->getChromaFormatIdc());
          width = width - subWidthC*sps->getConformanceWindow().getWindowLeftOffset() - 
              subWidthC*sps->getConformanceWindow().getWindowRightOffset();
          const int subHeightC  = SPS::getWinUnitY(sps->getChromaFormatIdc());
          height = height - subHeightC*sps->getConformanceWindow().getWindowTopOffset() - 
              subWidthC*sps->getConformanceWindow().getWindowBottomOffset();
        }
        msg(INFO, "width %i\n", width);
        msg(INFO, "height %i\n", height);
      }
    }

    // validateSubpics();

    // Write NN picture data to a file
    if (m_subpics->at(0).slices.at(0).isIntra() && !m_outBaseFileName.empty())
    {
      std::string outFileName;

      generateFilename(outFileName, m_outBaseFileName, intraPicNum, ".bin");
      m_outputStream.open(outFileName, std::ios_base::binary);
      if (!m_outputStream.is_open())
      {
        std::cerr << "Error: cannot open output file " << outFileName << " for writing" << std::endl;
        return;
      }

      const std::vector<uint8_t> &sliceData = m_subpics->at(0).sliceData.at(0).getFifo();
      m_outputStream.write(reinterpret_cast<const char *>(sliceData.data()), sliceData.size());
      m_outputStream.close();
    }

    // Write NN picture config params to a file
    if (m_subpics->at(0).slices.at(0).isIntra() && !m_outConfigBaseFileName.empty())
    {
      std::string outConfigFileName;

      generateFilename(outConfigFileName, m_outConfigBaseFileName, intraPicNum, ".txt");
      m_outputConfigStream.open(outConfigFileName);
      if (!m_outputConfigStream.is_open())
      {
        std::cerr << "Error: cannot open output config file " << outConfigFileName << " for writing" << std::endl;
        return;
      }

      m_outputConfigStream << "intra_model_id " << m_subpics->at(0).slices.at(0).getNNIrapModelIdc() << std::endl;
      m_outputConfigStream << "intra_filter_flag " << (m_subpics->at(0).slices.at(0).getNNIrapFilterFlag() ? 1 : 0) << std::endl;
      m_outputConfigStream << "intra_filter_patch_wise_flag " << (m_subpics->at(0).slices.at(0).getNNIrapFilterPatchWiseFlag() ? 1 : 0) << std::endl;
      if (m_subpics->at(0).slices.at(0).getNNIrapFilterPatchWiseFlag())
      {
        m_outputConfigStream << "intra_filter_patch_size " << (m_subpics->at(0).slices.at(0).getNNIrapFilterPatchSize() ? 1 : 0) << std::endl;
        if (m_subpics->at(0).slices.at(0).getNNIrapFilterPatchFlags().size() > 0)
        {
          m_outputConfigStream << "intra_filter_patch_flags";
          for (auto patchFlag : m_subpics->at(0).slices.at(0).getNNIrapFilterPatchFlags())
          {
            m_outputConfigStream << " " << patchFlag;
          }
          m_outputConfigStream << std::endl;
        }
      }

      m_outputConfigStream.close();
    }

    // Write NN inter machine adapter config params to a file
    if (m_subpics->at(0).NNPfaSEIEnabled && !m_outInterMachineAdapterConfigBaseFileName.empty())
    {
      std::string outConfigFileName;

      generateFilename(outConfigFileName, m_outInterMachineAdapterConfigBaseFileName, m_subpics->at(0).slices.at(0).getPOC(), ".txt");
      m_outputConfigStream.open(outConfigFileName);
      if (!m_outputConfigStream.is_open())
      {
        std::cerr << "Error: cannot open inter machine adapter output config file " << outConfigFileName << " for writing" << std::endl;
        return;
      }

      m_outputConfigStream << "nnpfa_id " << m_subpics->at(0).NNPfaId << std::endl;
      m_outputConfigStream.close();
    }

    // writeOnePic();


    // generateMergedPic(psManager, false);

    // Update prevTid0Poc flags for subpictures
    for (auto &subpic : *m_subpics)
    {
      if (subpic.slices.size() > 0 && subpic.slices[0].getTLayer() == 0 &&
          subpic.slices[0].getNalUnitType() != NAL_UNIT_CODED_SLICE_RADL &&
          subpic.slices[0].getNalUnitType() != NAL_UNIT_CODED_SLICE_RASL )
      {
        subpic.prevTid0Poc = subpic.slices[0].getPOC();
      }
      subpic.firstPic = false;
    }

    m_prevPicPOC = m_subpics->at(0).slices.at(0).getPOC();

    picNum++;
    if (m_subpics->at(0).slices.at(0).isIntra())
    {
      intraPicNum++;
    }
  }
}

#if ZJU_BIT_STRUCT

void parseSampleStream(InputBitstream &inBs, InputBitstream &outBs)
{
  uint32_t numBytes;
  uint32_t zeroBits;
  uint32_t valueBits;
  uint32_t lengthSubStream;
  InputBitstream *tmpBs;

  inBs.read(3, valueBits);
  numBytes = valueBits + 1;
  inBs.read(5, zeroBits);
  CHECK(zeroBits != 0, "Found non-zero sample stream");
  inBs.read(numBytes * 8, lengthSubStream);
  tmpBs = inBs.extractSubstream(lengthSubStream * 8);

  outBs.getFifo().insert(outBs.getFifo().end(), tmpBs->getFifo().begin(), tmpBs->getFifo().end());
}

void writeRsdToFile(std::ofstream &outRsdStream, RSDInfo &rsdInfo)
{
  OutputBitstream rsdBitstream;
  OutputBitstream subBitstream;
  RSDCompMap CompType;
  uint32_t sizeSubStream = 0;

  rsdBitstream.write(rsdInfo.spatial_resampling_simple_flag, 8);
  rsdBitstream.write(0, 8);

  // collect rsd data in bin format.
  for (uint16_t i = 0; i < COMP_INVALID; i++)
  {
    // write data type.
    bool compExist = false;
    CompType = RSDCompMap(i);
    if (CompType == COMP_PostFilter || CompType == COMP_InnerCodec)
    {
      continue;
    }
    // write data detail to sub bitstream.
    if (CompType == COMP_TemporalResample && rsdInfo.temporal_restoration_flag)
    {
      compExist = true;
      subBitstream.write(rsdInfo.m_trdRatioId, 8);
      subBitstream.write(rsdInfo.m_trdNumPicToBeRecon, 16);
    }
    else if (CompType == COMP_BitDepthTruncation && rsdInfo.bit_depth_on_flag)
    {
      compExist = true;
      subBitstream.write(rsdInfo.bit_depth_shift_flag, 8);
      subBitstream.write(rsdInfo.bit_depth_shift_luma, 8);
      subBitstream.write(rsdInfo.bit_depth_shift_chroma, 8);
      subBitstream.write(rsdInfo.bit_depth_luma_enhance, 8);
    }
    else if (CompType == COMP_SpatialResample && rsdInfo.spatial_on_flag)
    {
      compExist = true;
      if (!rsdInfo.spatial_resampling_simple_flag)
      {
        // Adaptive downsample.
        subBitstream.write(rsdInfo.vcm_spatial_resampling_flag, 8);
        if (rsdInfo.vcm_spatial_resampling_flag)
        {
          subBitstream.write(rsdInfo.spatial_resample_width >> 8, 8);
          subBitstream.write(rsdInfo.spatial_resample_width & 0xFF, 8);
          subBitstream.write(rsdInfo.spatial_resample_height >> 8, 8);
          subBitstream.write(rsdInfo.spatial_resample_height & 0xFF, 8);
          subBitstream.write(rsdInfo.spatial_resample_filter_idx, 8);
        }
      }
      else
      {
        subBitstream.write(rsdInfo.scale_factor_id, 8);
      }
    }
    else if (CompType == COMP_Colorize && rsdInfo.colorizer_on_flag)
    {
      compExist = true;
      subBitstream.write(rsdInfo.colorizer_period&0xFF, 8);
      subBitstream.write((rsdInfo.colorizer_period>>8)&0xFF, 8);
      rsdInfo.colorizer_gops_num = rsdInfo.colorizerInfo.size();
      subBitstream.write(rsdInfo.colorizer_gops_num&0xFF, 8);
      subBitstream.write((rsdInfo.colorizer_gops_num>>8)&0xFF, 8);
      for (size_t i = 0; i < rsdInfo.colorizer_gops_num; i++)
      {
          ColorizerInfo tmpColorizerInfo = rsdInfo.colorizerInfo.front();
          subBitstream.write(tmpColorizerInfo.colorizer_enable_flag, 8);
          if (tmpColorizerInfo.colorizer_enable_flag)
          {
              subBitstream.write(tmpColorizerInfo.colorizer_index, 8);
              //subBitstream.write(tmpColorizerInfo.luma_pre_shift, 8);
          }
          
          rsdInfo.colorizerInfo.pop_front();
      }
    }
    else if (CompType == COMP_ROI && rsdInfo.roi_on_flag)
    {
      compExist = true;
      subBitstream.write(rsdInfo.roi_update_period_len, 5);
      subBitstream.write(rsdInfo.roi_update_period, rsdInfo.roi_update_period_len); // roi_update_period - 1
      rsdInfo.roi_gops_num = rsdInfo.roiInfo.size();
      uint32_t numBits = ceil(log2(rsdInfo.roi_gops_num + 1));
      subBitstream.write(numBits, 5);
      subBitstream.write(rsdInfo.roi_gops_num, numBits);

      for (size_t i = 0; i < rsdInfo.roi_gops_num; i++)
      {
        ROIInfo tmpRoiInfo = rsdInfo.roiInfo.front();
        subBitstream.write(tmpRoiInfo.rtg_image_size_len, 5);
        subBitstream.write(tmpRoiInfo.rtg_image_size_width, tmpRoiInfo.rtg_image_size_len);
        subBitstream.write(tmpRoiInfo.rtg_image_size_height, tmpRoiInfo.rtg_image_size_len);

        uint32_t orgWidth = tmpRoiInfo.rtg_image_size_width;
        uint32_t orgHeight = tmpRoiInfo.rtg_image_size_height;

        subBitstream.write(tmpRoiInfo.rtg_image_size_difference_flag, 1);
        if (tmpRoiInfo.rtg_image_size_difference_flag)
        {
          subBitstream.write(tmpRoiInfo.rtg_to_output_difference_len, 5);
          subBitstream.write(tmpRoiInfo.rtg_to_output_difference_width, tmpRoiInfo.rtg_to_output_difference_len);
          subBitstream.write(tmpRoiInfo.rtg_to_output_difference_height, tmpRoiInfo.rtg_to_output_difference_len);
          orgWidth += tmpRoiInfo.rtg_to_output_difference_width;
          orgHeight += tmpRoiInfo.rtg_to_output_difference_height;
        }

        uint32_t MAXIMAL_SCALE_FACTOR = 15;
        subBitstream.write(tmpRoiInfo.rtg_rois_flag, 1);
        if (tmpRoiInfo.rtg_rois_flag)
        {
          subBitstream.write(tmpRoiInfo.roi_size_len, 5);
          subBitstream.write(tmpRoiInfo.num_rois_len, 4);
          subBitstream.write(tmpRoiInfo.num_rois, tmpRoiInfo.num_rois_len);
          
          uint32_t current_roi_scale_factor = 0;
          uint32_t bitsPos = ceil(log2((orgWidth > orgHeight ? orgWidth: orgHeight) + 1));
          uint32_t bitsScaleFactor = ceil(log2(MAXIMAL_SCALE_FACTOR + 1));
          for (size_t i = 0; i < tmpRoiInfo.num_rois; i++)
          {
            if (current_roi_scale_factor != MAXIMAL_SCALE_FACTOR)
            {
              subBitstream.write(tmpRoiInfo.roi_scale_factor_flag[i], 1);
              if (tmpRoiInfo.roi_scale_factor_flag[i])
              {
                subBitstream.write(tmpRoiInfo.roi_scale_factor[i], bitsScaleFactor);
              }
            }
            subBitstream.write(tmpRoiInfo.roi_pos_x[i], bitsPos);
            subBitstream.write(tmpRoiInfo.roi_pos_y[i], bitsPos);
            subBitstream.write(tmpRoiInfo.roi_size_x[i], tmpRoiInfo.roi_size_len);
            subBitstream.write(tmpRoiInfo.roi_size_y[i], tmpRoiInfo.roi_size_len);
          }
        }
        rsdInfo.roiInfo.pop_front();
      }
      subBitstream.writeByteAlignment();
    }
    // write size and data of sub bitstream
    if (compExist)
    {
      rsdBitstream.write(i, 8);
      sizeSubStream = subBitstream.getNumberOfWrittenBits()/8;
      rsdBitstream.write(sizeSubStream & 0xFF, 8);
      rsdBitstream.write(sizeSubStream >> 8, 8);
      rsdBitstream.addSubstream(&subBitstream);
      subBitstream.clear();
    }
  }
  // add end.
  rsdBitstream.write(0xFF, 8);
  std::vector<uint8_t> outRsd = rsdBitstream.getFifo();
  outRsdStream.write(reinterpret_cast<const char *>(outRsd.data()), outRsd.size());
}

/**
  - Merge subpicture bitstreams into one bitstream
 */
void FrameSplitterNNApp::splitVCMBitstream()
{
  InputBitstream inBs;

  std::cout << "Read all bytes" << std::endl;
  std::ifstream *frd = m_subpics->back().fp;
  frd->seekg(0, std::ios_base::end);
  int allByteSize = (int)frd->tellg();
  frd->seekg(0, std::ios_base::beg);
  std::vector<uint8_t> allBytes(allByteSize);
  frd->read(reinterpret_cast<char*>(allBytes.data()), allByteSize);
  frd->close();
  inBs.getFifo() = allBytes;

  /*
    parse sample stream.
  */
  std::cout << "Parsing VCM units sample stream" << std::endl;
  uint32_t remainSize = inBs.getNumBitsLeft();
  std::list<VCMUnit*> vcmUnits;
  while (remainSize > 0)
  {
    vcmUnits.push_back(new VCMUnit());
    parseSampleStream(inBs, vcmUnits.back()->m_inBitstream);
    vcmUnits.back()->parseHeader();
    remainSize = inBs.getNumBitsLeft();
  }

  /*
    parse VCM units.
  */
  std::cout << "Parsing VCM units" << std::endl;
  InputBitstream bsVcmNalu;
  OutputBitstream bsVCMCvd;
  VCMUnit *storeVCMPS = NULL;
  for (auto vcmu: vcmUnits)
  {
      switch (vcmu->m_vcmUnitType)
    {
    case VCM_VPS:
      vcmu->parseVCMPS();
      storeVCMPS = vcmu;
      break;
    case VCM_RSD:
      vcmu->parseRSD(&bsVcmNalu);
      break;
    case VCM_CVD:
      vcmu->parseCVD(bsVCMCvd);
      break;
    
    default:
      break;
    }
  }

  /*
    parse VCM NAL units.
  */
  std::cout << "Parsing VCM NAL units" << std::endl;
  std::list<VCMNalu*> vcmNalUnits;
  remainSize = bsVcmNalu.getNumBitsLeft();
  while (remainSize > 0)
  {
    vcmNalUnits.push_back(new VCMNalu());
    parseSampleStream(bsVcmNalu, vcmNalUnits.back()->m_inBitstream);
    vcmNalUnits.back()->parseVCMNaluHeader();
    remainSize = bsVcmNalu.getNumBitsLeft();
  }

  VCMNalu *storeSRD = NULL;
  std::vector<uint8_t> outRsd;
  RSDInfo rsdInfo;
  uint32_t countFrame = 0;
  // struct TRDInfo
  // {
  //   uint32_t m_picPoc;
  //   bool m_isDecodedPic;
  //   std::vector<uint32_t> m_refPicPocList;
  // };
  // std::vector<TRDInfo*> outTrdInfo;
  for (auto vcmNalu: vcmNalUnits) 
  {
    switch (vcmNalu->m_nalUnitType)
    {
    case VCM_NAL_SRD:
      std::cout << "Parsing VCM NAL unit SRD" << std::endl;
      vcmNalu->m_refVPS = storeVCMPS; //temporary solution. should based on the ID.
      vcmNalu->parseSRD(rsdInfo);
      storeSRD = vcmNalu;
      break;
    case VCM_NAL_PRD:
      vcmNalu->m_refSRD = storeSRD; // temporary solution. should based on the ID.
      vcmNalu->parsePRD(rsdInfo);
      countFrame++;
      std::cout << "Parsing VCM NAL unit PRD " << vcmNalu->m_prdPocLsb << std::endl;
      // outTrdInfo.push_back(new TRDInfo);
      // outTrdInfo.back()->m_isDecodedPic = vcmNalu->m_prdDecodedPicFlag;
      // outTrdInfo.back()->m_picPoc = vcmNalu->m_prdPocLsb;
      // if (!vcmNalu->m_prdDecodedPicFlag)
      // {
      //   outTrdInfo.back()->m_refPicPocList.resize(vcmNalu->m_prdTrd.m_numRefPic);
      //   for (uint32_t i = 0; i < vcmNalu->m_prdTrd.m_numRefPic; i++)
      //   {
      //     outTrdInfo.back()->m_refPicPocList[i] = vcmNalu->m_prdPocLsb + vcmNalu->m_prdTrd.m_deltaPOC[i];
      //   }
      // }
      break;
    case VCM_NAL_SEI:
      std::cout << "Parsing VCM NAL unit SEI" << std::endl;
      vcmNalu->parseSEI();
      break;
    case VCM_NAL_EOSS:
      std::cout << "Parsing VCM NAL unit EOSS" << std::endl;
      CHECK(vcmNalu != vcmNalUnits.back(), "Found EOSS VCM NAL unit not the last one");
      vcmNalu->m_refVPS = storeVCMPS; //temporary solution. should based on the ID.
      vcmNalu->parseEOSS(rsdInfo);
      break;
    default:
      THROW("Found unexpected VCM NAL unit type");
      break;
    }
  }
  /*
    write rsd file.
  */
  std::cout << "Writing restoration data" << std::endl;
  if (!m_outRestorationDataFileName.empty())
  {
    m_outRsdStream.open(m_outRestorationDataFileName, std::ios_base::binary);
    if (!m_outRsdStream.is_open())
    {
      std::cerr << "Error: cannot open output restoration data file " << m_outRestorationDataFileName << " for writing" << std::endl;
      return;
    }
    // set tool flag for rsd file writting.
    rsdInfo.spatial_on_flag = storeVCMPS->m_vpsSpatialFlag;
    rsdInfo.roi_on_flag = storeVCMPS->m_vpsRetargetFlag;
    rsdInfo.colorizer_on_flag = storeVCMPS->m_vpsColorizerFlag;
    rsdInfo.temporal_restoration_flag = storeVCMPS->m_vpsTemporalFlag;
    rsdInfo.bit_depth_on_flag = storeVCMPS->m_vpsBitDepthShiftFlag;
    writeRsdToFile(m_outRsdStream, rsdInfo);
    m_outRsdStream.close();
  }

  /*
    write cvd file.
  */
  std::cout << "Writing coded video data" << std::endl;
  if (!m_outCodedVideoDataFileName.empty())
  {
    m_outCvdStream.open(m_outCodedVideoDataFileName, std::ios_base::binary);
    if (!m_outCvdStream.is_open())
    {
      std::cerr << "Error: cannot open output restoration data file " << m_outCodedVideoDataFileName << " for writing" << std::endl;
      return;
    }

    std::vector<uint8_t> outCvd = bsVCMCvd.getFifo();
    m_outCvdStream.write(reinterpret_cast<const char *>(outCvd.data()), outCvd.size());
    m_outCvdStream.close();
  }

  /*
    write trd file.
  */
  if(!m_outTemporalRestorationDataFileName.empty())
  {
    FILE*  file     = fopen(m_outTemporalRestorationDataFileName.c_str(), "r");
    if (file)
    {
      fclose(file);
      remove(m_outTemporalRestorationDataFileName.c_str());
    }
    std::ofstream outfile(m_outTemporalRestorationDataFileName, std::ios::app);
    if (!outfile)
    {
      std::cout << " TEMPAR_OUT.cfg file open failed" << std::endl;
    }
    outfile << "VCMEnabled:" << rsdInfo.m_vcmExtensionEnabledFlag << std::endl;
    outfile << "TemporalEnabled:" << rsdInfo.temporal_restoration_flag << std::endl;
    outfile << "TemporalRemain:" << rsdInfo.m_numTemporalRemain << std::endl;
#ifdef TEM_MPEG_148
    outfile << "srdTemporalRestorationMode:" << rsdInfo.m_trdMode << std::endl;
    if (rsdInfo.m_trdMode == 0)
    {
      outfile << "srdTemporalInterpolationRatioId:" << rsdInfo.m_trd_inter_ratio_id << std::endl;
    }
    else if (rsdInfo.m_trdMode == 1)
    {
      outfile << "srdTemporalExtraResamplingNumId:" << rsdInfo.m_trd_extra_resample_num_id << std::endl;
      outfile << "srdTemporalExtraPredictNumId:" << rsdInfo.m_trd_extra_predict_num_id << std::endl;
    }
    // sort the flags and indexes
    size_t   len_flags = rsdInfo.m_picTemporalChangedPOCs.size();
    std::map<uint32_t, bool>     tmpFlagVec;
    std::map<uint32_t, uint32_t> tmpModeVec;
    std::map<uint32_t, uint32_t> tmpIndexVec;
    std::map<uint32_t, uint32_t> tmpResampleNumVec;
    std::map<uint32_t, uint32_t> tmpPredictNumVec;

    int modeId = 0;
    int ratioId = 0;
    int resampleId = 0;
    int predictId  = 0;
    for (size_t i = 0; i < len_flags; i++)
    {
      CHECK(rsdInfo.m_picTemporalChangedPOCs[i] >= len_flags, "POC value for temporal exceeds the list length");
      tmpFlagVec[rsdInfo.m_picTemporalChangedPOCs[i]]  = rsdInfo.m_trd_pic_changed_flags[i];
      if (tmpFlagVec[rsdInfo.m_picTemporalChangedPOCs[i]]) 
      {
        tmpModeVec[rsdInfo.m_picTemporalChangedPOCs[i]] = rsdInfo.m_trd_pic_modes[modeId++];
        if (tmpModeVec[rsdInfo.m_picTemporalChangedPOCs[i]] == 0) 
        {
          tmpIndexVec[rsdInfo.m_picTemporalChangedPOCs[i]] = rsdInfo.m_trd_pic_inter_ratio_ids[ratioId++];
        }
        else if (tmpModeVec[rsdInfo.m_picTemporalChangedPOCs[i]] == 1) 
        {
          tmpResampleNumVec[rsdInfo.m_picTemporalChangedPOCs[i]] = rsdInfo.m_trd_pic_extra_resample_num_ids[resampleId++];
          tmpPredictNumVec[rsdInfo.m_picTemporalChangedPOCs[i]] = rsdInfo.m_trd_pic_extra_predict_num_ids[predictId++];
        }
      }
    }

    std::vector<bool>     outFlagVec;
    std::vector<uint32_t>    outModeVec;
    std::vector<uint32_t> outIndexVec;
    std::vector<uint32_t> outResampleNumVec;
    std::vector<uint32_t> outPredictNumVec;

    for (const auto& pair : tmpFlagVec)
    {
      outFlagVec.push_back(pair.second);
    }
    for (const auto& pair: tmpModeVec)
    {
      outModeVec.push_back(pair.second);
    }
    for (const auto& pair: tmpIndexVec)
    {
      outIndexVec.push_back(pair.second);
    }
    for (const auto& pair: tmpResampleNumVec)
    {
      outResampleNumVec.push_back(pair.second);
    }
    for (const auto& pair: tmpPredictNumVec)
    {
      outPredictNumVec.push_back(pair.second);
    }
      
    rsdInfo.m_trd_pic_changed_flags          = outFlagVec;
    rsdInfo.m_trd_pic_modes = outModeVec;
    rsdInfo.m_trd_pic_inter_ratio_ids = outIndexVec;
    rsdInfo.m_trd_pic_extra_resample_num_ids = outResampleNumVec;
    rsdInfo.m_trd_pic_extra_predict_num_ids = outPredictNumVec;

  if (len_flags > 0)
  {
    outfile << "PHTemporalChangedFlags:";
    for (int i = 0; i < len_flags; i++)
    {
      outfile << rsdInfo.m_trd_pic_changed_flags[i];
      if (i != (len_flags - 1))
        outfile << ",";
    }
    outfile << std::endl;

    std::string str;
    str = writeListToCfgLine("PHTemporalRestorationModes", outModeVec);
    if (str != "")        outfile << str << std::endl;
    str = writeListToCfgLine("PHTemporalInterpolationRatioInds", outIndexVec);
    if (str != "")        outfile << str << std::endl;
    str = writeListToCfgLine("PHTemporalExtraResamplingNumIds", outResampleNumVec);
    if (str != "")        outfile << str << std::endl;
    str = writeListToCfgLine("PHTemporalExtraPredictNumIds", outPredictNumVec);
    if (str != "")        outfile << str << std::endl;
  }
#ifdef KHU_MPEG_148
    outfile << "TemporalResamplingPostHintFlag:" << rsdInfo.m_srd_trph_flag <<std::endl;
    if (rsdInfo.m_srd_trph_flag)
    {
      size_t len_trph_flags = rsdInfo.m_picTemporalChangedPOCs.size();
      if (len_trph_flags > 0)
      {
        outfile << "trph_quality_valid_flag:";
        for (size_t i = 0; i < len_trph_flags; i++)
        {
          outfile << rsdInfo.m_trph_quality_valid_flag[i];
          if (i != (len_trph_flags - 1)) 
            outfile << ",";
        }
        outfile << std::endl;
        outfile << "trph_quality_value:";
        for (size_t i = 0; i < len_trph_flags; i++)
        {
          outfile << rsdInfo.m_trph_quality_value[i];
          if (i != (len_trph_flags - 1)) 
            outfile << ",";
        }
        outfile << std::endl;      
      }
    }
#endif      
  outfile.close();
}
#else
    // sort the flags and indexes
    CHECK(rsdInfo.m_picTemporalChangedFlags.size() != rsdInfo.m_picTemporalRatioIndexes.size() || rsdInfo.m_picTemporalChangedFlags.size() != rsdInfo.m_picTemporalChangedPOCs.size(), "picture level lists are not in equal length")
    size_t len_flags = rsdInfo.m_picTemporalChangedPOCs.size();
    std::vector<bool> tmpFlagVec;
    std::vector<uint32_t> tmpIndexVec;
    tmpFlagVec.resize(len_flags);
    tmpIndexVec.resize(len_flags);
    for (size_t i = 0; i < len_flags; i++)
    {
      CHECK(rsdInfo.m_picTemporalChangedPOCs[i] >= len_flags, "POC value for temporal exceeds the list length");
      tmpFlagVec[rsdInfo.m_picTemporalChangedPOCs[i]] = rsdInfo.m_picTemporalChangedFlags[i];
      tmpIndexVec[rsdInfo.m_picTemporalChangedPOCs[i]] = rsdInfo.m_picTemporalRatioIndexes[i];
    }
    rsdInfo.m_picTemporalChangedFlags = tmpFlagVec;
    rsdInfo.m_picTemporalRatioIndexes = tmpIndexVec;

    if (len_flags > 0)
    {
      outfile << "PHTemporalChangedFlags:";
      for (int i = 0; i < len_flags; i++)
      {
        outfile << rsdInfo.m_picTemporalChangedFlags[i];
        if (i != (len_flags - 1))
          outfile << ",";
      }
      outfile << std::endl;
    }
    else
    {
      outfile << "PHTemporalChangedFlags:";
      uint32_t ratio = rsdInfo.temporal_restoration_flag ? rsdInfo.m_trdRatio : 1;
      uint32_t len   = (countFrame - 1) / ratio + 1;
      for (int i = 0; i < len; i++)
      {
        outfile << 0;
        if (i != (len - 1))
          outfile << ",";
      }
      outfile << std::endl;
    }

    if (len_flags > 0)
    {
      int count = 0;
      for (int i = 0; i < len_flags; i++)
      {
        if (rsdInfo.m_picTemporalChangedFlags[i])
        {
          if (count == 0)
            outfile << "PHTemporalRatioIndexes:";
          count++;
          outfile << rsdInfo.m_picTemporalRatioIndexes[i];
          if (i != (len_flags - 1))
            outfile << ",";
        }
      }
      outfile << std::endl;
    }
    outfile.close();
  }
#endif
}
#endif


//! \}
