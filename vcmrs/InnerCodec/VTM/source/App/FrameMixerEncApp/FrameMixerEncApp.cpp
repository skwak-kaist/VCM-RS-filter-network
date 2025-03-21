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

 /** \file     FrameMixerEncApp.cpp
     \brief    Frame mixer enc application
 */

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
#include "AnnexBread.h"
#include "NALread.h"
#include "Slice.h"
#include "SEIwrite.h"
#include "VLCWriter.h"
#include "NALwrite.h"
#include "AnnexBwrite.h"
#include "FrameMixerEncApp.h"
#if ZJU_BIT_STRUCT
#include "VCMBitStruct.h"
#endif


 //! \ingroup FrameMixerEncApp
 //! \{


static const int MIXED_NALU_PPS_OFFSET = 8;


struct Subpicture {
  int                                  width;
  int                                  height;
  int                                  topLeftCornerX;
  int                                  topLeftCornerY;
  std::ifstream                        *fp;
  InputByteStream                      *bs;
  bool                                 firstSliceInPicture;
  int                                  NNintraModelIdc;
  bool                                 NNIntraFilterFlag;
  bool                                 NNIntraFilterPatchWiseFlag;
  int                                  NNIntraFilterPatchSize;
  std::vector<bool>                    NNIntraFilterPatchFlag;
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


FrameMixerEncApp::FrameMixerEncApp(std::vector<SubpicParams> &subpicParams, std::ofstream &outputStream, std::string &inputIntraBaseFileName, std::string &inputConfigBaseFileName, bool removePs, std::string &interMachineAdapterConfigBaseFileName
#if ZJU_BIT_STRUCT
, std::string &inputRestorationDataFileName
, std::string &inputTemporalRestorationDataFileName
#endif
) :
  m_outputStream(outputStream),
  m_inputIntraBaseFileName(inputIntraBaseFileName),
  m_inputConfigBaseFileName(inputConfigBaseFileName),
  m_removePs(removePs),
  m_interMachineAdapterConfigBaseFileName(interMachineAdapterConfigBaseFileName),
#if ZJU_BIT_STRUCT
  m_inputRestorationDataFileName(inputRestorationDataFileName),
  m_inputTemporalRestorationDataFileName(inputTemporalRestorationDataFileName),
#endif
  m_prevPicPOC(std::numeric_limits<int>::max()),
  m_picWidth(0),
  m_picHeight(0)
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

FrameMixerEncApp::~FrameMixerEncApp()
{
  delete m_subpics;
}


/**
 - lookahead through next NAL units to determine if current NAL unit is the first NAL unit in a new picture
 */
bool FrameMixerEncApp::isNewPicture(std::ifstream *bitstreamFile, InputByteStream *bytestream, bool firstSliceInPicture)
{
  bool ret = false;
  bool finished = false;

  // cannot be a new picture if there haven't been any slices yet
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
bool FrameMixerEncApp::parseDCI(HLSyntaxReader &hlsReader, DCI &dci)
{
  hlsReader.parseDCI(&dci);
  msg( INFO, "  DCI");
  return true;
}

/**
  - Parse VPS and store it in parameter set manager
*/
int FrameMixerEncApp::parseVPS(HLSyntaxReader &hlsReader, ParameterSetManager &psManager)
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
int FrameMixerEncApp::parseSPS(HLSyntaxReader &hlsReader, ParameterSetManager &psManager)
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
int FrameMixerEncApp::parsePPS(HLSyntaxReader &hlsReader, ParameterSetManager &psManager)
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
void FrameMixerEncApp::parseAPS(HLSyntaxReader &hlsReader, ParameterSetManager &psManager, int &apsId, int &apsType)
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
void FrameMixerEncApp::parsePictureHeader(HLSyntaxReader &hlsReader, PicHeader &picHeader, ParameterSetManager &psManager)
{
  hlsReader.parsePictureHeader(&picHeader, &psManager, true);
  picHeader.setValid();
  msg( INFO, "  PH");
}

/**
  - Parse slice header and store slice data
*/
void FrameMixerEncApp::parseSliceHeader(HLSyntaxReader &hlsReader, InputNALUnit &nalu, Slice &slice, PicHeader &picHeader, OutputBitstream &sliceData, ParameterSetManager &psManager, int prevTid0Poc)
{
  slice.initSlice();
  slice.setNalUnitType(nalu.m_nalUnitType);
  slice.setTLayer(nalu.m_temporalId);
  slice.setPicHeader(&picHeader);
  if (nalu.m_nalUnitType == NAL_UNIT_CODED_SLICE_NN_IRAP)
  {
    hlsReader.parseSliceHeaderNN(&slice, &picHeader, &psManager, prevTid0Poc, m_prevPicPOC);
  }
  else
  {
    hlsReader.parseSliceHeader(&slice, &picHeader, &psManager, prevTid0Poc, m_prevPicPOC);
  }
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
  - Decode NAL unit if it is parameter set or picture header, or decode slice header of VLC NAL unit
 */
void FrameMixerEncApp::decodeNalu(Subpicture &subpic, InputNALUnit &nalu)
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
      parseSliceHeader(hlsReader, nalu, subpic.slices.back(), subpic.picHeader, subpic.sliceData.back(), subpic.psManager, subpic.prevTid0Poc);
    }
    else if (nalu.isSei())
    {
      msg( INFO, "  SEI");
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
void FrameMixerEncApp::parseSubpic(Subpicture &subpic, bool &morePictures)
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

  subpic.NNintraModelIdc = 0;
  subpic.NNIntraFilterFlag = false;
  subpic.NNIntraFilterPatchWiseFlag = false;
  subpic.NNIntraFilterPatchSize = 0;
  subpic.NNIntraFilterPatchFlag.resize(0);
  subpic.NNPfaId = 0;
  subpic.NNPfaSEIEnabled = false;

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
void FrameMixerEncApp::generateMergedStreamVPSes(std::vector<VPS*> &vpsList)
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
void FrameMixerEncApp::generateMergedStreamSPSes(std::vector<SPS*> &spsList)
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
void FrameMixerEncApp::generateMergedStreamPPSes(ParameterSetManager &, std::vector<PPS*> &)
{
  return;
}

/**
  - Configure slice headers of all subpicture for merged stream
*/
void FrameMixerEncApp::updateSliceHeadersForMergedStream(ParameterSetManager &psManager)
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
void FrameMixerEncApp::copyInputNaluToOutputNalu(OutputNALUnit &outNalu, InputNALUnit &inNalu)
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
void FrameMixerEncApp::copyNalUnitsToAccessUnit(AccessUnit &accessUnit, std::vector<InputNALUnit> &nalus, int naluType)
{
  for (auto &inNalu : nalus)
  {
    if (inNalu.m_nalUnitType == (NalUnitType)naluType)
    {
      if (!(naluType == NAL_UNIT_SUFFIX_SEI && SEI::PayloadType(inNalu.getBitstream().getFifo().at(2)) == SEI::PayloadType::DECODED_PICTURE_HASH))  // Don't copy decoded_picture_hash SEI
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
void FrameMixerEncApp::generateNNPfaSEI(AccessUnit *au)
{
  Subpicture &subpic = m_subpics->at(0);

  if (!subpic.NNPfaSEIEnabled)
  {
    return;
  }

  auto naluIt = au->begin();

  // Skip nalus before VLC nalus
  while (naluIt != au->end() &&  !(*naluIt)->isVcl())
  {
    naluIt++;
  }

  if (naluIt == au->end())
  {
    return;
  }

  OutputNALUnit outNalu(NAL_UNIT_PREFIX_SEI, (*naluIt)->m_nuhLayerId, (*naluIt)->m_temporalId);

  SEIWriter *seiWriter = new SEIWriter;
  HRD *hrd = new HRD;

  SEIMessages seiMsgs;
  
#if ZJU_VCM_SINGLE || 1
  SEINeuralNetworkPostFilterCharacteristics *pfc = new SEINeuralNetworkPostFilterCharacteristics;
  
  pfc->m_id = subpic.NNPfaId;
  pfc->m_propertyPresentFlag = true;
  pfc->m_baseFlag = true;
  pfc->m_modeIdc = 1;
  
  seiMsgs.push_back(pfc);
  SEINeuralNetworkPostFilterActivation *pfa = new SEINeuralNetworkPostFilterActivation;

  pfa->m_targetId = subpic.NNPfaId;
  seiMsgs.push_back(pfa);

  seiWriter->writeSEImessages(outNalu.m_bitstream, seiMsgs, *hrd, false, outNalu.m_temporalId);
  au->insert(naluIt, new NALUnitEBSP(outNalu));
  
  
#endif

  delete hrd;
  delete seiWriter;
  delete pfc;
  delete pfa;
}


/**
  - Write NAL units for one picture
 */
void FrameMixerEncApp::writeOnePic()
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

  generateNNPfaSEI(&accessUnit);

  writeAnnexBAccessUnit(m_outputStream, accessUnit);
}

void FrameMixerEncApp::writeOneNNPic(std::vector<uint8_t> &NNSliceData, bool removePs)
{
  HLSWriter hlsWriter;
  AccessUnit accessUnit;

  for (auto& subpic : *m_subpics)
  {
    auto sliceItr = subpic.slices.begin();

    if (removePs)
    {
      for (auto spsId : subpic.spsIds)
      {
        CHECK(subpic.psManager.getSPSChangedFlag(spsId), "Cannot remove changed SPS");
      }
      for (auto ppsId : subpic.ppsIds)
      {
        CHECK(subpic.psManager.getPPSChangedFlag(ppsId), "Cannot remove changed PPS");
      }
    }

    for (auto& nalu : subpic.nalus)
    {
      if (nalu.isVcl())
      {
        OutputNALUnit naluOut(NAL_UNIT_CODED_SLICE_NN_IRAP, nalu.m_nuhLayerId, nalu.m_temporalId);
        hlsWriter.setBitstream( &naluOut.m_bitstream );
        Slice *slice = &(*sliceItr);

        NNIrapSubtype naluSubtype;
        if (nalu.m_nalUnitType == NAL_UNIT_CODED_SLICE_IDR_W_RADL)
        {
          naluSubtype = NN_IRAP_SUBTYPE_IDR_W_RADL;
        }
        else if (nalu.m_nalUnitType == NAL_UNIT_CODED_SLICE_IDR_N_LP)
        {
          naluSubtype = NN_IRAP_SUBTYPE_IDR_N_LP;
        }
        else if (nalu.m_nalUnitType == NAL_UNIT_CODED_SLICE_CRA)
        {
          naluSubtype = NN_IRAP_SUBTYPE_CRA;
        }
        else
        {
          CHECK(true, "Illegal VCL NALU type in writeOneNNPic()");
        }
        slice->setNalUnitType(NAL_UNIT_CODED_SLICE_NN_IRAP);
        slice->setNNIrapSubtype(naluSubtype);
        slice->setNNIrapModelIdc(subpic.NNintraModelIdc);
        slice->setNNIrapFilterFlag(subpic.NNIntraFilterFlag);
        slice->setNNIrapFilterPatchWiseFlag(subpic.NNIntraFilterPatchWiseFlag);
        slice->setNNIrapFilterPatchSize(subpic.NNIntraFilterPatchSize);
        slice->setNNIrapFilterPatchFlags(subpic.NNIntraFilterPatchFlag);

        hlsWriter.codeSliceHeaderNN(&(*slice), &subpic.picHeader);
        naluOut.m_bitstream.writeByteAlignment();

        // Add NN slice data
        OutputBitstream SliceDataBitstream;
        SliceDataBitstream.getFifo() = NNSliceData;
        naluOut.m_bitstream.addSubstream(&SliceDataBitstream);
        accessUnit.push_back(new NALUnitEBSP(naluOut));

        sliceItr++;
      }
      else if (!removePs || ((nalu.m_nalUnitType != NAL_UNIT_SPS) && (nalu.m_nalUnitType != NAL_UNIT_PPS)))
      {
        OutputNALUnit outNalu((NalUnitType)nalu.m_nalUnitType);
        copyInputNaluToOutputNalu(outNalu, nalu);
        accessUnit.push_back(new NALUnitEBSP(outNalu));
      }
    }
  }

  generateNNPfaSEI(&accessUnit);

  writeAnnexBAccessUnit(m_outputStream, accessUnit);
}


/**
  - Create file name
*/
void FrameMixerEncApp::generateFilename(std::string &generatedFilename, std::string &baseFilename, int n, const char* suffix)
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
  - Parse NN frame paramaters
*/
bool FrameMixerEncApp::parseIntraNNConfig(Subpicture &subpic, std::ifstream &inputConfigStream)
{
  subpic.NNintraModelIdc = 0;
  subpic.NNIntraFilterFlag = false;

  std::string line;
  while (std::getline(inputConfigStream, line))
  {
    if (line.at(0) != '#')  // Lines beginning with # are ignored
    {
      std::istringstream iss(line);
      std::string paramName;
      int paramVal;

      if (!(iss >> paramName >> paramVal))
      {
        CHECK(true, "Error: config file has incorrect formatting");
      }

      if (paramName == "intra_model_id")
      {
        subpic.NNintraModelIdc = paramVal;
      }
      else if (paramName == "intra_filter_flag")
      {
        subpic.NNIntraFilterFlag = (paramVal == 1 ? true : false);
      }
      else if (paramName == "intra_filter_patch_wise_flag")
      {
        subpic.NNIntraFilterPatchWiseFlag = (paramVal == 1 ? true : false);
      }
      else if (paramName == "intra_filter_patch_size")
      {
        subpic.NNIntraFilterPatchSize = paramVal;
      }
      else if (paramName == "intra_filter_patch_flags")
      {
        int patchSize = 1 << (subpic.NNIntraFilterPatchSize + 6);
        int numOfPatches = std::max(1, (int)ceil((double)subpic.slices.at(0).getPPS()->getPicWidthInLumaSamples()/patchSize - 0.25)) * std::max(1, (int)ceil((double)subpic.slices.at(0).getPPS()->getPicHeightInLumaSamples()/patchSize - 0.25));
        subpic.NNIntraFilterPatchFlag.resize(0);
        if (numOfPatches > 0)
        {
          subpic.NNIntraFilterPatchFlag.push_back(paramVal == 1 ? true : false);
          for (int i = 1; i < numOfPatches; i++)
          {
            if (iss >> paramVal)
            {
              subpic.NNIntraFilterPatchFlag.push_back(paramVal == 1 ? true : false);
            }
            else
            {
              CHECK(false, "Too few intra patch flags");
            }
          }
        }
      }
      else
      {
        std::cerr << "Unknown NN paramater: " << paramName << std::endl;
        CHECK(true, "Unknown NN paramater");
      }
    }
  }

  return true;
}


/**
  - Parse inter machine adapter config for SEI
*/
bool FrameMixerEncApp::parseInterMachineAdapterConfig(Subpicture& subpic, std::ifstream& interMachineAdapterConfigStream)
{
  subpic.NNPfaId = 0;
  subpic.NNPfaSEIEnabled = false;

  std::string line;
  while (std::getline(interMachineAdapterConfigStream, line))
  {
    if (line.at(0) != '#')  // Lines beginning with # are ignored
    {
      std::istringstream iss(line);
      std::string paramName;
      int paramVal;

      if (!(iss >> paramName >> paramVal))
      {
        CHECK(true, "Error: inter machine adapter config file has incorrect formatting");
      }

      if (paramName == "nnpfa_id")
      {
        subpic.NNPfaId = paramVal;
        subpic.NNPfaSEIEnabled = true;
      }
      else
      {
        std::cerr << "Unknown inter machine adapter config paramater: " << paramName << std::endl;
        CHECK(true, "Unknown inter machine adapter config paramater");
      }
    }
  }

  return true;
}

/**
  - Mix NN coded intra frames with VVC bitstream
 */
void FrameMixerEncApp::mixFrames()
{
  ParameterSetManager psManager;  // Parameter sets for merged stream
  int picNum = 0;
  int intraPicNum = 0;

  // msg( INFO, "Output picture size is %ix%i\n", m_picWidth, m_picHeight);

  for (auto &subpic : *m_subpics)
  {
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
    }

    // validateSubpics();

    if (!m_interMachineAdapterConfigBaseFileName.empty())
    {
      std::string interMachineAdapterConfigFileName;
      std::ifstream interMachineAdapterConfigStream;

      generateFilename(interMachineAdapterConfigFileName, m_interMachineAdapterConfigBaseFileName, m_subpics->at(0).slices.at(0).getPOC(), ".txt");
      interMachineAdapterConfigStream.open(interMachineAdapterConfigFileName);
      if (interMachineAdapterConfigStream.is_open())
      {
        std::cout << "Found inter machine adapter config file for POC=" << m_subpics->at(0).slices.at(0).getPOC() << std::endl;

        parseInterMachineAdapterConfig(m_subpics->at(0), interMachineAdapterConfigStream);
        interMachineAdapterConfigStream.close();
      }
    }

    if (m_subpics->at(0).slices.at(0).isIntra())
    {
      std::string inputIntraFileName;
      std::string inputConfigFileName;
      std::ifstream inputIntraStream;
      std::ifstream inputConfigStream;

      // Open NN picture data file
      if (!m_inputIntraBaseFileName.empty())
      {
        generateFilename(inputIntraFileName, m_inputIntraBaseFileName, intraPicNum, ".bin");
        inputIntraStream.open(inputIntraFileName, std::ios_base::binary);
        if (!inputIntraStream.is_open())
        {
          std::cerr << "Error: cannot open intput intra file " << inputIntraFileName << " for reading" << std::endl;
          CHECK(true, "Cannot open intput intra file");
        }
      }

      // Open NN picture config file
      if (!m_inputConfigBaseFileName.empty())
      {
        generateFilename(inputConfigFileName, m_inputConfigBaseFileName, intraPicNum, ".txt");
        inputConfigStream.open(inputConfigFileName);
        if (!inputConfigStream.is_open())
        {
          std::cerr << "Error: cannot open intput config file " << inputConfigFileName << " for reading" << std::endl;
          CHECK(true, "Cannot open intput intra config file");
        }

        parseIntraNNConfig(m_subpics->at(0), inputConfigStream);
        inputConfigStream.close();
      }

      // Read NN coded data
      if (!m_inputIntraBaseFileName.empty())
      {
        inputIntraStream.seekg(0, std::ios_base::end);
        int fileSize = (int)inputIntraStream.tellg();
        inputIntraStream.seekg(0, std::ios_base::beg);
        std::vector<uint8_t> NNSliceData(fileSize);
        inputIntraStream.read(reinterpret_cast<char*>(NNSliceData.data()), fileSize);    // Read NN slice data
        inputIntraStream.close();

        writeOneNNPic(NNSliceData, intraPicNum == 0 ? false : m_removePs);
      }
      else
      {
        writeOnePic();
      }
    }
    else
    {
      writeOnePic();
    }

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
void FrameMixerEncApp::writeOneCVDPic(OutputBitstream &cvdBs)
{
  AccessUnit accessUnit;

  for (auto& subpic : *m_subpics)
  {
    for (auto& nalu : subpic.nalus)
    {
      OutputNALUnit outNalu((NalUnitType)nalu.m_nalUnitType);
      copyInputNaluToOutputNalu(outNalu, nalu);
      accessUnit.push_back(new NALUnitEBSP(outNalu));
      // CHECK(outNalu.m_bitstream.getByteStreamLength()%8, "Found non byte aligned cvd 2");
    }
  }

  generateNNPfaSEI(&accessUnit);

  /* write AnnexB NAL units */
  for (AccessUnit::const_iterator it = accessUnit.begin(); it != accessUnit.end(); it++)
  {
    const NALUnitEBSP& nalu = **it;
    const bool useLongStartCode = (it == accessUnit.begin() || nalu.m_nalUnitType == NAL_UNIT_OPI || nalu.m_nalUnitType == NAL_UNIT_DCI || nalu.m_nalUnitType == NAL_UNIT_VPS || nalu.m_nalUnitType == NAL_UNIT_SPS
                                   || nalu.m_nalUnitType == NAL_UNIT_PPS || nalu.m_nalUnitType == NAL_UNIT_PREFIX_APS || nalu.m_nalUnitType == NAL_UNIT_SUFFIX_APS);
    uint32_t startCodePrefix[] = {0,0,0,1};

    if (useLongStartCode)
    {
      cvdBs.write(*startCodePrefix, 8);
      cvdBs.write(*(startCodePrefix+1), 8);
      cvdBs.write(*(startCodePrefix+2), 8);
      cvdBs.write(*(startCodePrefix+3), 8);
    }
    else
    {
      cvdBs.write(*(startCodePrefix+1), 8);
      cvdBs.write(*(startCodePrefix+2), 8);
      cvdBs.write(*(startCodePrefix+3), 8);
    }
    std::istringstream tmpInBs(nalu.m_nalUnitData.str());
    uint32_t size = nalu.m_nalUnitData.str().size();
    std::vector<uint8_t> tmpNaluData(size);
    OutputBitstream tmpOutBs;
    tmpInBs.read(reinterpret_cast<char*>(tmpNaluData.data()), size);
    tmpOutBs.getFifo() = tmpNaluData;
    cvdBs.addSubstream(&tmpOutBs);
    // CHECK(cvdBs.getByteStreamLength()%8, "Found non byte aligned cvd 1");
  }
}

void writeSampleStream(OutputBitstream &inBs, OutputBitstream &outBs)
{
  uint32_t numBytes = 4; // TODO: should find the largest size and decide the numbytes used for unit size.

  outBs.write(numBytes - 1, 3);
  outBs.write(0, 5);
  outBs.write(inBs.getByteStreamLength(), numBytes * 8);
  outBs.addSubstream(&inBs);
}

void parseRsdFromFile(std::string rsdFileName, std::string trdFileName, RSDInfo &rsdInfo)
{
  // read file.
  std::ifstream frd;
  frd.open(rsdFileName, std::ios_base::binary);
  if (!frd.is_open())
  {
    std::cerr << "Error: cannot open intput restoration data file " << rsdFileName << " for reading" << std::endl;
  }
  frd.seekg(0, std::ios_base::end);
  int rsdByteSize = (int)frd.tellg();
  frd.seekg(0, std::ios_base::beg);
  std::vector<uint8_t> rsd(rsdByteSize);
  frd.read(reinterpret_cast<char*>(rsd.data()), rsdByteSize);
  frd.close();

  // collect temporal restoration data.
  InputBitstream inRsd;
  InputBitstream *extractData;
  inRsd.getFifo() = rsd;
  uint32_t valueBits = 0;
  RSDCompMap CompType;
  uint32_t sizeSubStream = 0;

  // init
  rsdInfo.temporal_restoration_flag = 0;
  rsdInfo.bit_depth_on_flag = 0;
  rsdInfo.spatial_on_flag = 0;
  rsdInfo.roi_gops_num = 0;
  rsdInfo.colorizer_gops_num = 0;

  inRsd.read(8, rsdInfo.m_vcmFrameRate);
  inRsd.read(8);
  inRsd.read(8, rsdInfo.spatial_resampling_simple_flag);
  inRsd.read(8);

  inRsd.read(8, valueBits);
  while (valueBits < 255)
  {
    CompType = RSDCompMap(valueBits);
    inRsd.read(8, valueBits); //
    sizeSubStream = valueBits;
    inRsd.read(8, valueBits); //
    sizeSubStream += valueBits << 8;
    extractData = inRsd.extractSubstream(sizeSubStream*8);
    if (CompType == COMP_TemporalResample)
    {
      rsdInfo.temporal_restoration_flag = 1;
      // parse trd.
      CHECK(extractData->getNumBitsLeft() != 3 * 8, "received temporal restoration data is not correct");
      extractData->read(8, valueBits);
      rsdInfo.m_trdRatioId = valueBits;
      rsdInfo.m_trdRatio = 1 << (valueBits + 1);
      extractData->read(16, valueBits);
      rsdInfo.m_trdNumPicToBeRecon = valueBits;
    }
    else if (CompType == COMP_BitDepthTruncation)
    {
      rsdInfo.bit_depth_on_flag = 1;
      CHECK(extractData->getNumBitsLeft() != 4 * 8, "received temporal restoration data is not correct");
      extractData->read(8, valueBits);
      rsdInfo.bit_depth_shift_flag = valueBits;
      extractData->read(8, valueBits);
      rsdInfo.bit_depth_shift_luma = valueBits;
      extractData->read(8, valueBits);
      rsdInfo.bit_depth_shift_chroma = valueBits;
      extractData->read(8, valueBits);
      rsdInfo.bit_depth_luma_enhance = valueBits;
    }
    else if (CompType == COMP_SpatialResample)
    {
      rsdInfo.spatial_on_flag = 1;
      if (!rsdInfo.spatial_resampling_simple_flag)
      {
        // Adaptive downsample.
        extractData->read(8, valueBits);
        rsdInfo.vcm_spatial_resampling_flag = valueBits;
        if (rsdInfo.vcm_spatial_resampling_flag)
        {
          extractData->read(8, valueBits);
          rsdInfo.spatial_resample_width = valueBits << 8;
          extractData->read(8, valueBits);
          rsdInfo.spatial_resample_width += valueBits;
          extractData->read(8, valueBits);
          rsdInfo.spatial_resample_height = valueBits << 8;
          extractData->read(8, valueBits);
          rsdInfo.spatial_resample_height += valueBits;
          extractData->read(8, valueBits);
          rsdInfo.spatial_resample_filter_idx = valueBits;
        }
        else
        {
          rsdInfo.spatial_resample_height = 0;
          rsdInfo.spatial_resample_width = 0;
          rsdInfo.spatial_resample_filter_idx = 0;
        }
      }
      else
      {
        rsdInfo.vcm_spatial_resampling_flag = 1;
        extractData->read(8, valueBits);
        rsdInfo.scale_factor_id = valueBits;
      }
    }
    else if (CompType == COMP_Colorize)
    {
      extractData->read(8, valueBits);
      rsdInfo.colorizer_period = valueBits;
      extractData->read(8, valueBits);
      rsdInfo.colorizer_period |= valueBits<<8;
      
      extractData->read(8, valueBits);
      rsdInfo.colorizer_gops_num = valueBits;
      extractData->read(8, valueBits);
      rsdInfo.colorizer_gops_num |= valueBits<<8;
          
      for (size_t i = 0; i < rsdInfo.colorizer_gops_num; i++)
      {
        ColorizerInfo tmpColorizerInfo;
        extractData->read(8, tmpColorizerInfo.colorizer_enable_flag);
        if (tmpColorizerInfo.colorizer_enable_flag)
        {
          extractData->read(8, tmpColorizerInfo.colorizer_index);
          //extractData->read(8, tmpColorizerInfo.luma_pre_shift); // not needed due to adopted m71663 (bit depth shift cleanup).
        }
        rsdInfo.colorizerInfo.push_back(tmpColorizerInfo);
      }
    }    
    else if (CompType == COMP_ROI)
    {
      extractData->read(5, rsdInfo.roi_update_period_len); // bits_roi_update_period
      extractData->read(rsdInfo.roi_update_period_len, rsdInfo.roi_update_period); // roi_update_period - 1
      extractData->read(5, valueBits); // bits_roi_gops_num
      extractData->read(valueBits, rsdInfo.roi_gops_num); // roi_gops_num

      for (size_t i = 0; i < rsdInfo.roi_gops_num; i++)
      {
        ROIInfo tmpRoiInfo;
        tmpRoiInfo.retargeting_flag = 1;
        extractData->read(5, tmpRoiInfo.rtg_image_size_len);
        extractData->read(tmpRoiInfo.rtg_image_size_len, tmpRoiInfo.rtg_image_size_width);
        extractData->read(tmpRoiInfo.rtg_image_size_len, tmpRoiInfo.rtg_image_size_height);
        uint32_t orgWidth = tmpRoiInfo.rtg_image_size_width;
        uint32_t orgHeight = tmpRoiInfo.rtg_image_size_height;

        extractData->read(1, tmpRoiInfo.rtg_image_size_difference_flag);
        if (tmpRoiInfo.rtg_image_size_difference_flag)
        {
          extractData->read(5, tmpRoiInfo.rtg_to_output_difference_len);
          extractData->read(tmpRoiInfo.rtg_to_output_difference_len, tmpRoiInfo.rtg_to_output_difference_width);
          extractData->read(tmpRoiInfo.rtg_to_output_difference_len, tmpRoiInfo.rtg_to_output_difference_height);
          orgWidth += tmpRoiInfo.rtg_to_output_difference_width;
          orgHeight += tmpRoiInfo.rtg_to_output_difference_height;
        }

        uint32_t MAXIMAL_SCALE_FACTOR = 15;
        extractData->read(1, tmpRoiInfo.rtg_rois_flag);
        if (tmpRoiInfo.rtg_rois_flag)
        {
          extractData->read(5, tmpRoiInfo.roi_size_len);
          extractData->read(4, tmpRoiInfo.num_rois_len);
          extractData->read(tmpRoiInfo.num_rois_len, tmpRoiInfo.num_rois);
          
          uint32_t current_roi_scale_factor = 0;
          uint32_t bitsPos = ceil(log2((orgWidth > orgHeight ? orgWidth: orgHeight) + 1));
          uint32_t bitsScaleFactor = ceil(log2(MAXIMAL_SCALE_FACTOR + 1));
          tmpRoiInfo.roi_scale_factor_flag.resize(tmpRoiInfo.num_rois);
          tmpRoiInfo.roi_scale_factor.resize(tmpRoiInfo.num_rois);
          tmpRoiInfo.roi_pos_x.resize(tmpRoiInfo.num_rois);
          tmpRoiInfo.roi_pos_y.resize(tmpRoiInfo.num_rois);
          tmpRoiInfo.roi_size_x.resize(tmpRoiInfo.num_rois);
          tmpRoiInfo.roi_size_y.resize(tmpRoiInfo.num_rois);
          for (size_t i = 0; i < tmpRoiInfo.num_rois; i++)
          {
            if (current_roi_scale_factor != MAXIMAL_SCALE_FACTOR)
            {
              extractData->read(1, tmpRoiInfo.roi_scale_factor_flag[i]);
              if (tmpRoiInfo.roi_scale_factor_flag[i])
              {
                extractData->read(bitsScaleFactor, tmpRoiInfo.roi_scale_factor[i]);
              }
            }
            extractData->read(bitsPos, tmpRoiInfo.roi_pos_x[i]);
            extractData->read(bitsPos, tmpRoiInfo.roi_pos_y[i]);
            extractData->read(tmpRoiInfo.roi_size_len, tmpRoiInfo.roi_size_x[i]);
            extractData->read(tmpRoiInfo.roi_size_len, tmpRoiInfo.roi_size_y[i]);
          }
        }
        rsdInfo.roiInfo.push_back(tmpRoiInfo);
      }
    }
    inRsd.read(8, valueBits);
  }

  // read temporal information
  std::ifstream infile(trdFileName);

  rsdInfo.m_srd_ratio_change_allowed_flag = 0;
  #ifdef KHU_MPEG_148
  rsdInfo.m_srd_trph_flag = 0;
  #endif
  if (infile)
  {
    std::string line, key, value;
    while (getline(infile, line))
    {
      if (line.find("VCMEnabled", 0) != -1)
      {
        rsdInfo.m_vcmExtensionEnabledFlag = true;
      }
      else if (line.find("TemporalEnabled", 0) != -1)
      {
        std::string str                          = line.substr(line.rfind(':') + 1, line.length() - 1).data();
        rsdInfo.m_temporalRestorationEnabledFlag = (bool) atoi(str.data());
        rsdInfo.temporal_restoration_flag        = (bool) atoi(str.data());
      }
      else if (line.find("TemporalRemain", 0) != -1)
      {
        rsdInfo.m_numTemporalRemain = atoi(line.substr(line.rfind(':') + 1, 1).c_str());
      }
#ifdef KHU_MPEG_148      
      else if (line.find("TemporalResamplingPostHintFlag", 0) != -1)
      {
        std::string str             = line.substr(line.rfind(':') + 1, 1);
        rsdInfo.m_srd_trph_flag           = (uint32_t)atoi(str.data());        
      }
      else if (line.find("trph_quality_valid_flag", 0) != -1)
      {
        std::string str = line.substr(line.rfind(':') + 1, line.length() - 1);
        getListfromCfgLine(str, rsdInfo.m_trph_quality_valid_flag);
      }
      else if (line.find("trph_quality_value", 0) != -1)
      {
        std::string str = line.substr(line.rfind(':') + 1, line.length() - 1);
        getListfromCfgLine(str, rsdInfo.m_trph_quality_value);
      }
#endif
#ifdef TEM_MPEG_148
      else if (line.find("srdTemporalRestorationMode", 0) != -1)
      {
        std::string str             = line.substr(line.rfind(':') + 1, line.length() - 1);
        rsdInfo.m_trdMode           = (uint32_t)atoi(str.data());
        printf("Parse TEMPAR.CFG srd_temporal_extrapolation_flag is %d\n", rsdInfo.m_trdMode);
      }
      else if (line.find("srdTemporalInterpolationRatioId", 0) != -1)
      {
        std::string str              = line.substr(line.rfind(':') + 1, line.length() - 1);
        rsdInfo.m_trd_inter_ratio_id = (uint32_t) (atoi(str.data()));
      }
      else if (line.find("srdTemporalExtraResamplingNumId", 0) != -1)
      {
        std::string str              = line.substr(line.rfind(':') + 1, line.length() - 1);
        rsdInfo.m_trd_extra_resample_num_id = (uint32_t) (atoi(str.data()));
      }
      else if (line.find("srdTemporalExtraPredictNumId", 0) != -1)
      {
        std::string str                     = line.substr(line.rfind(':') + 1, line.length() - 1);
        rsdInfo.m_trd_extra_predict_num_id = (uint32_t) (atoi(str.data()));
      }
      else if (line.find("PHTemporalChangedFlags", 0) != -1)
      {
        std::string str = line.substr(line.rfind(':') + 1, line.length() - 1);
        getListfromCfgLine(str, rsdInfo.m_trd_pic_changed_flags);
        if (rsdInfo.m_trd_pic_changed_flags.back())
        {
          rsdInfo.m_srd_ratio_change_allowed_flag = 1;
        }
      }
      else if (line.find("PHTemporalRestorationModes", 0) != -1)
      {
        std::string str = line.substr(line.rfind(':') + 1, line.length() - 1);
        getListfromCfgLine(str, rsdInfo.m_trd_pic_changed_flags, rsdInfo.m_trdMode, rsdInfo.m_trd_pic_modes);
      }
      else if (line.find("PHTemporalInterpolationRatioInds", 0) != -1)
      {
        std::string str = line.substr(line.rfind(':') + 1, line.length() - 1);
        getListfromCfgLine(str, rsdInfo.m_trd_pic_changed_flags, (rsdInfo.m_trd_inter_ratio_id + 1), rsdInfo.m_trd_pic_inter_ratio_ids);
      }
      else if (line.find("PHTemporalExtraResamplingNumIds", 0) != -1)
      {
        std::string strflags = line.substr(line.rfind(':') + 1, line.length() - 1);
        getListfromCfgLine(strflags, rsdInfo.m_trd_pic_changed_flags, rsdInfo.m_trd_extra_resample_num_id, rsdInfo.m_trd_pic_extra_resample_num_ids);
      }
      else if (line.find("PHTemporalExtraPredictNumIds", 0) != -1)
      {
        std::string strflags = line.substr(line.rfind(':') + 1, line.length() - 1);
        getListfromCfgLine(strflags, rsdInfo.m_trd_pic_changed_flags, (rsdInfo.m_trd_extra_predict_num_id - 1), rsdInfo.m_trd_pic_extra_predict_num_ids);
      }
#else
      else if (line.find("PHTemporalChangedFlags", 0) != -1)
      {
        std::string strflags = line.substr(line.rfind(':') + 1, line.length() - 1);
        char*       p        = strtok(const_cast<char*>(strflags.data()), ",");
        while (p)
        {
          rsdInfo.m_picTemporalChangedFlags.push_back((bool) atoi(p));
          if(rsdInfo.m_picTemporalChangedFlags.back())
          {
            rsdInfo.m_srd_ratio_change_allowed_flag = 1;
          }
          p = strtok(NULL, ",");
        }
      }
      else if (line.find("PHTemporalRatioIndexes", 0) != -1)
      {
        std::vector<uint32_t> picTemporalRatioIndexes;
        std::string                               strflags = line.substr(line.rfind(':') + 1, line.length() - 1);
        char*                                     p        = strtok(const_cast<char*>(strflags.data()), ",");
        while (p)
        {
          picTemporalRatioIndexes.push_back((uint32_t) atoi(p));
          p = strtok(NULL, ",");
        }
        int count = 0;
        for (int i = 0; i < rsdInfo.m_picTemporalChangedFlags.size(); i++)
        {
          if (rsdInfo.m_picTemporalChangedFlags[i])
          {
            rsdInfo.m_picTemporalRatioIndexes.push_back(picTemporalRatioIndexes[count++]);
          }
          else
          {
            rsdInfo.m_picTemporalRatioIndexes.push_back(picTemporalRatioIndexes[count - 1]);
          }
        }
      }
#endif
    }
    infile.close();
  }
}

/**
 * @brief Change all coded bits to VCM bitstream structure.
 * 
 */
void FrameMixerEncApp::genVCMBitstream(){
  ParameterSetManager psManager;  // Parameter sets for merged stream
  int picNum = 0;

  // msg( INFO, "Output picture size is %ix%i\n", m_picWidth, m_picHeight);

  for (auto &subpic : *m_subpics)
  {
    subpic.bs = new InputByteStream(*(subpic.fp));
    subpic.prevTid0Poc = 0;
    subpic.psManager.storeVPS(new VPS, std::vector<uint8_t>());  // Create VPS with default values (VTM slice header parser needs this)
  }

  /*
    Get restoration data from temp file.
  */
  RSDInfo rsdInfo;
  parseRsdFromFile(m_inputRestorationDataFileName, m_inputTemporalRestorationDataFileName, rsdInfo);

  OutputBitstream vcmBs;
  /*
    Set VCM unit
  */
  std::list<VCMUnit*> vcmUnits;
  VCMUnit vcmuVPS(VCM_VPS);
  vcmuVPS.m_vpsId = 0;
  vcmuVPS.m_vpsBitsForPOCLsb = 16;
  vcmuVPS.m_vpsSpatialFlag = rsdInfo.spatial_on_flag;
  vcmuVPS.m_vpsRetargetFlag = (rsdInfo.roi_gops_num > 0 ? 1: 0);
  vcmuVPS.m_vpsColorizerFlag = (rsdInfo.colorizer_gops_num > 0 ? 1 : 0);
  vcmuVPS.m_vpsTemporalFlag = rsdInfo.temporal_restoration_flag;
  vcmuVPS.m_vpsBitDepthShiftFlag = rsdInfo.bit_depth_on_flag;
  vcmuVPS.writeVCMPS(); // write and use once.
  /* collect VPS once*/
  vcmUnits.push_back(&vcmuVPS);

  /* 
    RSD
  */
  VCMUnit vcmuRSD(VCM_RSD, vcmuVPS.m_vpsId);
  std::list<VCMNalu*> vcmNalUnits;
  OutputBitstream vcmRsd;
  VCMNalu vcmNaluSRD(VCM_NAL_SRD);
  VCMNalu vcmNaluPRD(VCM_NAL_PRD);
  VCMNalu vcmNaluEOSS(VCM_NAL_EOSS);
  // SRD
  vcmNaluSRD.m_srdId = 0;
  vcmNaluSRD.m_refVPS = &vcmuVPS;
  // set TRD in SRD. temporarily ratio 4 is supported.
  rsdInfo.m_trdRatio                         = 4;
  vcmNaluSRD.m_srdTrdRatio = rsdInfo.m_trdRatio;
  vcmNaluSRD.m_srdTrdNumCodedFrame = rsdInfo.m_trdNumPicToBeRecon;
  vcmNaluSRD.m_srd_ratio_change_allowed_flag = rsdInfo.m_srd_ratio_change_allowed_flag;
#ifdef KHU_MPEG_148  
  vcmNaluSRD.m_srd_trph_flag = rsdInfo.m_srd_trph_flag;
#endif  
  //vcmNaluSRD.setSrdTrdSet();
  // PRD
  vcmNaluPRD.m_prdRefSrdId = vcmNaluSRD.m_srdId;
  vcmNaluPRD.m_prd_tr_ratio_changed_flag = 0;
  vcmNaluPRD.m_refSRD = &vcmNaluSRD;
#ifdef TEM_MPEG_148
  vcmNaluPRD.m_prd_tr_update_mode = 0;
  vcmNaluPRD.m_prd_tr_update_inter_ratio_index = 1;
  vcmNaluPRD.m_prd_tr_update_extra_resample_num_index = 0;
  vcmNaluPRD.m_prd_tr_update_extra_predict_num_index = 2;
#endif
#ifdef KHU_MPEG_148  
  vcmNaluPRD.m_prd_trph_quality_valid_flag = 0;
  vcmNaluPRD.m_prd_trph_quality_value = 0;
#endif  
  // EOSS
  vcmNaluEOSS.m_refVPS = &vcmuVPS;
  vcmNaluEOSS.m_numTemporalRemain = rsdInfo.m_numTemporalRemain;
  vcmNaluEOSS.writeEOSS(); // write and used once.

  /*
    CVD
  */
  VCMUnit vcmuCVD(VCM_CVD, vcmuVPS.m_vpsId);
  OutputBitstream vcmCvdBs;

  bool morePictures = true;
  uint32_t numGOPs = 0;
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
    }

    // Collect SRD for each intra period or ROI update period
    if (m_subpics->at(0).slices.at(0).isIntra()) // IRAP starts an intra period
    {
      numGOPs++;
      vcmNaluSRD.writeSRD(rsdInfo);
      if (vcmuVPS.m_vpsRetargetFlag)
      {
        CHECK( numGOPs > rsdInfo.roi_gops_num, "ROI info is not enough.");
      }
      if (vcmuVPS.m_vpsColorizerFlag)
      {
        CHECK( numGOPs > rsdInfo.colorizer_gops_num, "Colorizer info is not enough.");
      }
      writeSampleStream(vcmNaluSRD.m_bitstream, vcmRsd);
      vcmuRSD.writeRSD(vcmRsd);
      vcmUnits.push_back(new VCMUnit(vcmuRSD));
      vcmRsd.clear();
    }
    else if (m_subpics->at(0).slices.at(0).getPOC() != 0 && m_subpics->at(0).slices.at(0).getPOC() % (rsdInfo.roi_update_period+1) == 0)
    {
      // TODO:This is used for ROI in LD mode only since ROI is update in period but not in intra period. (Is this correct?)
      vcmNaluSRD.writeSRD(rsdInfo);
      writeSampleStream(vcmNaluSRD.m_bitstream, vcmRsd);
      vcmuRSD.writeRSD(vcmRsd);
      vcmUnits.push_back(new VCMUnit(vcmuRSD));
      vcmRsd.clear();
    }

    // collect PRD for each picture
    // set PRD info first
    vcmNaluPRD.m_prdPrevCodedPicPoc = m_subpics->at(0).slices.at(0).getPOC();
    vcmNaluPRD.m_refSRD = &vcmNaluSRD;
    vcmNaluPRD.m_prdRefSrdId = vcmNaluSRD.m_srdId;
    vcmNaluPRD.m_prdPocLsb = vcmNaluPRD.m_prdPrevCodedPicPoc;
    vcmNaluPRD.m_prdDecodedPicFlag = 1;
#ifdef TEM_MPEG_148
    if (rsdInfo.temporal_restoration_flag)
    {
      vcmNaluPRD.m_prd_tr_ratio_changed_flag = rsdInfo.m_trd_pic_changed_flags[vcmNaluPRD.m_prdPocLsb];
      if (vcmNaluPRD.m_prd_tr_ratio_changed_flag)
      {
        vcmNaluPRD.m_prd_tr_update_mode = rsdInfo.m_trd_pic_modes[vcmNaluPRD.m_prdPocLsb];
        if (vcmNaluPRD.m_prd_tr_update_mode == 0)
        {
          vcmNaluPRD.m_prd_tr_update_inter_ratio_index = rsdInfo.m_trd_pic_inter_ratio_ids[vcmNaluPRD.m_prdPocLsb];
        }
        else if (vcmNaluPRD.m_prd_tr_update_mode == 1)
        {
          vcmNaluPRD.m_prd_tr_update_extra_resample_num_index =
            rsdInfo.m_trd_pic_extra_resample_num_ids[vcmNaluPRD.m_prdPocLsb];
          vcmNaluPRD.m_prd_tr_update_extra_predict_num_index =
            rsdInfo.m_trd_pic_extra_predict_num_ids[vcmNaluPRD.m_prdPocLsb];
        }
      }
    }
#endif
#ifdef KHU_MPEG_148
    if (rsdInfo.temporal_restoration_flag && rsdInfo.m_srd_trph_flag) 
    {
      vcmNaluPRD.m_prd_trph_quality_valid_flag = rsdInfo.m_trph_quality_valid_flag[vcmNaluPRD.m_prdPocLsb];
      vcmNaluPRD.m_prd_trph_quality_value = rsdInfo.m_trph_quality_value[vcmNaluPRD.m_prdPocLsb];
    }
#endif      
    // write PRD.
    vcmNaluPRD.writePRD();
    // collect CVD NAL units.
    writeOneCVDPic(vcmCvdBs);

    // store the each NAL unit in VCM unit
    writeSampleStream(vcmNaluPRD.m_bitstream, vcmRsd);
    vcmuRSD.writeRSD(vcmRsd);
    vcmUnits.push_back(new VCMUnit(vcmuRSD));
    vcmRsd.clear();
    vcmuCVD.writeCVD(vcmCvdBs);
    vcmUnits.push_back(new VCMUnit(vcmuCVD));
    vcmCvdBs.clear();

    // Update prevTid0Poc flags for subpictures
    for (auto &subpic : *m_subpics)
    {
      if (subpic.slices.size() > 0 && subpic.slices[0].getTLayer() == 0 &&
          subpic.slices[0].getNalUnitType() != NAL_UNIT_CODED_SLICE_RADL &&
          subpic.slices[0].getNalUnitType() != NAL_UNIT_CODED_SLICE_RASL )
      {
        subpic.prevTid0Poc = subpic.slices[0].getPOC();
      }
    }

    m_prevPicPOC = m_subpics->at(0).slices.at(0).getPOC();

    picNum++;
  }
  // write the EOSS NAL unit at the last
  writeSampleStream(vcmNaluEOSS.m_bitstream, vcmRsd);
  vcmuRSD.writeRSD(vcmRsd);
  vcmUnits.push_back(new VCMUnit(vcmuRSD));
  vcmRsd.clear();

  /*
    convert VCM units to sample stream and output.
  */
  for (auto vcmu : vcmUnits)
  {
    writeSampleStream(vcmu->m_bitstream, vcmBs);
  }
  m_outputStream.write(reinterpret_cast<const char*>(vcmBs.getByteStream()), vcmBs.getByteStreamLength());
}
#endif


//! \}
