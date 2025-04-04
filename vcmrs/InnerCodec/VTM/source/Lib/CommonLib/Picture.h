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

/** \file     Picture.h
 *  \brief    Description of a coded picture
 */

#ifndef __PICTURE__
#define __PICTURE__

#include "CommonDef.h"

#include "Common.h"
#include "Unit.h"
#include "Buffer.h"
#include "Unit.h"
#include "Slice.h"
#include "CodingStructure.h"
#include "Hash.h"
#include "MCTS.h"
#include "SEIColourTransform.h"
#include <deque>
#include "SEIFilmGrainSynthesizer.h"

#if NN_LF_UNIFIED
#include "CommonLib/NNFilterUnified.h"
#endif

class SEI;
class AQpLayer;

typedef std::list<SEI*> SEIMessages;

#define M_BUFS(JID,PID) m_bufs[PID]

#if GDR_ENABLED
struct GdrPicParam
{
  bool inGdrInterval = false;
  int  verBoundary   = -1;
};
#endif

struct Picture : public UnitArea
{
  uint32_t margin;
  Picture();

#if JVET_Z0120_SII_SEI_PROCESSING
  void create(const ChromaFormat &_chromaFormat, const Size &size, const unsigned _maxCUSize, const unsigned margin, const bool bDecoder, const int layerId, const bool enablePostFilteringForHFR, const bool gopBasedTemporalFilterEnabled = false, const bool fgcSEIAnalysisEnabled = false);
#else
  void create(const ChromaFormat &_chromaFormat, const Size &size, const unsigned _maxCUSize, const unsigned margin, const bool bDecoder, const int layerId, const bool gopBasedTemporalFilterEnabled = false, const bool fgcSEIAnalysisEnabled = false);
#endif
  void destroy();

  void createTempBuffers( const unsigned _maxCUSize );
  void destroyTempBuffers();

  int                       m_padValue;
  bool                      m_isMctfFiltered;
  SEIFilmGrainSynthesizer*  m_grainCharacteristic;
  PelStorage*               m_grainBuf;
  void              createGrainSynthesizer(bool firstPictureInSequence, SEIFilmGrainSynthesizer* grainCharacteristics, PelStorage* grainBuf, int width, int height, ChromaFormat fmt, int bitDepth);
  PelUnitBuf        getDisplayBufFG       (bool wrap = false);

  SEIColourTransformApply* m_colourTranfParams;
  PelStorage*              m_invColourTransfBuf;
  void              createColourTransfProcessor(bool firstPictureInSequence, SEIColourTransformApply* ctiCharacteristics, PelStorage* ctiBuf, int width, int height, ChromaFormat fmt, int bitDepth);
  PelUnitBuf        getDisplayBuf();

#if JVET_AC0074_USE_OF_NNPFC_FOR_PIC_RATE_UPSAMPLING
  SEIMessages m_nnpfcActivated;
#endif

#if JVET_Z0120_SII_SEI_PROCESSING
  void copyToPic(const SPS *sps, PelStorage *pcPicYuvSrc, PelStorage *pcPicYuvDst);
  Picture*  findPrevPicPOC(Picture* pcPic, PicList* pcListPic);
  Picture*  findNextPicPOC(Picture* pcPic, PicList* pcListPic);
  void  xOutputPostFilteredPic(Picture* pcPic, PicList* pcListPic, int blendingRatio);
  void  xOutputPreFilteredPic(Picture* pcPic, PicList* pcListPic, int blendingRatio, int intraPeriod);
#endif

         PelBuf     getOrigBuf(const CompArea &blk);
  const CPelBuf     getOrigBuf(const CompArea &blk) const;
         PelUnitBuf getOrigBuf(const UnitArea &unit);
  const CPelUnitBuf getOrigBuf(const UnitArea &unit) const;
         PelUnitBuf getOrigBuf();
  const CPelUnitBuf getOrigBuf() const;
         PelBuf     getOrigBuf(const ComponentID compID);
  const CPelBuf     getOrigBuf(const ComponentID compID) const;
         PelBuf     getTrueOrigBuf(const ComponentID compID);
  const CPelBuf     getTrueOrigBuf(const ComponentID compID) const;
         PelUnitBuf getTrueOrigBuf();
  const CPelUnitBuf getTrueOrigBuf() const;
        PelBuf      getTrueOrigBuf(const CompArea &blk);
  const CPelBuf     getTrueOrigBuf(const CompArea &blk) const;

         PelUnitBuf getFilteredOrigBuf();
  const CPelUnitBuf getFilteredOrigBuf() const;
         PelBuf     getFilteredOrigBuf(const CompArea &blk);
  const CPelBuf     getFilteredOrigBuf(const CompArea &blk) const;

         PelBuf     getPredBuf(const CompArea &blk);
  const CPelBuf     getPredBuf(const CompArea &blk) const;
         PelUnitBuf getPredBuf(const UnitArea &unit);
  const CPelUnitBuf getPredBuf(const UnitArea &unit) const;

         PelBuf     getResiBuf(const CompArea &blk);
  const CPelBuf     getResiBuf(const CompArea &blk) const;
         PelUnitBuf getResiBuf(const UnitArea &unit);
  const CPelUnitBuf getResiBuf(const UnitArea &unit) const;

         PelBuf     getRecoBuf(const ComponentID compID, bool wrap=false);
  const CPelBuf     getRecoBuf(const ComponentID compID, bool wrap=false) const;
         PelBuf     getRecoBuf(const CompArea &blk, bool wrap=false);
  const CPelBuf     getRecoBuf(const CompArea &blk, bool wrap=false) const;
         PelUnitBuf getRecoBuf(const UnitArea &unit, bool wrap=false);
  const CPelUnitBuf getRecoBuf(const UnitArea &unit, bool wrap=false) const;
         PelUnitBuf getRecoBuf(bool wrap=false);
  const CPelUnitBuf getRecoBuf(bool wrap=false) const;

         PelBuf     getBuf(const ComponentID compID, const PictureType &type);
  const CPelBuf     getBuf(const ComponentID compID, const PictureType &type) const;
         PelBuf     getBuf(const CompArea &blk,      const PictureType &type);
  const CPelBuf     getBuf(const CompArea &blk,      const PictureType &type) const;
         PelUnitBuf getBuf(const UnitArea &unit,     const PictureType &type);
  const CPelUnitBuf getBuf(const UnitArea &unit,     const PictureType &type) const;
#if NNVC_USE_BS
         PelBuf     getBsMapBuf(const ComponentID compID, bool wrap=false);
         PelUnitBuf getBsMapBuf(bool wrap=false);
         PelUnitBuf getBsMapBuf(const UnitArea &unit);
const   CPelUnitBuf getBsMapBuf(const UnitArea &unit) const;
         PelBuf     getBsMapBuf(const CompArea &blk);
const   CPelBuf     getBsMapBuf(const CompArea &blk) const;
#endif

#if JVET_AF0043_AF0205_PADDING
  void paddingPicBufBorder(const PictureType& type, const int& padSize);
  void paddingBsMapBufBorder(const int& padSize) { paddingPicBufBorder(PIC_BS_MAP, padSize); };
  void paddingRecBeforeDbfBufBorder(const int& padSize) { paddingPicBufBorder(PIC_REC_BEFORE_DBF, padSize); };
  void paddingPredBufBorder(const int& padSize) { paddingPicBufBorder(PIC_PREDICTION_CUSTOM, padSize); };
  void paddingBPMBufBorder(const int& padSize) { paddingPicBufBorder(PIC_BLOCK_PRED_MODE, padSize); };
#endif

#if NNVC_USE_PRED
     //    PelUnitBuf getPredBuf(bool wrap=false);
         PelBuf     getPredBufCustom(const ComponentID compID, bool wrap=false);
         PelUnitBuf getPredBufCustom(bool wrap=false);
         PelBuf     getPredBufCustom(const CompArea &blk);
  const CPelBuf     getPredBufCustom(const CompArea &blk)  const;
         PelUnitBuf getPredBufCustom(const UnitArea &unit);
  const CPelUnitBuf getPredBufCustom(const UnitArea &unit) const;
#endif
#if JVET_AC0089_NNVC_USE_BPM_INFO
  PelUnitBuf        getBlockPredModeBuf();
  const CPelUnitBuf getBlockPredModeBuf() const;
  PelUnitBuf        getBlockPredModeBuf(const UnitArea &unit);
  const CPelUnitBuf getBlockPredModeBuf(const UnitArea &unit) const;
  PelBuf            getBlockPredModeBuf(const CompArea &blk);
  const CPelBuf     getBlockPredModeBuf(const CompArea &blk) const;
  void              dumpPicBpmInfo();
#endif

#if NNVC_USE_REC_BEFORE_DBF
  PelBuf            getRecBeforeDbfBuf(const ComponentID compID, bool wrap=false);
  PelUnitBuf        getRecBeforeDbfBuf(bool wrap=false);
  PelBuf            getRecBeforeDbfBuf(const CompArea &blk);
  const CPelBuf     getRecBeforeDbfBuf(const CompArea &blk)  const;
  PelUnitBuf        getRecBeforeDbfBuf(const UnitArea &unit);
  const CPelUnitBuf getRecBeforeDbfBuf(const UnitArea &unit) const;
#endif

#if JVET_Z0120_SII_SEI_PROCESSING
        PelUnitBuf getPostRecBuf();
  const CPelUnitBuf getPostRecBuf() const;
#endif

  void extendPicBorder( const PPS *pps );
  void extendWrapBorder( const PPS *pps );
  void finalInit( const VPS* vps, const SPS& sps, const PPS& pps, PicHeader *picHeader, APS** alfApss, APS* lmcsAps, APS* scalingListAps );

  int  getPOC()                               const { return poc; }
  int  getDecodingOrderNumber()               const { return m_decodingOrderNumber; }
  void setDecodingOrderNumber(const int val)        { m_decodingOrderNumber = val;  }
  NalUnitType getPictureType()                const { return m_pictureType;         }
  void setPictureType(const NalUnitType val)        { m_pictureType = val;          }
  void        setBorderExtension(bool flag)
  {
    m_extendedBorder = flag;
  }
  Pel* getOrigin( const PictureType &type, const ComponentID compID ) const;
  int  getEdrapRapId()                        const { return edrapRapId ; }
  void setEdrapRapId(const int val)                 { edrapRapId = val; }

  void setLossyQPValue(int i)                 { m_lossyQP = i; }
  int getLossyQPValue()                       const { return m_lossyQP; }
  void      fillSliceLossyLosslessArray(std::vector<uint16_t> sliceLosslessArray, bool mixedLossyLossless);
  bool      losslessSlice(uint32_t sliceIdx)  const { return m_lossylosslessSliceArray[sliceIdx]; }

  int           getSpliceIdx(uint32_t idx) const { return m_spliceIdx[idx]; }
  void          setSpliceIdx(uint32_t idx, int poc) { m_spliceIdx[idx] = poc; }
  void          createSpliceIdx(int nums);
  bool          getSpliceFull();
  static void   sampleRateConv(const ScalingRatio scalingRatio, int scaleX, int scaleY, const CPelBuf &beforeScale,
                               const int beforeScaleLeftOffset, const int beforeScaleTopOffset, const PelBuf &afterScale,
                               const int afterScaleLeftOffset, const int afterScaleTopOffset, const int bitDepth,
                               const bool useLumaFilter, const bool downsampling,
                              const bool horCollocatedPositionFlag, const bool verCollocatedPositionFlag,
                              const bool rescaleForDisplay, const int upscaleFilterForDisplay
  );

  static void rescalePicture(const ScalingRatio scalingRatio, const CPelUnitBuf& beforeScaling,
                             const Window& scalingWindowBefore, const PelUnitBuf& afterScaling,
                             const Window& scalingWindowAfter, const ChromaFormat chromaFormatIdc,
                             const BitDepths& bitDepths, const bool useLumaFilter, const bool downsampling,
                             const bool horCollocatedChromaFlag, const bool verCollocatedChromaFlag,
                             bool rescaleForDisplay = false, int upscaleFilterForDisplay = 0);

private:
  Window        m_conformanceWindow;
  Window        m_scalingWindow;
  int           m_decodingOrderNumber;
  NalUnitType   m_pictureType;
#if GREEN_METADATA_SEI_ENABLED
  FeatureCounterStruct m_featureCounter;
#endif
public:
  bool m_isSubPicBorderSaved;

  PelStorage m_bufSubPicAbove;
  PelStorage m_bufSubPicBelow;
  PelStorage m_bufSubPicLeft;
  PelStorage m_bufSubPicRight;

  PelStorage m_bufWrapSubPicAbove;
  PelStorage m_bufWrapSubPicBelow;

  void    saveSubPicBorder(int POC, int subPicX0, int subPicY0, int subPicWidth, int subPicHeight);
  void  extendSubPicBorder(int POC, int subPicX0, int subPicY0, int subPicWidth, int subPicHeight);
  void restoreSubPicBorder(int POC, int subPicX0, int subPicY0, int subPicWidth, int subPicHeight);
#if GREEN_METADATA_SEI_ENABLED
  void setFeatureCounter (FeatureCounterStruct b ) { m_featureCounter = b;}
  FeatureCounterStruct getFeatureCounter (){return m_featureCounter;}
#endif
  
  bool getSubPicSaved()          { return m_isSubPicBorderSaved; }
  void setSubPicSaved(bool bVal) { m_isSubPicBorderSaved = bVal; }
  bool     m_extendedBorder;
  bool m_wrapAroundValid;
  unsigned m_wrapAroundOffset;
  bool referenced;
  bool reconstructed;
  bool neededForOutput;
  bool usedByCurr;
  bool longTerm;
  bool topField;
  bool fieldPic;
  EnumArray<int, ChannelType> m_prevQP;
  bool precedingDRAP; // preceding a DRAP picture in decoding order
  int  edrapRapId;
  bool nonReferencePictureFlag;

  int  poc;
  uint32_t temporalId;
  int      layerId;
  std::vector<SubPic> subPictures;
  int numSlices;
  std::vector<int> sliceSubpicIdx;

  bool subLayerNonReferencePictureDueToSTSA;

  int* m_spliceIdx;
  int  m_ctuNums;
  int m_lossyQP;
  std::vector<bool> m_lossylosslessSliceArray;
  bool interLayerRefPicFlag;
  bool mixedNaluTypesInPicFlag;

  PelStorage m_bufs[NUM_PIC_TYPES];
  const Picture*           unscaledPic;

  Hash               m_hashMap;
  Hash              *getHashMap() { return &m_hashMap; }
  const Hash        *getHashMap() const { return &m_hashMap; }
  void               addPictureToHashMapForInter();

  CodingStructure*   cs;
#if GDR_ENABLED
  GdrPicParam        gdrParam;
#endif
  std::deque<Slice*> slices;
  SEIMessages        SEIs;

  uint32_t           getPicWidthInLumaSamples() const                                { return  getRecoBuf( COMPONENT_Y ).width; }
  uint32_t           getPicHeightInLumaSamples() const                               { return  getRecoBuf( COMPONENT_Y ).height; }
  Window&            getConformanceWindow()                                          { return  m_conformanceWindow; }
  const Window&      getConformanceWindow() const                                    { return  m_conformanceWindow; }
  Window&            getScalingWindow()                                              { return  m_scalingWindow; }
  const Window&      getScalingWindow()                                        const { return  m_scalingWindow; }
  bool               isRefScaled( const PPS* pps ) const                             { return  unscaledPic->getPicWidthInLumaSamples()    != pps->getPicWidthInLumaSamples()                ||
                                                                                               unscaledPic->getPicHeightInLumaSamples()   != pps->getPicHeightInLumaSamples()               ||
                                                                                               getScalingWindow().getWindowLeftOffset()   != pps->getScalingWindow().getWindowLeftOffset()  ||
                                                                                               getScalingWindow().getWindowRightOffset()  != pps->getScalingWindow().getWindowRightOffset() ||
                                                                                               getScalingWindow().getWindowTopOffset()    != pps->getScalingWindow().getWindowTopOffset()   ||
                                                                                               getScalingWindow().getWindowBottomOffset() != pps->getScalingWindow().getWindowBottomOffset(); }
  bool               isWrapAroundEnabled( const PPS* pps ) const                     { return  pps->getWrapAroundEnabledFlag() && !isRefScaled( pps ); }

  void         allocateNewSlice();
  Slice        *swapSliceObject(Slice * p, uint32_t i);
  void         clearSliceBuffer();

  MCTSInfo     mctsInfo;
  std::vector<AQpLayer*> aqlayer;

  ChromaFormat m_chromaFormatIdc;
  BitDepths    m_bitDepths;

#if !KEEP_PRED_AND_RESI_SIGNALS
private:
  UnitArea m_ctuArea;
#endif

  std::vector<AlfMode> m_alfModes[MAX_NUM_COMPONENT];

  std::vector<SAOBlkParam> m_sao[2];

public:
  SAOBlkParam    *getSAO(int id = 0)                        { return &m_sao[id][0]; };
  void            resizeSAO(unsigned numEntries, int dstid) { m_sao[dstid].resize(numEntries); }
  void            copySAO(const Picture& src, int dstid)    { std::copy(src.m_sao[0].begin(), src.m_sao[0].end(), m_sao[dstid].begin()); }

#if ENABLE_QPA
  std::vector<double>     m_uEnerHpCtu;                         ///< CTU-wise L2 or squared L1 norm of high-passed luma input
  std::vector<Pel>        m_iOffsetCtu;                         ///< CTU-wise DC offset (later QP index offset) of luma input
#if ENABLE_QPA_SUB_CTU
  std::vector<int8_t>     m_subCtuQP;                           ///< sub-CTU-wise adapted QPs for delta-QP depth of 1 or more
#endif
#endif

  void copyAlfData(const Picture &p);
  void resizeAlfData(int numEntries);

  AlfMode *getAlfModes(int compIdx) { return m_alfModes[compIdx].data(); }

  #if NN_LF_UNIFIED
  NNFilterUnified::FilterParameters m_picprm;
  void initPicprms( const Slice& slice)
  {
    const SPS& sps = *slice.getSPS();
    const SPSVcmExt& spsVcmExtension  = sps.getSpsVCMExtension();
    NnlfUnifiedInferGranularity inferGranularity = slice.getNnlfUnifiedInferGranularity();
    m_picprm.block_size                          = spsVcmExtension.getNnlfUnifiedInferSize(inferGranularity);
    m_picprm.extension                           = spsVcmExtension.getNnlfUnifiedInfSizeExt();
    m_picprm.prmNum                              = spsVcmExtension.getNnlfUnifiedMaxNumPrms();
#if JVET_AG0130_UNIFIED_SR
    const PPS& pps = *slice.getPPS();
    if(pps.getPPSId() == ENC_PPS_ID_RPR)
    {
      m_picprm.nb_blocks_height = (pps.getPicHeightInLumaSamples() + m_picprm.block_size - 1) /  m_picprm.block_size;
      m_picprm.nb_blocks_width  = (pps.getPicWidthInLumaSamples() + m_picprm.block_size - 1) / m_picprm.block_size;
    }
    else
    {
      m_picprm.nb_blocks_height = (sps.getMaxPicHeightInLumaSamples() + m_picprm.block_size - 1) / m_picprm.block_size;
      m_picprm.nb_blocks_width  = (sps.getMaxPicWidthInLumaSamples() + m_picprm.block_size - 1) / m_picprm.block_size;
    }
 #else
    m_picprm.nb_blocks_height = (getPicHeightInLumaSamples() + m_picprm.block_size - 1) / m_picprm.block_size;
    m_picprm.nb_blocks_width  = (getPicWidthInLumaSamples() + m_picprm.block_size - 1) / m_picprm.block_size;
#endif
    m_picprm.prmId.resize(m_picprm.nb_blocks_height * m_picprm.nb_blocks_width);
    fill(m_picprm.prmId.begin(), m_picprm.prmId.end(), -1);
#if NN_HOP_UNIFIED_TEMPORAL_FILTERING
    m_picprm.temporal = spsVcmExtension.getNnlfHopTemporalFilteringEnabledFlag() && slice.getSliceType() != I_SLICE
                        && slice.getTLayer() >= NNFilterUnified::minimumTidUseTemporalFiltering;
#endif
  }
#endif
};

int calcAndPrintHashStatus(const CPelUnitBuf& pic, const class SEIDecodedPictureHash* pictureHashSEI, const BitDepths &bitDepths, const MsgLevel msgl);

uint32_t calcMD5(const CPelUnitBuf& pic, PictureHash &digest, const BitDepths &bitDepths);
uint32_t calcMD5WithCropping(const CPelUnitBuf &pic, PictureHash &digest, const BitDepths &bitDepths,
                             const int leftOffset, const int rightOffset, const int topOffset, const int bottomOffset);

std::string hashToString(const PictureHash &digest, int numChar);

typedef std::list<Picture*> PicList;

#endif
