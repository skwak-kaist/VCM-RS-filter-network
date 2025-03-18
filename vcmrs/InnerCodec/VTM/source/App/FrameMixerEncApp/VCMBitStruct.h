/* The copyright in this software is being made available under the BSD
 * License, included below. This software may be subject to other third party
 * and contributor rights, including patent rights, and no such rights are
 * granted under this license.
 *
 * Copyright (c) 2024-2034, Zhejiang University
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
 *  * Neither the name of the Zhejiang University nor the names of its contributors may
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

#define TEM_MPEG_148
#define KHU_MPEG_148

#include <vector>
#include <stdint.h>
#include "VLCWriter.h"

//------------------------------------------------------------------
//    VCM unit
//------------------------------------------------------------------
enum VCMUnitType
{
  VCM_VPS = 0,       // 0
  VCM_RSD,           // 1
  VCM_CVD,           // 2
  VCM_RSV_3  = 3,    // 3
  VCM_RSV_31 = 31,   // 31
  VCM_UNIT_INVALID   // 32
};

struct VCMUnit
{
  VCMUnitType     m_vcmUnitType;   ///< vuh_unit_type
  uint32_t        m_refVpsId;         ///< vuh_vps_id
  OutputBitstream m_bitstream;
  InputBitstream m_inBitstream;

  /** VPS syntax */
  uint32_t m_vpsId;
  uint32_t m_vpsBitsForPOCLsb;
  uint32_t m_vpsSpatialFlag;
  uint32_t m_vpsRetargetFlag;
  uint32_t m_vpsTemporalFlag;
  uint32_t m_vpsBitDepthShiftFlag;
  uint32_t m_vpsColorizerFlag;

  VCMUnit(VCMUnit& src)
    : m_vcmUnitType(src.m_vcmUnitType)
    , m_refVpsId(src.m_refVpsId)
  {m_bitstream.addSubstream(&src.m_bitstream);}
  /** construct an VCMUnit structure with given header values. */
  VCMUnit(VCMUnitType nalUnitType, int vpsId = 0)
    : m_vcmUnitType(nalUnitType)
    , m_refVpsId(vpsId)
  {}

  /** default constructor - no initialization; must be performed by user */
  VCMUnit() {}

  virtual ~VCMUnit() {}

  /** returns true if the VCM unit is a VPS */
  bool isVPS() { return m_vcmUnitType == VCM_VPS; }
  /** returns true if the VCM unit is a RSD */
  bool isRSD() { return m_vcmUnitType == VCM_RSD; }
  /** returns true if the VCM unit is a CVD */
  bool isCVD() { return m_vcmUnitType == VCM_CVD; }
  /** write VCM unit data */
  void writeHeader();
  void writeVCMPS();
  void writeRSD(OutputBitstream &inRsd);
  void writeCVD(OutputBitstream &inCvd);
  /** parse VCM unit data */
  void parseHeader();
  void parseVCMPS();
  void parseRSD(InputBitstream *inRsd);
  void parseCVD(OutputBitstream &inCvd);
};

//------------------------------------------------------------------
//    VCM NAL unit
//------------------------------------------------------------------
enum VCMNaluType
{
  VCM_NAL_SRD = 0,       // 0
  VCM_NAL_PRD,           // 1
  VCM_NAL_RSV_2,         // 2
  VCM_NAL_RSV_31 = 31,   // 31
  VCM_NAL_EOSS,           // 32
  VCM_NAL_SEI,           // 33

  VCM_NAL_RSV_34,   // 34
  VCM_NAL_RSV_59,   // 59
  VCM_NAL_UNSPEC_60,
  VCM_NAL_UNSPEC_63 = 63,   // 63
  VCM_NAL_UNIT_INVALID          // 64
};

struct ROIInfo
{
  uint32_t retargeting_flag;
  uint32_t rtg_image_size_len;
  uint32_t rtg_image_size_width;
  uint32_t rtg_image_size_height;
  uint32_t rtg_image_size_difference_flag;
  uint32_t rtg_to_output_difference_len;
  uint32_t rtg_to_output_difference_width;
  uint32_t rtg_to_output_difference_height;
  uint32_t rtg_rois_flag;
  uint32_t roi_size_len;
  uint32_t num_rois_len;
  uint32_t num_rois;
  std::vector<uint32_t> roi_scale_factor_flag;
  std::vector<uint32_t> roi_scale_factor;
  std::vector<uint32_t> roi_pos_x;
  std::vector<uint32_t> roi_pos_y;
  std::vector<uint32_t> roi_size_x;
  std::vector<uint32_t> roi_size_y;
};

struct ColorizerInfo
{
  uint32_t colorizer_enable_flag;
  uint32_t colorizer_index;
  //uint32_t luma_pre_shift; // not needed due to adopted m71663 (bit depth shift cleanup).
};

struct RSDInfo
{
  /* pad and ima
     TODO: these are temporal solution only for ima and the pad info in ima modeId. Should be solved later.
  */
  uint32_t pad_h;
  uint32_t pad_w;
  uint32_t imaOn_flag;
  /* Spatial*/
  uint32_t spatial_on_flag;
  uint32_t vcm_spatial_resampling_flag;
  uint32_t spatial_resampling_simple_flag;
  uint32_t spatial_resample_width;
  uint32_t spatial_resample_height;
  uint32_t spatial_resample_filter_idx;
  uint32_t scale_factor_id;
  /* ROI */
  uint32_t roi_on_flag;
  uint32_t roi_update_period_len;
  uint32_t roi_update_period;
  uint32_t roi_gops_num;
  std::list<ROIInfo> roiInfo;
  /* Temporal */
  uint32_t temporal_restoration_flag;
#ifdef TEM_MPEG_148
  uint32_t m_trdMode;
  uint32_t m_trd_inter_ratio_id;
  uint32_t m_trd_extra_resample_num_id;
  uint32_t m_trd_extra_predict_num_id;
  std::vector<bool> m_trd_pic_changed_flags;
  std::vector<uint32_t> m_trd_pic_modes;
  std::vector<uint32_t> m_trd_pic_inter_ratio_ids;
  std::vector<uint32_t> m_trd_pic_extra_resample_num_ids;
  std::vector<uint32_t> m_trd_pic_extra_predict_num_ids;
#endif
#ifdef KHU_MPEG_148
  uint32_t m_srd_trph_flag;
  std::vector<bool> m_trph_quality_valid_flag;
  std::vector<uint32_t> m_trph_quality_value;
#endif
  uint32_t m_trdRatioId;
  uint32_t m_trdRatio;
  uint32_t m_trdNumPicToBeRecon;
  uint32_t m_vcmFrameRate;
  uint32_t m_cvdFrameRate;
  uint32_t m_srd_ratio_change_allowed_flag;
  bool      m_vcmExtensionEnabledFlag;
  bool m_temporalRestorationEnabledFlag;
  std::vector<bool> m_picTemporalChangedFlags;
  std::vector<uint32_t> m_picTemporalRatioIndexes;
  std::vector<uint32_t> m_picTemporalChangedPOCs;
  uint32_t m_numTemporalRemain;
  /* Bit depth shift*/
  uint32_t bit_depth_on_flag;
  uint32_t bit_depth_shift_flag;
  uint32_t bit_depth_shift_luma;
  uint32_t bit_depth_shift_chroma;
  uint32_t bit_depth_luma_enhance;
  /* Colorization */
  uint32_t colorizer_on_flag;
  uint32_t colorizer_period;
  uint32_t colorizer_gops_num;
  std::list<ColorizerInfo> colorizerInfo;
};

enum RSDCompMap
{
  COMP_ROI = 0,       // 0
  COMP_SpatialResample, //1, 
  COMP_TemporalResample, //2, 
  COMP_PostFilter, //3,
  COMP_BitDepthTruncation, // 4,
  COMP_Colorize, //5,
  COMP_InnerCodec, //6,
  COMP_INVALID   // 7
};
struct VCMNalu
{
  VCMNaluType     m_nalUnitType;   ///< nal_unit_type
  uint32_t        m_temporalId;    ///< temporal_id
  uint32_t        m_forbiddenZeroBit;
  uint32_t        m_nuhReservedZeroBit;
  OutputBitstream m_bitstream;
  InputBitstream m_inBitstream;

  /* SRD syntax*/
  uint32_t  m_srdId;
  VCMUnit *m_refVPS;
  uint32_t  m_srd_ratio_change_allowed_flag;
  uint32_t m_srd_trph_flag;

  /* PRD syntax*/
  uint32_t  m_prdRefSrdId;
  uint32_t  m_prdPocLsb;
  uint32_t  m_prd_tr_ratio_changed_flag;
#ifdef TEM_MPEG_148
  uint32_t  m_prd_tr_update_mode;
  uint32_t  m_prd_tr_update_inter_ratio_index;
  uint32_t  m_prd_tr_update_extra_resample_num_index;
  uint32_t  m_prd_tr_update_extra_predict_num_index;
#endif
#ifdef KHU_MPEG_148  
  uint32_t  m_prd_trph_quality_valid_flag;
  uint32_t  m_prd_trph_quality_value;
#endif
  uint32_t  m_prd_tr_ratio_index;
  uint32_t  m_prdDecodedPicFlag;
  uint32_t  m_prdUseSrdTrdFlag;
  uint32_t  m_prdSrdTrdIdx;

  VCMNalu *m_refSRD;

  /* TRD syntax*/
  // struct
  struct TRD
  {
    uint32_t m_numRefPic;
    std::vector<int32_t> m_deltaPOC;
  };
  //SRD
  std::vector<TRD> m_srdTrdSet;
  uint32_t m_srdNumTrd;
  uint32_t m_srdTrdRatio;
  uint32_t m_srdTrdNumCodedFrame;
  //PRD
  TRD m_prdTrd;
  uint32_t m_prdPrevCodedPicPoc;

  // EOSS
  uint32_t m_numTemporalRemain;

  /* SEI syntax*/
  /* TODO */

  VCMNalu(VCMNalu& src)
    : m_nalUnitType(src.m_nalUnitType)
    , m_temporalId(src.m_temporalId)
    , m_forbiddenZeroBit(src.m_forbiddenZeroBit)
    , m_nuhReservedZeroBit(src.m_nuhReservedZeroBit)
  {m_bitstream.addSubstream(&src.m_bitstream);}
  /** construct an NALunit structure with given header values. */
  VCMNalu(VCMNaluType nalUnitType, int temporalId = 0, uint32_t nuhReservedZeroBit = 0, uint32_t forbiddenZeroBit = 0)
    : m_nalUnitType(nalUnitType)
    , m_temporalId(temporalId)
    , m_forbiddenZeroBit(forbiddenZeroBit)
    , m_nuhReservedZeroBit(nuhReservedZeroBit)
  {}

  /** default constructor - no initialization; must be performed by user */
  VCMNalu() {}

  virtual ~VCMNalu() {}

  /** returns true if the NALunit is a RDL NALunit */
  bool isRdl() { return m_nalUnitType == VCM_NAL_SRD || m_nalUnitType == VCM_NAL_PRD; }
  /** returns true if the NALunit is a SEI NALunit */
  bool isSei() { return m_nalUnitType == VCM_NAL_SEI; }
  /** write VCM NAL unit data */
  void writeVCMNaluHeader();
  void writeSRD(RSDInfo &rsdInfo);
  void writePRD();
  void writeSEI();
  void writeEOSS();
  void writeULVC(uint32_t value); // copy from VLCwriter. should consider reuse VLCwriter.
  void writeOneTRD(TRD &trd);
  void setSrdTrdSet();
  void setPrdTrd();
  /** parse VCM NAL unit data */
  void parseVCMNaluHeader();
  void parseSRD(RSDInfo &rsdInfo);
  void parsePRD(RSDInfo &rsdInfo);
  void parseEOSS(RSDInfo &rsdInfo);
  void parseSEI();
  void readULVC(uint32_t& value); 
  void readOneTRD(TRD &trd);
};

//------------------------------------------------------------------
//    VCM profile
//------------------------------------------------------------------
class VCMProfile
{
private:
  bool m_ptlTierFlag;
  int  m_ptlProfileCodecGroupIdc;
  int  m_ptlProfileRestorationIdc;
  int  m_ptlLevelIdc;

public:
  VCMProfile();
  virtual ~VCMProfile(){};

  int  getVPSId() const { return m_ptlProfileCodecGroupIdc; }
  void setVPSId(int i) { m_ptlProfileCodecGroupIdc = i; }
};

//------------------------------------------------------------------
//    VCM parameter set
//------------------------------------------------------------------
class VCMPS
{
private:
  int  m_refVpsId;
  int  m_log2MaxRestorationDataPOCLSB;
  bool m_spatialResampleEnableFlag;
  bool m_retargetingEnableFlag;
  bool m_temporalResampleEnableFlag;
  bool m_bitDepthShiftEnableFlag;

public:
  VCMPS();
  virtual ~VCMPS(){};

  int  getVPSId() const { return m_refVpsId; }
  void setVPSId(int i) { m_refVpsId = i; }
};

//------------------------------------------------------------------
//    VCM NAL sequence restoration data
//------------------------------------------------------------------
class SRD
{
private:
  int m_srdId;

public:
  SRD();
  virtual ~SRD(){};

  int  getSRDId() const { return m_srdId; }
  void setSRDId(int i) { m_srdId = i; }
};

//------------------------------------------------------------------
//    VCM NAL picture restoration data
//------------------------------------------------------------------
class PRD
{
private:
  int  m_refSRDId;
  bool m_spatialResampleEnableFlag;
  bool m_retargetingEnableFlag;
  bool m_temporalResampleEnableFlag;
  bool m_bitDepthShiftEnableFlag;

public:
  PRD();
  virtual ~PRD(){};

  int  getRefSRDId() const { return m_refSRDId; }
  void setRefSRDId(int i) { m_refSRDId = i; }
};

//------------------------------------------------------------------
//    VCM NAL picture restoration data
//------------------------------------------------------------------
class VCMHLSWriter : public HLSWriter
{
public:
  VCMHLSWriter() {}
  virtual ~VCMHLSWriter() {}

public:
  void codeVCMPS();
  void codeRSD();
  void codeCVD();
  void codeSRD();
  void codePRD();
  void codeSEI();
  void codeEOSS();

private:
};