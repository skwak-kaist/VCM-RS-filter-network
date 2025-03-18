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

#include "VideoParameterSet.h"
#include "VCMBitStruct.h"

/**
 * @brief VCM parameter set
 * 
 */

VCMPS::VCMPS()
  : m_refVpsId(0)
  , m_log2MaxRestorationDataPOCLSB(0)
  , m_spatialResampleEnableFlag(false)
  , m_retargetingEnableFlag(false)
  , m_temporalResampleEnableFlag(false)
  , m_bitDepthShiftEnableFlag(false)
{}

void VCMUnit::writeHeader() 
{
  m_bitstream.write(m_vcmUnitType, 5);
  if (m_vcmUnitType > 0)
  {
    m_bitstream.write(m_refVpsId, 4);
    m_bitstream.write(0, 23);   
  }
  else
  {
    m_bitstream.write(0, 27);
  }
}

void VCMUnit::writeVCMPS()
{
  m_bitstream.clear();
  writeHeader();

  m_bitstream.write(0, 1);
  m_bitstream.write(0, 7);
  m_bitstream.write(0, 8);
  m_bitstream.write(0, 8);

  m_bitstream.write(m_vpsId, 4);
  m_bitstream.write(m_vpsBitsForPOCLsb-4, 4);
  m_bitstream.write(m_vpsSpatialFlag, 1);
  m_bitstream.write(m_vpsRetargetFlag, 1);
  m_bitstream.write(m_vpsTemporalFlag, 1);
  m_bitstream.write(m_vpsBitDepthShiftFlag, 1);
  m_bitstream.write(m_vpsColorizerFlag, 1);
  m_bitstream.writeByteAlignment();
}

void VCMUnit::writeRSD(OutputBitstream& inRsd) 
{
  m_bitstream.clear();
  writeHeader();
  m_bitstream.addSubstream(&inRsd);
  //m_bitstream.writeByteAlignment();
}

void VCMUnit::writeCVD(OutputBitstream& inCvd) 
{
  m_bitstream.clear();
  writeHeader();
  m_bitstream.addSubstream(&inCvd);
  //m_bitstream.writeByteAlignment();
}

void VCMUnit::parseHeader() 
{
  uint32_t valueBits;
  uint32_t zeroBits;
  m_inBitstream.read(5, valueBits);
  m_vcmUnitType = VCMUnitType(valueBits);
  CHECK(m_vcmUnitType > 2, "Wrong VCM unit type" );

  if (m_vcmUnitType > 0)
  {
    m_inBitstream.read(4, m_refVpsId);
    m_inBitstream.read(23, zeroBits);
    CHECK(zeroBits != 0, "Found non-zero VCM unit header 1");
  }
  else
  {
    m_inBitstream.read(27, zeroBits);
    CHECK(zeroBits != 0, "Found non-zero VCM unit header 2");
  }
}

void VCMUnit::parseVCMPS() 
{
  uint32_t valueBits;

  m_inBitstream.read(1, valueBits);
  CHECK(valueBits != 0, "Found non-zero VCMPS 1");
  m_inBitstream.read(7, valueBits);
  CHECK(valueBits != 0, "Found non-zero VCMPS 2");
  m_inBitstream.read(8, valueBits);
  CHECK(valueBits != 0, "Found non-zero VCMPS 3");
  m_inBitstream.read(8, valueBits);
  CHECK(valueBits != 0, "Found non-zero VCMPS 4");

  m_inBitstream.read(4, m_vpsId);
  m_inBitstream.read(4, valueBits);
  m_vpsBitsForPOCLsb = valueBits + 4;
  m_inBitstream.read(1, m_vpsSpatialFlag);
  m_inBitstream.read(1, m_vpsRetargetFlag);
  m_inBitstream.read(1, m_vpsTemporalFlag);
  m_inBitstream.read(1, m_vpsBitDepthShiftFlag);
  m_inBitstream.read(1, m_vpsColorizerFlag);
  m_inBitstream.readByteAlignment();
}

void VCMUnit::parseRSD(InputBitstream* inRsd) 
{
  InputBitstream *tmpBs;
  tmpBs = m_inBitstream.extractSubstream(m_inBitstream.getNumBitsLeft());
  inRsd->getFifo().insert(inRsd->getFifo().end(), tmpBs->getFifo().begin(), tmpBs->getFifo().end()); // add the bits to the end.
}

void VCMUnit::parseCVD(OutputBitstream& inCvd) 
{
  InputBitstream *tmpBs;
  OutputBitstream tmpOutBs;
  tmpBs = m_inBitstream.extractSubstream(m_inBitstream.getNumBitsLeft());
  tmpOutBs.getFifo() = tmpBs->getFifo();
  inCvd.addSubstream(&tmpOutBs); // add the bits to the end.
}

void VCMNalu::writeVCMNaluHeader()
{
  m_forbiddenZeroBit = 0;
  m_nuhReservedZeroBit = 0;

  m_bitstream.write(m_forbiddenZeroBit, 1);
  m_bitstream.write(m_nalUnitType, 6);
  m_bitstream.write(m_temporalId, 3);
  m_bitstream.write(m_nuhReservedZeroBit, 6);
}

void VCMNalu::writeSRD(RSDInfo &rsdInfo)
{
  m_bitstream.clear();
  writeVCMNaluHeader();
  
  uint32_t timeScale = 27000000; // TODO: set this in config
  m_bitstream.write(m_srdId, 4);
  m_bitstream.write(timeScale / rsdInfo.m_vcmFrameRate, 32); // TODO: should consider various framerate
  m_bitstream.write(timeScale, 32);

  /* write sequence level restoration data*/
  if (m_refVPS->m_vpsSpatialFlag)
  {
    m_bitstream.write(rsdInfo.vcm_spatial_resampling_flag, 1);
    m_bitstream.write(rsdInfo.spatial_resampling_simple_flag, 1);
    if (rsdInfo.vcm_spatial_resampling_flag && !rsdInfo.spatial_resampling_simple_flag)
    {
      writeULVC(rsdInfo.spatial_resample_width);
      writeULVC(rsdInfo.spatial_resample_height);
      writeULVC(rsdInfo.spatial_resample_filter_idx);
    }
    if (rsdInfo.spatial_resampling_simple_flag)
    {
      m_bitstream.write(rsdInfo.scale_factor_id, 2);
    }
  }
  if (m_refVPS->m_vpsRetargetFlag)
  {
    CHECK(rsdInfo.roiInfo.empty(), "Number of ROI info is not sufficient 1!");

    m_bitstream.write(rsdInfo.roi_update_period_len, 5);
    m_bitstream.write(rsdInfo.roi_update_period, rsdInfo.roi_update_period_len);
    
    ROIInfo tmpRoiInfo = rsdInfo.roiInfo.front();
    m_bitstream.write(tmpRoiInfo.retargeting_flag, 1);
    if (tmpRoiInfo.retargeting_flag)
    {
      m_bitstream.write(tmpRoiInfo.rtg_image_size_len, 5);
      m_bitstream.write(tmpRoiInfo.rtg_image_size_width, tmpRoiInfo.rtg_image_size_len);
      m_bitstream.write(tmpRoiInfo.rtg_image_size_height, tmpRoiInfo.rtg_image_size_len);
      m_bitstream.write(tmpRoiInfo.rtg_image_size_difference_flag, 1);
      uint32_t orgWidth = tmpRoiInfo.rtg_image_size_width;
      uint32_t orgHeight = tmpRoiInfo.rtg_image_size_height;
      if (tmpRoiInfo.rtg_image_size_difference_flag)
      {
        m_bitstream.write(tmpRoiInfo.rtg_to_output_difference_len, 5);
        m_bitstream.write(tmpRoiInfo.rtg_to_output_difference_width, tmpRoiInfo.rtg_to_output_difference_len);
        m_bitstream.write(tmpRoiInfo.rtg_to_output_difference_height, tmpRoiInfo.rtg_to_output_difference_len);
        orgWidth += tmpRoiInfo.rtg_to_output_difference_width;
        orgHeight += tmpRoiInfo.rtg_to_output_difference_height;
      }
      uint32_t bitsPos = ceil(log2((orgWidth > orgHeight ? orgWidth: orgHeight) + 1));
      m_bitstream.write(tmpRoiInfo.rtg_rois_flag, 1);
      if (tmpRoiInfo.rtg_rois_flag)
      {
        m_bitstream.write(tmpRoiInfo.roi_size_len, 5);
        m_bitstream.write(tmpRoiInfo.num_rois_len, 4);
        m_bitstream.write(tmpRoiInfo.num_rois, tmpRoiInfo.num_rois_len);
        for (size_t i = 0; i < tmpRoiInfo.num_rois; i++)
        {
          if (i==0 || tmpRoiInfo.roi_scale_factor_flag[i - 1] != 15)
          {
            m_bitstream.write(tmpRoiInfo.roi_scale_factor_flag[i], 1);
            if (tmpRoiInfo.roi_scale_factor_flag[i])
            {
              m_bitstream.write(tmpRoiInfo.roi_scale_factor[i], 4);
            }
          }
          m_bitstream.write(tmpRoiInfo.roi_pos_x[i], bitsPos);
          m_bitstream.write(tmpRoiInfo.roi_pos_y[i], bitsPos);
          m_bitstream.write(tmpRoiInfo.roi_size_x[i], tmpRoiInfo.roi_size_len);
          m_bitstream.write(tmpRoiInfo.roi_size_y[i], tmpRoiInfo.roi_size_len);
        }
      }
    }
    rsdInfo.roiInfo.pop_front();
  }
  if (m_refVPS->m_vpsTemporalFlag)
  {
    m_bitstream.write(timeScale / rsdInfo.m_vcmFrameRate / rsdInfo.m_trdRatio, 32); // TODO: should consider various framerate
    m_bitstream.write(timeScale, 32);
    ///////////////////
    m_bitstream.write(rsdInfo.m_trdMode, 1);
    if (rsdInfo.m_trdMode == 0)
    {
      m_bitstream.write(rsdInfo.m_trd_inter_ratio_id, 2);
    }
    else if (rsdInfo.m_trdMode == 1)
    {
      m_bitstream.write(rsdInfo.m_trd_extra_resample_num_id, 2);
      m_bitstream.write(rsdInfo.m_trd_extra_predict_num_id, 2);
    }
    ///////////////////
    //m_bitstream.write(rsdInfo.m_trdRatioId, 2);
    m_bitstream.write(rsdInfo.m_srd_ratio_change_allowed_flag, 1);
    #ifdef KHU_MPEG_148    
    m_bitstream.write(rsdInfo.m_srd_trph_flag, 1);
    #endif
  }
  if (m_refVPS->m_vpsBitDepthShiftFlag)
  {
    m_bitstream.write(rsdInfo.bit_depth_shift_flag, 1);
    m_bitstream.write(rsdInfo.bit_depth_shift_luma, 3);
    m_bitstream.write(rsdInfo.bit_depth_shift_chroma, 3);
    m_bitstream.write(rsdInfo.bit_depth_luma_enhance, 1);
  }
  if (m_refVPS->m_vpsColorizerFlag)
  {
    CHECK(rsdInfo.colorizerInfo.empty(), "Number of Colorizer info is not sufficient!");

    m_bitstream.write(rsdInfo.colorizer_period, 8);
    
    ColorizerInfo tmpColorizerInfo = rsdInfo.colorizerInfo.front();
    m_bitstream.write(tmpColorizerInfo.colorizer_enable_flag, 1);
    if (tmpColorizerInfo.colorizer_enable_flag)
    {
      m_bitstream.write(tmpColorizerInfo.colorizer_index, 1);
      //m_bitstream.write(tmpColorizerInfo.luma_pre_shift, 3);// not needed due to adopted m71663 (bit depth shift cleanup).
    };
    rsdInfo.colorizerInfo.pop_front();
  }
  m_bitstream.writeByteAlignment();
}

void VCMNalu::writePRD() 
{
  m_bitstream.clear();
  writeVCMNaluHeader();

  m_bitstream.write(m_prdRefSrdId, 4);
  m_bitstream.write(m_prdPocLsb, m_refSRD->m_refVPS->m_vpsBitsForPOCLsb);
  if (m_refSRD->m_srd_ratio_change_allowed_flag)
  {
    m_bitstream.write(m_prd_tr_ratio_changed_flag, 1);
    if (m_prd_tr_ratio_changed_flag)
    {
 #ifdef TEM_MPEG_148
      ////////////////////////////////////
      m_bitstream.write(m_prd_tr_update_mode, 1);
      if (m_prd_tr_update_mode == 0)
      {
        m_bitstream.write(m_prd_tr_update_inter_ratio_index, 2);
      }
      else if (m_prd_tr_update_mode == 1)
      {
        m_bitstream.write(m_prd_tr_update_extra_resample_num_index, 2);
        m_bitstream.write(m_prd_tr_update_extra_predict_num_index, 2);
      }
#endif
      ////////////////////////////////////
      //m_bitstream.write(m_prd_tr_ratio_index, 2);
    }
  }
#ifdef KHU_MPEG_148
  if (m_refSRD->m_srd_trph_flag && m_prd_tr_update_mode == 0)
  {
    m_bitstream.write(m_prd_trph_quality_valid_flag, 1);
    m_bitstream.write(m_prd_trph_quality_value, 7);
  }
#endif
	m_bitstream.writeByteAlignment();
}

void VCMNalu::writeSEI() 
{
  m_bitstream.clear();
  writeVCMNaluHeader();
  /* TODO */
}

void VCMNalu::writeEOSS() 
{
  m_bitstream.clear();
  writeVCMNaluHeader();
  if (m_refVPS->m_vpsTemporalFlag)
  {
    m_bitstream.write(m_numTemporalRemain, 3);
  }
  m_bitstream.writeByteAlignment();
}

void VCMNalu::writeULVC(uint32_t value)
{
  uint32_t length   = 1;
  uint32_t temp     = ++value;

  CHECK(!temp, "Integer overflow");
  while (1 != temp)
  {
    temp >>= 1;
    length += 2;
  }
  // Take care of cases where length > 32
  m_bitstream.write(0, length >> 1);
  m_bitstream.write(value, (length + 1) >> 1);
}

void VCMNalu::writeOneTRD(TRD &trd)
{
  m_bitstream.write(trd.m_numRefPic - 1, 1);
  for (uint32_t i = 0; i < trd.m_numRefPic; i++)
  {
    writeULVC(abs(trd.m_deltaPOC[i]));
    m_bitstream.write(trd.m_deltaPOC[i]>=0 ? 1: 0, 1);
  }  
}

void VCMNalu::setSrdTrdSet() 
{
  // temporarily suppose that ratio is 2^n.
  m_srdNumTrd = log2(m_srdTrdRatio);
  m_srdTrdSet.resize(m_srdNumTrd);
  for (uint32_t i = 0; i < m_srdNumTrd; i++)
  {
    m_srdTrdSet[i].m_numRefPic = 2;
    m_srdTrdSet[i].m_deltaPOC.resize(m_srdTrdSet[i].m_numRefPic);
    m_srdTrdSet[i].m_deltaPOC[0] = -1 * (m_srdTrdRatio >> (i + 1));
    m_srdTrdSet[i].m_deltaPOC[1] = (m_srdTrdRatio >> (i + 1));
  }
}

void VCMNalu::setPrdTrd()
{
  // temporarily use ratio.
  if (!m_prdDecodedPicFlag)
  {
    // check whether it is the tail picture.
    bool isTail = (m_refSRD->m_srdTrdNumCodedFrame - 1 - m_prdPrevCodedPicPoc * m_refSRD->m_srdTrdRatio) < m_refSRD->m_srdTrdRatio;

    if (isTail)
    {
      m_prdTrd.m_numRefPic = 1;
      m_prdTrd.m_deltaPOC.resize(m_prdTrd.m_numRefPic);
      m_prdTrd.m_deltaPOC[0] = -1 * (m_prdPocLsb % m_refSRD->m_srdTrdRatio); // temporarily use the nearest previous coded picture by copying.
      m_prdUseSrdTrdFlag = 0;
    }
    else
    {
      // temporarily not support adaptive ratio until we have the list of changed ratio.
      m_prdSrdTrdIdx = 0;
      for (uint32_t i = 0; i < m_refSRD->m_srdNumTrd; i++)
      {
        if (!(m_prdPocLsb % (m_refSRD->m_srdTrdRatio >> (i+1))))
        {
          m_prdSrdTrdIdx = i;
          m_prdUseSrdTrdFlag = 1;
          break;
        }
      }
    }
  }
}

void VCMNalu::parseVCMNaluHeader() 
{
  uint32_t valueBits;

  m_inBitstream.read(1, m_forbiddenZeroBit);
  CHECK(m_forbiddenZeroBit != 0, "Found non-zero VCM NAL header 1");

  m_inBitstream.read(6, valueBits);
  m_nalUnitType = VCMNaluType(valueBits);
  CHECK(!(m_nalUnitType == VCM_NAL_SRD || m_nalUnitType == VCM_NAL_PRD || m_nalUnitType == VCM_NAL_SEI || m_nalUnitType == VCM_NAL_EOSS), "Found wrong VCM NAL type");
  
  m_inBitstream.read(3, m_temporalId);
  
  m_inBitstream.read(6, m_nuhReservedZeroBit);
  CHECK(m_nuhReservedZeroBit != 0, "Found non-zero VCM NAL header 2");
}

void VCMNalu::parseSRD(RSDInfo &rsdInfo) 
{
  m_inBitstream.read(4, m_srdId);

  uint32_t timeScale;
  uint32_t numUnitsInTick;
  m_inBitstream.read(32, numUnitsInTick);
  m_inBitstream.read(32, timeScale);
  rsdInfo.m_vcmFrameRate = timeScale / numUnitsInTick;

  rsdInfo.pad_h = 0;
  rsdInfo.pad_w = 0;
  rsdInfo.imaOn_flag = 0;

  /* parse sequence level temporal restoration data*/
  // init
  rsdInfo.vcm_spatial_resampling_flag = 10;
  rsdInfo.spatial_resampling_simple_flag = 0;
  rsdInfo.spatial_resample_width = 0;
  rsdInfo.spatial_resample_height = 0;
  rsdInfo.spatial_resample_filter_idx=0;
  rsdInfo.scale_factor_id = -1;
  if (m_refVPS->m_vpsSpatialFlag)
  {
    m_inBitstream.read(1, rsdInfo.vcm_spatial_resampling_flag);
    m_inBitstream.read(1, rsdInfo.spatial_resampling_simple_flag);
    
    if (rsdInfo.vcm_spatial_resampling_flag && !rsdInfo.spatial_resampling_simple_flag)
    {
      readULVC(rsdInfo.spatial_resample_width);
      readULVC(rsdInfo.spatial_resample_height);
      readULVC(rsdInfo.spatial_resample_filter_idx);
    }
    if (rsdInfo.spatial_resampling_simple_flag)
    {
      m_inBitstream.read(2, rsdInfo.scale_factor_id);
    }
  }
  // init
  rsdInfo.roi_update_period = 0;
  if (m_refVPS->m_vpsRetargetFlag)
  {
    m_inBitstream.read(5, rsdInfo.roi_update_period_len);
    m_inBitstream.read(rsdInfo.roi_update_period_len, rsdInfo.roi_update_period);

    ROIInfo tmpRoiInfo;
    m_inBitstream.read(1, tmpRoiInfo.retargeting_flag);
    if (tmpRoiInfo.retargeting_flag)
    {
      m_inBitstream.read(5, tmpRoiInfo.rtg_image_size_len);
      m_inBitstream.read(tmpRoiInfo.rtg_image_size_len, tmpRoiInfo.rtg_image_size_width);
      m_inBitstream.read(tmpRoiInfo.rtg_image_size_len, tmpRoiInfo.rtg_image_size_height);
      m_inBitstream.read(1, tmpRoiInfo.rtg_image_size_difference_flag);
      uint32_t orgWidth = tmpRoiInfo.rtg_image_size_width;
      uint32_t orgHeight = tmpRoiInfo.rtg_image_size_height;
      if (tmpRoiInfo.rtg_image_size_difference_flag)
      {
        m_inBitstream.read(5, tmpRoiInfo.rtg_to_output_difference_len);
        m_inBitstream.read(tmpRoiInfo.rtg_to_output_difference_len, tmpRoiInfo.rtg_to_output_difference_width);
        m_inBitstream.read(tmpRoiInfo.rtg_to_output_difference_len, tmpRoiInfo.rtg_to_output_difference_height);
        orgWidth += tmpRoiInfo.rtg_to_output_difference_width;
        orgHeight += tmpRoiInfo.rtg_to_output_difference_height;
      }
      uint32_t bitsPos = ceil(log2((orgWidth > orgHeight ? orgWidth: orgHeight) + 1));
      m_inBitstream.read(1, tmpRoiInfo.rtg_rois_flag);
      if (tmpRoiInfo.rtg_rois_flag)
      {
        m_inBitstream.read(5, tmpRoiInfo.roi_size_len);
        m_inBitstream.read(4, tmpRoiInfo.num_rois_len);
        m_inBitstream.read(tmpRoiInfo.num_rois_len, tmpRoiInfo.num_rois);
        tmpRoiInfo.roi_scale_factor_flag.resize(tmpRoiInfo.num_rois);
        tmpRoiInfo.roi_scale_factor.resize(tmpRoiInfo.num_rois);
        tmpRoiInfo.roi_pos_x.resize(tmpRoiInfo.num_rois);
        tmpRoiInfo.roi_pos_y.resize(tmpRoiInfo.num_rois);
        tmpRoiInfo.roi_size_x.resize(tmpRoiInfo.num_rois);
        tmpRoiInfo.roi_size_y.resize(tmpRoiInfo.num_rois);
        for (size_t i = 0; i < tmpRoiInfo.num_rois; i++)
        {
          if (i==0 || tmpRoiInfo.roi_scale_factor_flag[i - 1] != 15)
          {
            m_inBitstream.read(1, tmpRoiInfo.roi_scale_factor_flag[i]);
            if (tmpRoiInfo.roi_scale_factor_flag[i])
            {
              m_inBitstream.read(4, tmpRoiInfo.roi_scale_factor[i]);
            }
          }
          m_inBitstream.read(bitsPos, tmpRoiInfo.roi_pos_x[i]);
          m_inBitstream.read(bitsPos, tmpRoiInfo.roi_pos_y[i]);
          m_inBitstream.read(tmpRoiInfo.roi_size_len, tmpRoiInfo.roi_size_x[i]);
          m_inBitstream.read(tmpRoiInfo.roi_size_len, tmpRoiInfo.roi_size_y[i]);
        }
      }
    }
    rsdInfo.roiInfo.push_back(tmpRoiInfo);
  }
  // init
  rsdInfo.temporal_restoration_flag = m_refVPS->m_vpsTemporalFlag;
  rsdInfo.m_vcmExtensionEnabledFlag = m_refVPS->m_vpsTemporalFlag;
  rsdInfo.m_temporalRestorationEnabledFlag = m_refVPS->m_vpsTemporalFlag;
  rsdInfo.m_trdRatioId = 0;
  rsdInfo.m_trdRatio  = 4;
  rsdInfo.m_trdNumPicToBeRecon = 0;
  //rsdInfo.m_numTemporalRemain = 0;
  rsdInfo.m_vcmExtensionEnabledFlag = true;
  rsdInfo.m_temporalRestorationEnabledFlag = true;
  //rsdInfo.m_picTemporalChangedFlags.resize(0);
  //rsdInfo.m_picTemporalRatioIndexes.resize(0);
  //rsdInfo.m_picTemporalChangedPOCs.resize(0);
#ifdef TEM_MPEG_148
  rsdInfo.m_trdMode = 0;
  rsdInfo.m_trd_inter_ratio_id = 1;
  rsdInfo.m_trd_extra_resample_num_id = 0;
  rsdInfo.m_trd_extra_predict_num_id = 1;
  //rsdInfo.m_trd_pic_changed_flags.resize(0);
  //rsdInfo.m_trd_pic_modes.resize(0);
  //rsdInfo.m_trd_pic_inter_ratio_ids.resize(0);
  //rsdInfo.m_trd_pic_extra_resample_num_ids.resize(0);
  //rsdInfo.m_trd_pic_extra_predict_num_ids.resize(0);
#endif
#ifdef KHU_MPEG_148
  rsdInfo.m_srd_trph_flag = false;
#endif

  if (m_refVPS->m_vpsTemporalFlag)
  {
    m_inBitstream.read(32, numUnitsInTick);
    m_inBitstream.read(32, timeScale);
    rsdInfo.m_cvdFrameRate = timeScale / numUnitsInTick;
 #ifdef TEM_MPEG_148
    m_inBitstream.read(1, rsdInfo.m_trdMode);
    if (rsdInfo.m_trdMode == 0)
    {
      m_inBitstream.read(2, rsdInfo.m_trd_inter_ratio_id);
    }
    else if (rsdInfo.m_trdMode == 1)
    {
      m_inBitstream.read(2, rsdInfo.m_trd_extra_resample_num_id);
      m_inBitstream.read(2, rsdInfo.m_trd_extra_predict_num_id);
    }
 #endif
    //m_inBitstream.read(2, rsdInfo.m_trdRatioId);
    rsdInfo.m_trdRatioId = rsdInfo.m_trd_inter_ratio_id;
    rsdInfo.m_trdRatio = 1 << (rsdInfo.m_trdRatioId + 1);

    m_inBitstream.read(1, rsdInfo.m_srd_ratio_change_allowed_flag);
    m_srd_ratio_change_allowed_flag = rsdInfo.m_srd_ratio_change_allowed_flag;
#ifdef KHU_MPEG_148    
    m_inBitstream.read(1, rsdInfo.m_srd_trph_flag);
    m_srd_trph_flag = rsdInfo.m_srd_trph_flag;
#endif
  }
  else{
    rsdInfo.m_srd_ratio_change_allowed_flag = 0;
    m_srd_ratio_change_allowed_flag = rsdInfo.m_srd_ratio_change_allowed_flag;
#ifdef KHU_MPEG_148        
    rsdInfo.m_srd_trph_flag = 0;
    m_srd_trph_flag = rsdInfo.m_srd_trph_flag;
#endif    
  }
  // init
  rsdInfo.bit_depth_shift_flag = 0;
  rsdInfo.bit_depth_shift_luma = 0;
  rsdInfo.bit_depth_shift_chroma = 0;
  rsdInfo.bit_depth_luma_enhance = 0;
  if (m_refVPS->m_vpsBitDepthShiftFlag)
  {
    m_inBitstream.read(1, rsdInfo.bit_depth_shift_flag);
    m_inBitstream.read(3, rsdInfo.bit_depth_shift_luma);
    m_inBitstream.read(3, rsdInfo.bit_depth_shift_chroma);
    m_inBitstream.read(1, rsdInfo.bit_depth_luma_enhance);
  }
  
  // init
  rsdInfo.colorizer_period = 0;
  if (m_refVPS->m_vpsColorizerFlag)
  {
    m_inBitstream.read(8, rsdInfo.colorizer_period);
    ColorizerInfo tmpColorizerInfo;
    m_inBitstream.read(1, tmpColorizerInfo.colorizer_enable_flag);
    if (tmpColorizerInfo.colorizer_enable_flag)
    {
      m_inBitstream.read(1, tmpColorizerInfo.colorizer_index);
      //m_inBitstream.read(3, tmpColorizerInfo.luma_pre_shift); //  not needed due to adopted m71663 (bit depth shift cleanup).
    }
    rsdInfo.colorizerInfo.push_back(tmpColorizerInfo);
  }
  
  m_inBitstream.readByteAlignment();
}

void VCMNalu::parsePRD(RSDInfo &rsdInfo)
{
  m_inBitstream.read(4, m_prdRefSrdId);
  CHECK(m_prdRefSrdId != m_refSRD->m_srdId, "Found wrong referenced SRD ID in PRD"); // temporary solution.

  m_inBitstream.read(m_refSRD->m_refVPS->m_vpsBitsForPOCLsb, m_prdPocLsb);

  if (m_refSRD->m_refVPS->m_vpsTemporalFlag && m_refSRD->m_srd_ratio_change_allowed_flag)
  {
    m_inBitstream.read(1, m_prd_tr_ratio_changed_flag);
#ifdef TEM_MPEG_148
    rsdInfo.m_trd_pic_changed_flags.push_back(m_prd_tr_ratio_changed_flag);
    if (m_prd_tr_ratio_changed_flag)
    {
      m_inBitstream.read(1, m_prd_tr_update_mode);
      rsdInfo.m_trd_pic_modes.push_back(m_prd_tr_update_mode);
      if (m_prd_tr_update_mode == 0)
      {
        m_inBitstream.read(2, m_prd_tr_update_inter_ratio_index);
        rsdInfo.m_trd_pic_inter_ratio_ids.push_back(m_prd_tr_update_inter_ratio_index);
      }
      else if (m_prd_tr_update_mode == 1)
      {
        m_inBitstream.read(2, m_prd_tr_update_extra_resample_num_index);
        m_inBitstream.read(2, m_prd_tr_update_extra_predict_num_index);
        rsdInfo.m_trd_pic_extra_resample_num_ids.push_back(m_prd_tr_update_extra_resample_num_index);
        rsdInfo.m_trd_pic_extra_predict_num_ids.push_back(m_prd_tr_update_extra_predict_num_index);
      }
    }
#else
    rsdInfo.m_picTemporalChangedFlags.push_back(m_prd_tr_ratio_changed_flag);
    m_prd_tr_ratio_index = 0;
    if (m_prd_tr_ratio_changed_flag)
    {
      m_inBitstream.read(2, m_prd_tr_ratio_index);
    }
    rsdInfo.m_picTemporalRatioIndexes.push_back(m_prd_tr_ratio_index);
#endif

    rsdInfo.m_picTemporalChangedPOCs.push_back(m_prdPocLsb);
  }
#ifdef KHU_MPEG_148  
  if (m_refSRD->m_refVPS->m_vpsTemporalFlag && m_refSRD->m_srd_trph_flag)
  {
    m_inBitstream.read(1, m_prd_trph_quality_valid_flag);
    m_inBitstream.read(7, m_prd_trph_quality_value);
    rsdInfo.m_trph_quality_valid_flag.push_back(m_prd_trph_quality_valid_flag);
    rsdInfo.m_trph_quality_value.push_back(m_prd_trph_quality_value);
  }
#endif  
  m_inBitstream.readByteAlignment();
}

void VCMNalu::parseEOSS(RSDInfo &rsdInfo)
{
  if (m_refVPS->m_vpsTemporalFlag)
  {
    m_inBitstream.read(3, rsdInfo.m_numTemporalRemain);
  }
  m_inBitstream.readByteAlignment();
}

void VCMNalu::parseSEI() 
{
  // TODO
}

void VCMNalu::readULVC(uint32_t& value)
{
  uint32_t suffix    = 0;
  uint32_t prefixBit = 0;
  m_inBitstream.read( 1, prefixBit );

  if( 0 == prefixBit )
  {
    uint32_t length = 0;

    while( prefixBit == 0 )
    {
      m_inBitstream.read( 1, prefixBit );
      length++;
    }

    m_inBitstream.read(length, suffix);
    suffix += (1 << length) - 1;

  }
  value = suffix;
}

void VCMNalu::readOneTRD(TRD &trd) 
{
  uint32_t valueBits;

  m_inBitstream.read(1, valueBits);
  trd.m_numRefPic = valueBits + 1;
  trd.m_deltaPOC.resize(trd.m_numRefPic);
  for (uint32_t i = 0; i < trd.m_numRefPic; i++)
  {
    readULVC(valueBits);
    trd.m_deltaPOC[i] = valueBits;
    m_inBitstream.read(1, valueBits);
    trd.m_deltaPOC[i] *= valueBits ? 1: -1;
  }
}
