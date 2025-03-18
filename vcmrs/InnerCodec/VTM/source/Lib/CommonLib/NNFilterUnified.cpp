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


#include "Picture.h"
#include "NNFilterUnified.h"
#include "NNInference.h"
//#include<sadl / model.h>
#include "../../sadl/sadl/model.h"

#include <fstream>
using namespace std;

// some constants
static constexpr int log2InputQpScale  = 6;
static constexpr int log2InputIbpScale = 0;
static constexpr int defaultInputSize  = 128;   // block size
static constexpr int defaultBlockExt   = 8;


struct Input
{
  enum
  {
    Rec = 0,
    Pred,
    BS,
    QPbase,
    QPSlice,
    IPB,
    nbInputs
  };
};
#if NN_HOP_UNIFIED_TEMPORAL_FILTERING
struct InputTemporal
{
  enum
  {
    Rec = 0,
    Pred,
    List0,
    List1,
    QPbase,
    nbInputsTemporal
  };
};
#endif

void NNFilterUnified::init(const std::string &filename, int picWidth, int picHeight, ChromaFormat format, int prmNum)
{
  ifstream file(filename, ios::binary);
  if (!file) {
    cerr << "[ERROR] unable to open NNFilter model " << filename << endl;
    exit(-1);
  }
  if (!m_model)
  {
    m_model.reset(new sadl::Model<TypeSadlLFUnified>());
    if (!m_model->load(file))
    {
      cerr << "[ERROR] issue loading model NNFilter " << filename << endl;
      exit(-1);
    }
  }

  // prepare inputs
  m_inputs.resize(Input::nbInputs);
  resizeInputs(defaultInputSize + defaultBlockExt * 2, defaultInputSize + defaultBlockExt * 2);

  if (m_filtered.size() > 0 || m_scaled[0].size() > 0)
  {
    return;
  }

  m_filtered.resize(prmNum);
  for (int i = 0; i < prmNum; i++)
  {
    m_filtered[i].create(format, Area(0, 0, picWidth, picHeight));
  }
  for (int j = 0; j < 3; j++)
  {
    m_scaled[j].resize(prmNum);
    for (int i = 0; i < prmNum; i++)
    {
      m_scaled[j][i].create(format, Area(0, 0, picWidth, picHeight));
    }
  }
}
#if NN_HOP_UNIFIED_TEMPORAL_FILTERING
void NNFilterUnified::initTemporal(const std::string &filename)
{
  ifstream file(filename, ios::binary);
  if (!file) {
    std::cerr << "[ERROR] unable to open NNFilter temporal model " << filename << std::endl;
    exit(-1);
  }
  if (!m_modelTemporal)
  {
    m_modelTemporal.reset(new sadl::Model<TypeSadlLFUnified>());
    if (!m_modelTemporal->load(file))
    {
      cerr << "[ERROR] issue loading temporal model NNFilter " << filename << endl;
      exit(-1);
    }
  }
  // prepare inputs
  m_inputsTemporal.resize(InputTemporal::nbInputsTemporal);
  resizeInputsTemporal(defaultInputSize + defaultBlockExt * 2, defaultInputSize + defaultBlockExt * 2);
}
#endif

void NNFilterUnified::destroy()
{
  for (int i = 0; i < (int)m_filtered.size(); i++)
  {
    m_filtered[i].destroy();
  }
  for (int j = 0; j < 3; j++)
  {
    for (int i = 0; i < (int)m_scaled[j].size(); i++)
    {
      m_scaled[j][i].destroy();
    }
  }
}

// default is square block + extension
void NNFilterUnified::resizeInputs(int width, int height)
{
  int sizeW = width;
  int sizeH = height;
  if (sizeH == m_blocksize[0] && sizeW == m_blocksize[1])
  {
    return;
  }
  m_blocksize[0] = sizeH;
  m_blocksize[1] = sizeW;
  // note: later QP inputs can be optimized to avoid some duplicate computation with a 1x1 input
  m_inputs[Input::Rec].resize(sadl::Dimensions({ 1, sizeH, sizeW, 3 }));
  m_inputs[Input::Pred].resize(sadl::Dimensions({ 1, sizeH, sizeW, 3 }));
  m_inputs[Input::BS].resize(sadl::Dimensions({ 1, sizeH, sizeW, 3 }));
  m_inputs[Input::QPbase].resize(sadl::Dimensions({ 1, sizeH, sizeW, 1 }));
  m_inputs[Input::QPSlice].resize(sadl::Dimensions({ 1, sizeH, sizeW, 1 }));
  m_inputs[Input::IPB].resize(sadl::Dimensions({ 1, sizeH, sizeW, 1 }));

  if (!m_model->init(m_inputs))
  {
    cerr << "[ERROR] issue init model NNFilterUnified " << endl;
    exit(-1);
  }
  assert(nb_inputs==m_inputs.size());
  for(int i=0;i<nb_inputs;++i)
    m_input_quantizer[i] = m_model->getInputsTemplate()[i].quantizer;

}
#if NN_HOP_UNIFIED_TEMPORAL_FILTERING
// default is square block + extension
void NNFilterUnified::resizeInputsTemporal(int width, int height) // assume using same quantizer as the primary model
{
  int sizeW = width;
  int sizeH = height;
  if (sizeH == m_blocksizeTemporal[0] && sizeW == m_blocksizeTemporal[1])
  {
    return;
  }
  m_blocksizeTemporal[0] = sizeH;
  m_blocksizeTemporal[1] = sizeW;
  // note: later QP inputs can be optimized to avoid some duplicate computation with a 1x1 input
  m_inputsTemporal[InputTemporal::Rec].resize(sadl::Dimensions({ 1, sizeH, sizeW, 1 }));
  m_inputsTemporal[InputTemporal::Pred].resize(sadl::Dimensions({ 1, sizeH, sizeW, 1 }));
  m_inputsTemporal[InputTemporal::List0].resize(sadl::Dimensions({ 1, sizeH, sizeW, 1 }));
  m_inputsTemporal[InputTemporal::List1].resize(sadl::Dimensions({ 1, sizeH, sizeW, 1 }));
  m_inputsTemporal[InputTemporal::QPbase].resize(sadl::Dimensions({ 1, sizeH, sizeW, 1 }));

  if (!m_modelTemporal->init(m_inputsTemporal))
  {
    cerr << "[ERROR] issue init temporal model NNFilterHOP " << endl;
    exit(-1);
  }
  m_input_quantizer_temporal = m_modelTemporal->getInputsTemplate()[0].quantizer;   // assume all image inputs have same quantizer

}
#endif
void roundToOutputBitdepth(const PelUnitBuf &src, PelUnitBuf &dst, const ClpRngs &clpRngs)
{
  for (int c = 0; c < MAX_NUM_COMPONENT; c++)
  {
    const int shift  = NNFilterUnified::log2OutputScale - clpRngs.comp[ComponentID(c)].bd;
    const int offset = 1 << (shift - 1);
    assert(shift >= 0);
    const PelBuf &bufSrc = src.get(ComponentID(c));
    PelBuf &      bufDst = dst.get(ComponentID(c));
    const int     width  = bufSrc.width;
    const int     height = bufSrc.height;
    for (int y = 0; y < height; ++y)
    {
      for (int x = 0; x < width; ++x)
      {
        bufDst.at(x, y) = Pel(Clip3<int>(0, 1023, (bufSrc.at(x, y) + offset) >> shift));
      }
    }
  }
}

// bufDst: temporary buffer to store results
// inferArea: area used for inference (include extension)
template<typename T>
static void extractOutputs(const Picture &pic, sadl::Model<T> &m, PelUnitBuf &bufDst, UnitArea inferArea, int extLeft,
                           int extRight, int extTop, int extBottom)
{
  const int log2InputBitdepth = pic.cs->slice->clpRng(COMPONENT_Y).bd;   // internal bitdepth
  auto      output            = m.result(0);
  const int q_output_sadl     = output.quantizer;
  const int shiftInput        = NNFilterUnified::log2OutputScale - log2InputBitdepth;
  const int shiftOutput       = NNFilterUnified::log2OutputScale - q_output_sadl;
  assert(shiftInput >= 0);
  assert(shiftOutput >= 0);

  const int width   = bufDst.Y().width;
  const int height  = bufDst.Y().height;
  PelBuf &  bufDstY = bufDst.get(COMPONENT_Y);
  CPelBuf   bufRecY = pic.getRecBeforeDbfBuf(inferArea).get(COMPONENT_Y);

  for (int c = 0; c < 4; ++c)   // unshuffle on C++ side
  {
    for (int y = 0; y < height / 2; ++y)
    {
      for (int x = 0; x < width / 2; ++x)
      {
        int yy = (y * 2) + c / 2;
        int xx = (x * 2) + c % 2;
        if (xx < extLeft || yy < extTop || xx >= width - extRight || yy >= height - extBottom)
        {
          continue;
        }
        int out;
        if constexpr (std::is_same<TypeSadlLFUnified, float>::value)
        {
          out = round((output(0, y, x, c) * (1 << shiftOutput) + (float) bufRecY.at(xx, yy) * (1 << shiftInput)));
        }
        else
        {
          out = ((output(0, y, x, c) << shiftOutput) + (bufRecY.at(xx, yy) << shiftInput));
        }

        bufDstY.at(xx, yy) = Pel(Clip3<int>(0, (1 << NNFilterUnified::log2OutputScale) - 1, out));
      }
    }
  }

  PelBuf &bufDstCb = bufDst.get(COMPONENT_Cb);
  PelBuf &bufDstCr = bufDst.get(COMPONENT_Cr);
  CPelBuf bufRecCb = pic.getRecBeforeDbfBuf(inferArea).get(COMPONENT_Cb);
  CPelBuf bufRecCr = pic.getRecBeforeDbfBuf(inferArea).get(COMPONENT_Cr);

  for (int y = 0; y < height / 2; ++y)
  {
    for (int x = 0; x < width / 2; ++x)
    {
      if (x < extLeft / 2 || y < extTop / 2 || x >= width / 2 - extRight / 2 || y >= height / 2 - extBottom / 2)
      {
        continue;
      }

      int outCb;
      int outCr;
      if constexpr (std::is_same<TypeSadlLFUnified, float>::value)
      {
        outCb = round(output(0, y, x, 4) * (1 << shiftOutput) + (float) bufRecCb.at(x, y) * (1 << shiftInput));
        outCr = round(output(0, y, x, 5) * (1 << shiftOutput) + (float) bufRecCr.at(x, y) * (1 << shiftInput));
      }
      else
      {
        outCb = ((output(0, y, x, 4) << shiftOutput) + (bufRecCb.at(x, y) << shiftInput));
        outCr = ((output(0, y, x, 5) << shiftOutput) + (bufRecCr.at(x, y) << shiftInput));
      }

      bufDstCb.at(x, y) = Pel(Clip3<int>(0, (1 << NNFilterUnified::log2OutputScale) - 1, outCb));
      bufDstCr.at(x, y) = Pel(Clip3<int>(0, (1 << NNFilterUnified::log2OutputScale) - 1, outCr));
    }
  }
}
#if NN_HOP_UNIFIED_TEMPORAL_FILTERING
// bufDst: temporary buffer to store results
// inferArea: area used for inference (include extension)
template<typename T>
static void extractOutputsTemporal(const Picture &pic, sadl::Model<T> &m, PelUnitBuf &bufDst, UnitArea inferArea, int extLeft,
                           int extRight, int extTop, int extBottom)
{
  const int log2InputBitdepth = pic.cs->slice->clpRng(COMPONENT_Y).bd;   // internal bitdepth
  auto      output            = m.result(0);
  const int q_output_sadl     = output.quantizer;
  const int shiftInput        = NNFilterUnified::log2OutputScale - log2InputBitdepth;
  const int shiftOutput       = NNFilterUnified::log2OutputScale - q_output_sadl;
  assert(shiftInput >= 0);
  assert(shiftOutput >= 0);

  const int width   = bufDst.Y().width;
  const int height  = bufDst.Y().height;
  PelBuf &  bufDstY = bufDst.get(COMPONENT_Y);
  CPelBuf   bufRecY = pic.getRecBeforeDbfBuf(inferArea).get(COMPONENT_Y);

  for (int c = 0; c < 4; ++c)   // unshuffle on C++ side
  {
    for (int y = 0; y < height / 2; ++y)
    {
      for (int x = 0; x < width / 2; ++x)
      {
        int yy = (y * 2) + c / 2;
        int xx = (x * 2) + c % 2;
        if (xx < extLeft || yy < extTop || xx >= width - extRight || yy >= height - extBottom)
        {
          continue;
        }
        int out;
        if constexpr (std::is_same<TypeSadlLFUnified, float>::value)
        {
          out = round((output(0, y, x, c) * (1 << shiftOutput) + (float) bufRecY.at(xx, yy) * (1 << shiftInput)));
        }
        else
        {
          out = ((output(0, y, x, c) << shiftOutput) + (bufRecY.at(xx, yy) << shiftInput));
        }

        bufDstY.at(xx, yy) = Pel(Clip3<int>(0, (1 << NNFilterUnified::log2OutputScale) - 1, out));
      }
    }
  }
  
  PelBuf &bufDstCb = bufDst.get(COMPONENT_Cb);
  PelBuf &bufDstCr = bufDst.get(COMPONENT_Cr);
  CPelBuf bufRecCb = pic.getRecoBuf(inferArea).get(COMPONENT_Cb);
  CPelBuf bufRecCr = pic.getRecoBuf(inferArea).get(COMPONENT_Cr);

  for (int y = 0; y < height / 2; ++y)
  {
    for (int x = 0; x < width / 2; ++x)
    {
      if (x < extLeft / 2 || y < extTop / 2 || x >= width / 2 - extRight / 2 || y >= height / 2 - extBottom / 2)
      {
        continue;
      }

      int outCb = (bufRecCb.at(x, y) << shiftInput);
      int outCr = (bufRecCr.at(x, y) << shiftInput);

      bufDstCb.at(x, y) = Pel(Clip3<int>(0, (1 << NNFilterUnified::log2OutputScale) - 1, outCb));
      bufDstCr.at(x, y) = Pel(Clip3<int>(0, (1 << NNFilterUnified::log2OutputScale) - 1, outCr));
    }
  }
}
#endif

void NNFilterUnified::filterBlock(Picture &pic, UnitArea inferArea, int extLeft, int extRight, int extTop, int extBottom,
                              int prmId)
{
  // get model
  auto &model    = *m_model;
  bool  inter    = pic.slices[0]->getSliceType() != I_SLICE ? true : false;
  int   qpOffset = (inter ? prmId * 5 : prmId * 2) * (pic.slices[0]->getTLayer() >= 4 ? 1 : -1);
  int   seqQp    = pic.slices[0]->getPPS()->getPicInitQPMinus26() + 26 + qpOffset;
  int   sliceQp  = pic.slices[0]->getSliceQp();
  resizeInputs(inferArea.Y().width, inferArea.Y().height);

  const int    log2InputBitdepth = pic.cs->slice->clpRng(COMPONENT_Y).bd;   // internal bitdepth
  const double inputScalePred    = (1 << log2InputBitdepth);
  const double inputScaleQp      = (1 << log2InputQpScale);
  const double inputScaleIpb     = (1 << log2InputIbpScale);

  std::vector<InputData> listInputData;
  listInputData.push_back({ NN_INPUT_REC, 0, inputScalePred, m_input_quantizer[0] - log2InputBitdepth, true, true });
  listInputData.push_back({ NN_INPUT_PRED, 1, inputScalePred, m_input_quantizer[1] - log2InputBitdepth, true, true });
  listInputData.push_back({ NN_INPUT_BS, 2, inputScalePred, m_input_quantizer[2] - log2InputBitdepth, true, true });
  listInputData.push_back({ NN_INPUT_GLOBAL_QP, 3, inputScaleQp, m_input_quantizer[3] - log2InputQpScale, true, false });
  listInputData.push_back({ NN_INPUT_LOCAL_QP, 4, inputScaleQp, m_input_quantizer[4] - log2InputQpScale, true, false });
#if NN_LF_FORCE_USE
  if (m_forceIntraType) {
    listInputData.push_back({ NN_INPUT_ZERO, 5, inputScaleIpb, m_input_quantizer[5] - log2InputIbpScale, true, false });
  } else
#endif
  listInputData.push_back({ NN_INPUT_IPB, 5, inputScaleIpb, m_input_quantizer[5] - log2InputIbpScale, true, false });

  NNInference::prepareInputs<TypeSadlLFUnified>(&pic, inferArea, m_inputs, seqQp, sliceQp, -1 /* sliceType */, listInputData);

  NNInference::infer<TypeSadlLFUnified>(model, m_inputs);

  PelUnitBuf bufDst = m_scaled[0][prmId].getBuf(inferArea);

  extractOutputs(pic, model, bufDst, inferArea, extLeft, extRight, extTop, extBottom);
}
#if NN_HOP_UNIFIED_TEMPORAL_FILTERING
void NNFilterUnified::filterBlockTemporal(Picture &pic, UnitArea inferArea, int extLeft, int extRight, int extTop, int extBottom,
                              int prmId)
{
  // get model
  auto &model    = *m_modelTemporal;
  bool  inter    = pic.slices[0]->getSliceType() != I_SLICE ? true : false;
  int   qpOffset = (inter ? prmId * 5 : prmId * 2) * (pic.slices[0]->getTLayer() >= 4 ? 1 : -1);
  int   seqQp    = pic.slices[0]->getPPS()->getPicInitQPMinus26() + 26 + qpOffset;
  resizeInputsTemporal(inferArea.Y().width, inferArea.Y().height);

  const int    log2InputBitdepth = pic.cs->slice->clpRng(COMPONENT_Y).bd;   // internal bitdepth
  const double inputScalePred    = (1 << log2InputBitdepth);
  const double inputScaleQp      = (1 << log2InputQpScale);

  std::vector<InputData> listInputData;
  listInputData.push_back({ NN_INPUT_REC, 0, inputScalePred, m_input_quantizer_temporal - log2InputBitdepth, true, false });
  listInputData.push_back({ NN_INPUT_PRED, 1, inputScalePred, m_input_quantizer_temporal - log2InputBitdepth, true, false });
  listInputData.push_back({ NN_INPUT_REF_LIST_0, 2, inputScalePred, m_input_quantizer_temporal - log2InputBitdepth, true, false });
  listInputData.push_back({ NN_INPUT_REF_LIST_1, 3, inputScalePred, m_input_quantizer_temporal - log2InputBitdepth, true, false });
  listInputData.push_back({ NN_INPUT_GLOBAL_QP, 4, inputScaleQp, m_input_quantizer_temporal - log2InputQpScale, true, false });

  NNInference::prepareInputs<TypeSadlLFUnified>(&pic, inferArea, m_inputsTemporal, seqQp, -1 /* localQp */, -1 /* sliceType */, listInputData);

  NNInference::infer<TypeSadlLFUnified>(model, m_inputsTemporal);

  PelUnitBuf bufDst = m_scaled[0][prmId].getBuf(inferArea);

  extractOutputsTemporal(pic, model, bufDst, inferArea, extLeft, extRight, extTop, extBottom);
}
#endif
void NNFilterUnified::filter(Picture &pic, const bool isDec)
{
  const CodingStructure &cs  = *pic.cs;
  const PreCalcValues &  pcv = *cs.pcv;
  int cpt = 0;
  for (int y = 0; y < m_picprm->nb_blocks_height; ++y)
  {
    for (int x = 0; x < m_picprm->nb_blocks_width; ++x, ++cpt)
    {
      int prmId = m_picprm->prmId[cpt];

      if (prmId == -1)
      {
        continue;
      }

      int xPos  = x * m_picprm->block_size;
      int yPos  = y * m_picprm->block_size;
      int width = (xPos + m_picprm->block_size > (int) pcv.lumaWidth) ? (pcv.lumaWidth - xPos) : m_picprm->block_size;
      int height =
        (yPos + m_picprm->block_size > (int) pcv.lumaHeight) ? (pcv.lumaHeight - yPos) : m_picprm->block_size;

#if JVET_AF0043_AF0205_PADDING
      int extLeft = m_picprm->extension;
      int extRight = m_picprm->extension;
      int extTop = m_picprm->extension;
      int extBottom = m_picprm->extension;
#else
      int extLeft   = xPos > 0 ? m_picprm->extension : 0;
      int extRight  = (xPos + width + m_picprm->extension > (int) pcv.lumaWidth) ? (pcv.lumaWidth - xPos - width)
                                                                                 : m_picprm->extension;
      int extTop    = yPos > 0 ? m_picprm->extension : 0;
      int extBottom = (yPos + height + m_picprm->extension > (int) pcv.lumaHeight) ? (pcv.lumaHeight - yPos - height)
                                                                                   : m_picprm->extension;
#endif

      int            extXPos   = xPos - extLeft;
      int            extYPos   = yPos - extTop;
      int            extWidth  = width + extLeft + extRight;
      int            extHeight = height + extTop + extBottom;
      const UnitArea inferArea(cs.area.chromaFormat, Area(extXPos, extYPos, extWidth, extHeight));
#if NN_HOP_UNIFIED_TEMPORAL_FILTERING
      if (m_picprm->temporal)
        filterBlockTemporal(pic, inferArea, extLeft, extRight, extTop, extBottom, prmId);
      else
#endif
        filterBlock(pic, inferArea, extLeft, extRight, extTop, extBottom, prmId);

      const UnitArea inferAreaNoExt(cs.area.chromaFormat, Area(xPos, yPos, width, height));
      PelUnitBuf     filteredBuf = getFilteredBuf(prmId, inferAreaNoExt);
      PelUnitBuf     scaledBuf   = getScaledBuf(0, prmId, inferAreaNoExt);
      PelUnitBuf     recBuf      = pic.getRecoBuf(inferAreaNoExt);

      roundToOutputBitdepth(scaledBuf, filteredBuf, cs.slice->clpRngs());

      if (!isDec)
      {
        continue;
      }
      if (m_picprm->sprm.scaleFlag != -1)
      {
        int scaleIdx = m_picprm->sprm.scaleFlag;
        for (int compIdx = 0; compIdx < MAX_NUM_COMPONENT; compIdx++)
        {
          ComponentID compID = ComponentID(compIdx);
          scaleIdx == 0 ? scaleResidualBlock(pic, compID, inferAreaNoExt, scaledBuf.get(compID), recBuf.get(compID),
                                             m_picprm->sprm.scale[compID][prmId]
#if JVET_AF0085_RESIDUAL_ADJ
            , m_picprm->sprm.offset[compID][prmId][scaleIdx]
#endif
          )
                        : scaleResidualBlock(pic, compID, inferAreaNoExt, scaledBuf.get(compID), recBuf.get(compID),
                                             scale_candidates[scaleIdx]
#if JVET_AF0085_RESIDUAL_ADJ
                          , m_picprm->sprm.offset[compID][prmId][scaleIdx]
#endif
                        );
        }
      }
      else
      {
#if JVET_AF0085_RESIDUAL_ADJ
        for (int compIdx = 0; compIdx < MAX_NUM_COMPONENT; compIdx++)
        {
          ComponentID compID = ComponentID(compIdx);
          scaleResidualBlock(pic, compID, inferAreaNoExt, scaledBuf.get(compID), filteredBuf.get(compID), 0, m_picprm->sprm.offset[compID][prmId][3]);
        }
#endif
        recBuf.copyFrom(filteredBuf);
      }
    }
  }
}

void NNFilterUnified::scaleResidualBlock(Picture &pic, ComponentID compID, UnitArea inferAreaNoExt, CPelBuf src, PelBuf tgt,
                                     int scale
#if JVET_AF0085_RESIDUAL_ADJ
  , int roa_offset
#endif
) const
{
  const CodingStructure &cs            = *pic.cs;
  const Slice &          slice         = *cs.slice;
  const int              inputBitdepth = slice.clpRng(COMPONENT_Y).bd;   // internal bitdepth
  const int              shift         = log2OutputScale - inputBitdepth;
  const int              shift2        = shift + log2ResidueScale;
  const int              offset        = (1 << shift2) / 2;
  CPelBuf                rec           = pic.getRecoBuf(inferAreaNoExt).get(compID);
  int                    width         = inferAreaNoExt.lwidth();
  int                    height        = inferAreaNoExt.lheight();
#if JVET_AF0085_RESIDUAL_ADJ
  CPelBuf                recBeforeDbf = pic.getRecBeforeDbfBuf(inferAreaNoExt).get(compID);
#endif

  if (compID)
  {
    width  = width / 2;
    height = height / 2;
  }

  for (int y = 0; y < height; ++y)
  {
    for (int x = 0; x < width; ++x)
    {
#if JVET_AF0085_RESIDUAL_ADJ
      if (scale > 0)
      {
        // positive-, negative+
        int v = 0;
        if ((src.at(x, y) - (recBeforeDbf.at(x, y) << shift)) >= (roa_offset << shift))
          v = (((int)rec.at(x, y) << shift2) + (src.at(x, y) - (rec.at(x, y) << shift) - (roa_offset << shift)) * scale + offset) >> shift2;
        else if ((src.at(x, y) - (recBeforeDbf.at(x, y) << shift)) <= (-roa_offset << shift))
          v = (((int)rec.at(x, y) << shift2) + (src.at(x, y) - (rec.at(x, y) << shift) + (roa_offset << shift)) * scale + offset) >> shift2;
        else
          v = (((int)rec.at(x, y) << shift2) + (src.at(x, y) - (rec.at(x, y) << shift)) * scale + offset) >> shift2;
        tgt.at(x, y) = Pel(Clip3<int>(0, (1 << inputBitdepth) - 1, v));
      }
      else
      {
        if ((src.at(x, y) - (recBeforeDbf.at(x, y) << shift)) >= (roa_offset << shift))
          tgt.at(x, y) = Pel(Clip3<int>(0, (1 << inputBitdepth) - 1, tgt.at(x, y) - roa_offset));
        else if ((src.at(x, y) - (recBeforeDbf.at(x, y) << shift)) <= (-roa_offset << shift))
          tgt.at(x, y) = Pel(Clip3<int>(0, (1 << inputBitdepth) - 1, tgt.at(x, y) + roa_offset));
        else
          tgt.at(x, y) = Pel(Clip3<int>(0, (1 << inputBitdepth) - 1, tgt.at(x, y)));
      }
#else
      int v = (((int) rec.at(x, y) << shift2) + (src.at(x, y) - (rec.at(x, y) << shift)) * scale + offset) >> shift2;
      tgt.at(x, y) = Pel(Clip3<int>(0, (1 << inputBitdepth) - 1, v));
#endif
    }
  }
}
