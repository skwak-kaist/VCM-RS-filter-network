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

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#if defined (__ARM_NEON__) || defined(__ARM_NEON)
#include <stdlib.h>
#else
#include <malloc.h>
#endif
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>
#include <list>

#define DUMP_MODEL_EXT virtual bool dump(std::ostream &file)
// trick to access inner data
#define private public
#define protected public
#include <sadl/model.h>
#undef private
#undef protected
#define private private
#define protected public

#include "helper.h"
#include "dumper.h"

using namespace std;

namespace
{
std::string showMagic(const std::string &in) {
  ifstream           file(in, ios::binary);
  char magic[9];
  file.read(magic, 8);
  magic[8] = '\0';
  return magic; 
}


template<typename T>
void load_save(const std::string &in,const std::string &out) {
  sadl::Model<T> model;
  ifstream           file(in, ios::binary);
  cout << "[INFO] Model loading" << endl;
  if (!model.load(file))
  {
    cerr << "[ERROR] Unable to read model " << in << endl;
    exit(-1);
  }
  std::cout<<"Input version: "<<showMagic(in)<<std::endl;

  // dump to file
  ofstream file_out(out, ios::binary);
  model.dump(file_out);
  cout << "[INFO] new model in " << out << endl;
  std::cout<<"Output version: "<<showMagic(out)<<std::endl;
}

}   // namespace

int main(int argc, char **argv)
{
  if (argc != 3)
  {
    cout << "[ERROR] update_model_version old_model.sadl new_model.sadl" << endl;
    return 1;
  }

  const string filename_model     = argv[1];
  const string filename_model_out = argv[2];

  sadl::layers::TensorInternalType::Type type_model = getModelType(filename_model); 
  switch (type_model)
  {
  case sadl::layers::TensorInternalType::Float:
    load_save<float>(filename_model,filename_model_out);
    break;
  case sadl::layers::TensorInternalType::Int32:
    load_save<int32_t>(filename_model,filename_model_out);
    break;
  case sadl::layers::TensorInternalType::Int16:
    load_save<int16_t>(filename_model,filename_model_out);
    break;
  default:
    cerr << "[ERROR] unsupported type" << endl;
    exit(-1);
  }
  return 0;
}
