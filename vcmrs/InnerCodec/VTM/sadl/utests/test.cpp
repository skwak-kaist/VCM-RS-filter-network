/* The copyright in this software is being made available under the BSD
 * License, included below. This software may be subject to other third party
 * and contributor rights, including patent rights, and no such rights are
 * granted under this license.
 *
 * Copyright (c) 2010-2024, ITU/ISO/IEC
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
#include <cmath>
#include <fstream>
#include <iomanip>
#include <sadl/model.h>

using namespace std;

namespace
{
int verbose = 2;
// results file
// nb inputs [int]
// for all input:
//   input dim size [int] # between 1 and 4
//   for all dim in input dim size
//      dim[k] [int]
//   input [float[nb elts]]
// nb output [int] # currently == 1
// for all output:
//   output dim size [int] # between 1 and  4
//   for all dim in output dim size
//      dim[k] [int]
//   output [float[nb_elts]]
template<typename T> bool readResults(const string &filename, std::vector<sadl::Tensor<T>> &inputs, std::vector<std::vector<float>> &outputs)
{
  ifstream file(filename, ios::binary);
  int      nb;
  file >> nb;
  if (nb != (int) inputs.size())
  {
    cerr << "[ERROR] invalid nb tensors" << endl;
    return false;
  }
  for (auto &t: inputs)
  {
    sadl::Dimensions d;
    file >> nb;
    if (nb < 1 || nb > 4)
    {
      cerr << "[ERROR] invalid dim in" << endl;
      return false;
    }
    d.resize(nb);
    for (auto &x: d)
      file >> x;

    if (!(d == t.dims()))
    {
      cerr << "[ERROR] invalid dimension tensor" << d << " " << t.dims() << endl;
      return false;
    }

    for (auto &x: t)
    {
      float z;
      file >> z;
      x = (T) z;
    }
  }

  int nb_outputs;
  file >> nb_outputs;

  if (nb_outputs < 1)
  {
    cerr << "[ERROR] invalid nb output " << nb << endl;
    return false;
  }

  outputs.resize(nb_outputs);

  for (int i = 0; i < nb_outputs; ++i)
  {
    file >> nb;
    if (nb < 1 || nb > 4)
    {
      cerr << "[ERROR] invalid dim out" << endl;
      return false;
    }
    sadl::Dimensions dims_out;
    dims_out.resize(nb);
    for (auto &x: dims_out)
      file >> x;
    outputs[i].resize(dims_out.nbElements());
    for (auto &x: outputs[i])
      file >> x;
  }

  return !file.fail();
}

template<typename T> bool checkResults(const std::vector<float> &gt, const sadl::Tensor<T> &test, double abs_tol)
{
  double max_a       = 0.;
  int    nb_e        = 0;
  double max_fabs    = 0.;
  auto   check_value = [&](auto x_test, auto x_gt)
  {
    float  x  = (float) x_test;
    double a  = fabs(x - x_gt);
    double fb = fabs(x_gt);
    max_fabs  = max(max_fabs, fb);
    max_a = max(a, max_a);
    if (a > abs_tol)
    {
      ++nb_e;
    }
  };

  int nb_tested = 0;
  for (int cpt = 0; cpt < (int) gt.size(); ++cpt, ++nb_tested)
    check_value(test[cpt], gt[cpt]);

  if (verbose > 1)
  {
    if (nb_e > 0)
    {
      cout << "[ERROR] test FAILED " << nb_e << "/" << nb_tested << " ";
    }
    else
    {
      cout << "[INFO] test OK ";
    }
  }
  return nb_e == 0;
}

template<typename T> int infer(const std::string &filename_results, const std::string &filename, double max_e)
{
  sadl::Model<T> model;
  std::ifstream  file(filename, ios::binary);

  if (!model.load(file))
  {
    cerr << "[ERROR] Unable to read model " << filename << endl;
    exit(-1);
  }

  std::vector<sadl::Tensor<T>> inputs = model.getInputsTemplate();

  if (inputs.size() == 0)
  {
    cerr << "[ERROR] missing inputs information (model or prm string)" << endl;
    exit(-1);
  }
  std::vector<std::vector<float>> outputs;   // not a tensor to be generic with integer network

  std::vector<sadl::Dimensions> dim_outs;
  if (!readResults<T>(filename_results, inputs, outputs))
  {
    cerr << "[ERROR] reading result file " << filename_results << endl;
    exit(-1);
  }

  if (!model.init(inputs))
  {
    cerr << "[ERROR] Pb init" << endl;
    exit(-1);
  }

  if (verbose > 1)
  {
    for (const auto &t: inputs)
      cout << "Input " << t << endl;
  }
  if (!model.apply(inputs))
  {
    cerr << "[ERROR] Pb apply" << endl;
    exit(-1);
  }

  for (int i = 0; i < (int) dim_outs.size(); ++i)
  {
    if (dim_outs[i].size() + 1 == model.result(i).dims().size())
    {
      // add batch
      auto d = dim_outs[i];
      dim_outs[i].resize(d.size() + 1);
      dim_outs[i][0] = 1;
      for (int k = 0; k < d.size(); ++k)
        dim_outs[i][k + 1] = d[k];
    }
    if (!(dim_outs[i] == model.result(i).dims()))
    {
      cerr << "[ERROR] output size different: file=" << dim_outs[i] << " model=" << model.result(i).dims() << endl;
      exit(-1);
    }
  }

  for (int i = 0; i < (int) outputs.size(); ++i)
  {
    if (verbose > 1)
    {
      auto t = model.result(i);
      for (int k = 0; k < t.size(); ++k)
        t[k] = outputs[i][k];
      cout << "[INFO] output file " << i << "\n" << t << endl;
    }
    if (verbose > 1)
      cout << "[INFO] output model " << i << "\n" << model.result(i) << endl;

    if (!checkResults<T>(outputs[i], model.result(i), max_e))
    {
      cerr << "[ERROR] difference onnx/sadl" << endl;
      return -1;
    }
  }
  cout << "[INFO] test passed" << endl;
  return 0;
}

}   // namespace

int main(int argc, char **argv)
{
  if (argc != 4)
  {
    cerr << "test model.sadl file.results max_error" << endl;
    return -1;
  }
  string filename_model   = argv[1];
  string filename_results = argv[2];
  double e_max            = atof(argv[3]);
  return infer<float>(filename_results, filename_model, e_max);
}
