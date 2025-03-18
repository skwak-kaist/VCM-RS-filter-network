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
#include <cstdlib>

using namespace std;
constexpr int canary=42;
namespace
{
constexpr int verbose = 3;
template<typename T> void print(const sadl::Tensor<T> &t,int c) {
  cout<<"Channel "<<c<<endl;
  const int H=t.dims()[1];
  const int W=t.dims()[2];
  for(int i=0;i<H;++i) {
   for(int j=0;j<W;++j) 
    cout<<t(0,i,j,c)<<'\t';
   cout<<'\n';
  }

   
}

template<typename T> bool checkResults(const sadl::Tensor<T> &gt, const sadl::Tensor<T> &test, int border_sizex, int border_sizey)
{
  if (gt.dims().size()!=4) {
    cerr<<"[ERROR] not a dim=4 tensor "<<gt.dims()<<endl;
    return false;
  }
  const int C=gt.dims()[3];
  const int H=gt.dims()[1];
  const int W=gt.dims()[2];
  for(int c=0;c<C;c++) {
    // check border is not touched
    for(int i=0;i<H;i++) {
      for(int j=0;j<border_sizex;++j) {
       if (test(0,i,j,c)!=canary||test(0,i,W-1-j,c)!=canary) {
          if (verbose) cerr<<"[ERROR] border v changed"<<endl;
          if (verbose>1) {
             std::cout<<"GT\n";
             print(gt,c);
             std::cout<<"\nTEST\n";
             print(test,c);
             std::cout<<std::endl;
          }
          return false;
       }
      }
    }
    for(int j=0;j<W;++j) {
     for(int i=0;i<border_sizey;++i) {
      if (test(0,i,j,c)!=canary||test(0,H-1-i,j,c)!=canary) {
       if (verbose) cerr<<"[ERROR] border h changed"<<endl;
       if (verbose>1) {
             std::cout<<"GT\n";
             print(gt,c);
             std::cout<<"\nTEST\n";
             print(test,c);
             std::cout<<std::endl;
        }
       return false;
      }
     }
    }   
   for(int i=border_sizey;i<H-border_sizey;i++) {
     for(int j=border_sizex;j<W-border_sizex;++j) {
       if (gt(0,i,j,c)!=test(0,i,j,c)) {
          if (verbose) cerr<<"[ERROR] core changed "<<i<<' '<<j<<' '<<c<<endl;
          if (verbose>1) {
             std::cout<<"GT\n";
             print(gt,c);
             std::cout<<"\nTEST\n";
             print(test,c);
             std::cout<<std::endl;
          }
          return false;
       }
     }
   } 
  } 
  return true;
}

template<typename T> sadl::Tensor<T> infer(const std::string &filename, bool no_border)
{
  sadl::Model<T> model;
  std::ifstream  file(filename, ios::binary);
  sadl::Tensor<T>::skip_border=no_border;
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

  if (!model.init(inputs))
  {
    cerr << "[ERROR] Pb init" << endl;
    exit(-1);
  }
  auto L=model.getIdsOutput();
  auto &out=*model.getLayer(L[0]).layer;
  out.output().fill(canary); 

  srand(42);
  for(auto &v: inputs)
   for(auto &x: v)
      x = (float)(rand() % 1024);

  if (!model.apply(inputs))
  {
    cerr << "[ERROR] Pb apply" << endl;
    exit(-1);
  }
  if (model.nbOutputs()>1) {
    cerr << "[ERROR] nb outputs " << model.nbOutputs()<<endl;
    exit(-1);
  }
  return model.result();
}

}   // namespace

int main(int argc, char **argv)
{
  if (argc != 4 && argc!= 5)
  {
    cerr << "test model.sadl border_x border_y [int]" << endl;
    return -1;
  }
  string filename_model   = argv[1];
  int bx=atoi(argv[2]);
  int by=atoi(argv[3]);
  if (argc==4) { 
   using T=float;
   auto gt=infer<T>(filename_model, false);
   auto test=infer<T>(filename_model, bx!=0||by!=0);
   checkResults(gt,test,bx,by);  
  }

    
}
