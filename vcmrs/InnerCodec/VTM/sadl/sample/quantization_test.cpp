#include <cmath>
#include <fstream>
#include <random>
#include <iomanip>
#include "sadl/model.h"

namespace
{

template<typename T> bool init_inputs(std::vector<sadl::Tensor<T>> &inputs, double inputs_range)
{
  unsigned                        seed = 5;
  std::mt19937                    gen(seed);
  std::uniform_int_distribution<> distribution((int) (inputs_range / 4.0), (int) inputs_range);
  for (auto &input: inputs)
  {
    for (auto &x: input)
    {
      double x0 = distribution(gen);
      // normalize inputs for float model
      x = (T)((std::is_same<T, float>::value) ? round(x0) / inputs_range : round(x0));
    }
  }
  return true;
}

template<typename T> std::vector<sadl::Tensor<T>> infer(const std::string &filename_model, double inputs_range)
{
  sadl::Model<T> model;
  std::ifstream  file(filename_model, std::ios::binary);
  if (!model.load(file))
  {
    std::cerr << "[ERROR] Unable to read model " << filename_model << std::endl;
    exit(-1);
  }
  std::vector<sadl::Tensor<T>> inputs = model.getInputsTemplate();
  if (!model.init(inputs))
  {
    std::cerr << "[ERROR] Pb init" << std::endl;
    exit(-1);
  }
  if (!init_inputs<T>(inputs, inputs_range))
  {
    std::cerr << "[ERROR] Pb init inputs" << std::endl;
    exit(-1);
  }
  if (!model.apply(inputs))
  {
    std::cerr << "[ERROR] Pb apply" << std::endl;
    exit(-1);
  }
  std::vector<sadl::Tensor<T>> outputs;
  const int                    N = (int) model.nbOutputs();
  for (int i = 0; i < N; i++)
  {
    outputs.push_back(model.result(i));
  }
  return outputs;
}

int compare(const std::string &filename_model1, const std::string &filename_model2, double inputs_range, int shift, int max_e)
{
  int  nb_e      = 0;
  int  nb_tested = 0;
  auto results1  = infer<float>(filename_model1, inputs_range);
  auto results2  = infer<int16_t>(filename_model2, inputs_range);
  for (int i = 0; i < (int) results1.size(); i++)
  {
    for (int j = 0; j < (int) results1[i].size(); j++)
    {
      nb_tested++;
      int x1 = (int) (results1[i][j] * inputs_range);
      int x2 = results2[i][j];
      if (shift > 0)
        x2 <<= shift;
      else if (shift < 0)
        x2 >>= (-shift);

      int difference = abs(x1 - x2);
      if (difference > max_e)
      {
        nb_e++;
        printf("Denormalized float value: %+10d || Int16 value:  %+10d || Difference: %+10d.\n", x1, x2, difference);
      }
    }
  }
  if (nb_e > 0)
  {
    std::cout << "[INFO] QUANTIZATION CONSISTENCY TEST FAILED: " << nb_e << "/" << nb_tested << "." << std::endl;
    return -1;
  }
  std::cout << "[INFO] QUANTIZATION CONSISTENCY TEST PASSED." << std::endl;
  return 0;
}
}   // namespace

int main(int argc, char **argv)
{
  // Usage: quantization_test model_float.sadl model_int16.sadl inputs_range shift max_error
  if (argc != 6)
  {
    std::cerr << "quantization_test model_float.sadl model_int16.sadl inputs_range shift max_error" << std::endl;
    return -1;
  }
  std::string filename_model1 = argv[1];         // Path to the float SADL model.
  std::string filename_model2 = argv[2];         // Path to the int16 SADL model.
  double      inputs_range    = atof(argv[3]);   // Should be a power of 2 (2^N), where N is the quantizer of Placeholder.
  int         shift           = (int) atof(argv[4]);   // Manually configured shift applied to the final output of the int16 model,
                                                 // based on different quantizers in int16 SADL model.
  int max_e = (int) atof(argv[5]);                     // The maximum absolute error between the dequantized float model and the int16
                                                 // model in the results.

  return compare(filename_model1, filename_model2, inputs_range, shift, max_e);
}
