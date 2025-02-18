#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>

#include <iostream>
#include <string>
#include <vector>

#define CHECK_RET_CODE(call, ret_code)                                                             \
  {                                                                                                \
    if ((call) != ret_code) {                                                                      \
      std::cout << "Failed in call: " << #call << std::endl;                                       \
      std::abort();                                                                                \
    }                                                                                              \
  }
#define HIP_CHECK(call) CHECK_RET_CODE(call, hipSuccess)
#define HIPRTC_CHECK(call) CHECK_RET_CODE(call, HIPRTC_SUCCESS)

// source code for hiprtc
static constexpr auto kernel_source{
    R"(
    extern "C"
    __global__ void vector_add(float* output, float* input1, float* input2, size_t size) {
      int i = threadIdx.x;
      if (i < size) {
        output[i] = input1[i] + input2[i];
      }
    }
)"};

int main() {
  hiprtcProgram prog;
  auto rtc_ret_code = hiprtcCreateProgram(&prog,            // HIPRTC program handle
                                          kernel_source,    // kernel source string
                                          "vector_add.cpp", // Name of the file
                                          0,                // Number of headers
                                          NULL,             // Header sources
                                          NULL);            // Name of header file

  if (rtc_ret_code != HIPRTC_SUCCESS) {
    std::cout << "Failed to create program" << std::endl;
    std::abort();
  }

  hipDeviceProp_t props;
  int device = 0;
  HIP_CHECK(hipGetDeviceProperties(&props, device));
  std::string sarg = std::string("--gpu-architecture=") +
      props.gcnArchName;  // device for which binary is to be generated

  const char* options[] = {sarg.c_str()};

  rtc_ret_code = hiprtcCompileProgram(prog,      // hiprtcProgram
                                      0,         // Number of options
                                      options);  // Clang Options
  if (rtc_ret_code != HIPRTC_SUCCESS) {
    std::cout << "Failed to create program" << std::endl;
    std::abort();
  }

  size_t logSize;
  HIPRTC_CHECK(hiprtcGetProgramLogSize(prog, &logSize));

  if (logSize) {
    std::string log(logSize, '\0');
    HIPRTC_CHECK(hiprtcGetProgramLog(prog, &log[0]));
    std::cout << "Compilation failed with: " << log << std::endl;
    std::abort();
  }

  size_t codeSize;
  HIPRTC_CHECK(hiprtcGetCodeSize(prog, &codeSize));

  std::vector<char> kernel_binary(codeSize);
  HIPRTC_CHECK(hiprtcGetCode(prog, kernel_binary.data()));

  HIPRTC_CHECK(hiprtcDestroyProgram(&prog));

  hipModule_t module;
  hipFunction_t kernel;

  HIP_CHECK(hipModuleLoadData(&module, kernel_binary.data()));
  HIP_CHECK(hipModuleGetFunction(&kernel, module, "vector_add"));

  constexpr size_t ele_size = 256;  // total number of items to add
  std::vector<float> hinput, output;
  hinput.reserve(ele_size);
  output.reserve(ele_size);
  for (size_t i = 0; i < ele_size; i++) {
    hinput.push_back(static_cast<float>(i + 1));
    output.push_back(0.0f);
  }

  float *dinput1, *dinput2, *doutput;
  HIP_CHECK(hipMalloc(&dinput1, sizeof(float) * ele_size));
  HIP_CHECK(hipMalloc(&dinput2, sizeof(float) * ele_size));
  HIP_CHECK(hipMalloc(&doutput, sizeof(float) * ele_size));

  HIP_CHECK(hipMemcpy(dinput1, hinput.data(), sizeof(float) * ele_size, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dinput2, hinput.data(), sizeof(float) * ele_size, hipMemcpyHostToDevice));

  struct {
    float* output;
    float* input1;
    float* input2;
    size_t size;
  } args{doutput, dinput1, dinput2, ele_size};

  auto size = sizeof(args);
  void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args, HIP_LAUNCH_PARAM_BUFFER_SIZE, &size,
                    HIP_LAUNCH_PARAM_END};

  HIP_CHECK(hipModuleLaunchKernel(kernel, 1, 1, 1, ele_size, 1, 1, 0, nullptr, nullptr, config));

  HIP_CHECK(hipMemcpy(output.data(), doutput, sizeof(float) * ele_size, hipMemcpyDeviceToHost));

  for (size_t i = 0; i < ele_size; i++) {
    if ((hinput[i] + hinput[i]) != output[i]) {
      std::cout << "Failed in validation: " << (hinput[i] + hinput[i]) << " - " << output[i]
                << std::endl;
      std::abort();
    }
  }
  std::cout << "Passed" << std::endl;

  HIP_CHECK(hipFree(dinput1));
  HIP_CHECK(hipFree(dinput2));
  HIP_CHECK(hipFree(doutput));
}