/******************************************************************************
 *                                                                            *
 *  In this file the OpenCL C++ APIs are used instead of the C APIs taught    *
 *  in class. Please refer to                                                 *
 *  https://github.khronos.org/OpenCL-CLHPP/namespacecl.html                  *
 *  or                                                                        *
 *  https://www.khronos.org/registry/OpenCL/specs/opencl-cplusplus-1.2.pdf    *
 *  for more information about how the C++ APIs wrap the C APIs.              *
 *                                                                            *
 ******************************************************************************/

#include <cstdlib>

#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "cnn.h"

using std::chrono::duration_cast;
using std::chrono::microseconds;
using std::chrono::steady_clock;
using std::clog;
using std::endl;
using std::istream_iterator;
using std::istreambuf_iterator;
using std::ostream;
using std::runtime_error;
using std::stoul;
using std::string;
using std::vector;

cl::Program::Binaries LoadBinaryFile(const string& file_name) {
  clog << "Loading binary file: " << file_name << endl;
  std::ifstream stream(file_name, std::ios::binary);
  vector<unsigned char> contents(istreambuf_iterator<char>{stream},
                                 istreambuf_iterator<char>());
  return {contents};
}

cl::Program::Sources LoadSourceFile(const string& file_name) {
  clog << "Loading source file: " << file_name << endl;
  std::ifstream stream(file_name);
  string contents(istreambuf_iterator<char>{stream},
                  istreambuf_iterator<char>());
  return {contents};
}

ostream& operator<<(ostream& stream, const cl::NDRange& range) {
  if (range.dimensions() == 0) {
    return stream << "<null>";
  }
  stream << "(";
  for (int i = 0; i < range.dimensions(); ++i) {
    if (i > 0) {
      stream << ", ";
    }
    stream << range.get()[i];
  }
  stream << ")";
  return stream;
}

void OpenclSetup(const string& kernel_name, cl::Context* context,
                 cl::CommandQueue* cmd, cl::Kernel* kernel) {
  // input params
  string binary;
  string source;
  if (auto var = getenv("OPENCL_BINARY")) {
    binary = var;
  } else {
    if ((var = getenv("OPENCL_SOURCE"))) {
      source = var;
    } else {
      throw runtime_error(
          "neither OPENCL_BINARY nor OPENCL_SOURCE not set");
    }
  }
  string target_device_name;
  if (auto var = getenv("OPENCL_DEVICE")) {
    target_device_name = var;
  } else {
    throw runtime_error("OPENCL_DEVICE not set");
  }
  string target_platform_name;
  if (auto var = getenv("OPENCL_PLATFORM")) {
    target_platform_name = var;
  } else {
    throw runtime_error("OPENCL_PLATFORM not set");
  }

  cl::Program::Binaries binaries;
  cl::Program::Sources sources;
  if (!binary.empty()) {
    binaries = LoadBinaryFile(binary);
  } else {
    sources = LoadSourceFile(source);
  }

  cl::Program program;

  vector<cl::Platform> platforms;
  CL_CHECK(cl::Platform::get(&platforms));
  cl_int err;

  class ConfigDone {};
  try {
    for (const auto& platform : platforms) {
      string platform_name = platform.getInfo<CL_PLATFORM_NAME>(&err);
      CL_CHECK(err);
      clog << "Found platform: " << platform_name.c_str() << endl;
      if (platform_name == target_platform_name) {
        vector<cl::Device> devices;
        CL_CHECK(platform.getDevices(CL_DEVICE_TYPE_ALL, &devices));
        for (const auto& device : devices) {
          const string device_name = device.getInfo<CL_DEVICE_NAME>(&err);
          CL_CHECK(err);
          clog << "Found device: " << device_name << endl;
          if (device_name == target_device_name) {
            clog << "Using device: " << device_name << endl;
            clog << "Global memory cache size: "
                 << device.getInfo<CL_DEVICE_GLOBAL_MEM_CACHE_SIZE>()
                 << " bytes" << endl;
            clog << "Global memory cache line size: "
                 << device.getInfo<CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE>()
                 << " bytes" << endl;
            clog << "Global device memory size: "
                 << device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>()
                 << " bytes" << endl;
            clog << "Local memory arena size: "
                 << device.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>()
                 << " bytes" << endl;
            clog << "Number of parallel compute cores: "
                 << device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << endl;
            clog << "Constant buffer size: "
                 << device.getInfo<CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE>()
                 << " bytes" << endl;
            clog << "Maximum number of work-items in the work-group: (";
            bool first = true;
            for (auto size : device.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>()) {
              if (first) {
                first = false;
              } else {
                clog << ", ";
              }
              clog << size;
            }
            clog << ")" << endl;

            *context = cl::Context(device, nullptr, nullptr, nullptr, &err);
            if (err == CL_DEVICE_NOT_AVAILABLE) {
              continue;
            }
            CL_CHECK(err);
            *cmd = cl::CommandQueue(
                *context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE |
                CL_QUEUE_PROFILING_ENABLE, &err);
            CL_CHECK(err);
            if (!binary.empty()) {
              vector<int> binary_status;
              program = cl::Program(
                  *context, {device}, binaries, &binary_status, &err);
              CL_CHECK(err);
              for (auto status : binary_status) {
                CL_CHECK(status);
              }
            } else {
              program = cl::Program(*context, sources, &err);
              CL_CHECK(err);
            }
            err = program.build({device});
            if (err != CL_SUCCESS) {
              throw runtime_error(
                  program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device));
            }
            *kernel = cl::Kernel(program, kernel_name.c_str(), &err);
            CL_CHECK(err);
            throw ConfigDone();
          }
        }
        throw runtime_error("device " + target_device_name + " not found");
      }
    }
    throw runtime_error("platform " + target_platform_name + " not found");
  } catch (ConfigDone) {
    clog << "OpenCL configuration finished" << endl;
  }
}

void CnnKernel(
    const float input[kNum][kInImSize][kInImSize],
    const float weight[kNum][kNum][kKernel][kKernel],
    const float bias[kNum], float output[kNum][kOutImSize][kOutImSize]) {
  cl::Context context;
  cl::CommandQueue cmd;
  cl::Kernel kernel;
  cl_int err;
  OpenclSetup("CnnKernel", &context, &cmd, &kernel);

  // Set up workgroups.
  cl::NDRange offset = cl::NullRange;
  cl::NDRange global(1);
  cl::NDRange local = cl::NullRange;

  std::map<string, cl::NDRange*> workgroup_settings;
  workgroup_settings["OPENCL_WORKGROUP_OFFSET"] = &offset;
  workgroup_settings["OPENCL_WORKGROUP_GLOBAL"] = &global;
  workgroup_settings["OPENCL_WORKGROUP_LOCAL"] = &local;

  for (const auto& setting : workgroup_settings) {
    const auto& name = setting.first;
    const auto& range = setting.second;
    if (auto var = getenv(name.c_str())) {
      std::istringstream iss{string(var)};
      vector<string> nums(istream_iterator<string>{iss},
                          istream_iterator<string>());
      switch (nums.size()) {
        case 1:
          *range = cl::NDRange(stoul(nums[0]));
          break;
        case 2:
          *range = cl::NDRange(stoul(nums[0]), stoul(nums[1]));
          break;
        case 3:
          *range = cl::NDRange(stoul(nums[0]), stoul(nums[1]), stoul(nums[2]));
          break;
        default:
          throw runtime_error("invalid workgroup setting: " + string(var));
      }
    }
  }
  clog << "Using global workgroup: " << global << endl;
  clog << "Using global workgroup offset: " << offset << endl;
  clog << "Using local workgroup: " << local << endl;

  // Map userspace memory to kernelspace.
  vector<cl::Memory> cl_buf_in;
  vector<cl::Memory> cl_buf_out;
  cl_buf_in.push_back(cl::Buffer(
      context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(*input) * kNum,
      const_cast<float*>(reinterpret_cast<const float*>(input)), &err));
  CL_CHECK(err);
  cl_buf_in.push_back(cl::Buffer(
      context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(*weight) * kNum,
      const_cast<float*>(reinterpret_cast<const float*>(weight)), &err));
  CL_CHECK(err);
  cl_buf_in.push_back(cl::Buffer(
      context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(*bias) * kNum,
      const_cast<float*>(reinterpret_cast<const float*>(bias)), &err));
  CL_CHECK(err);
  cl_buf_out.push_back(cl::Buffer(
      context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY,
      sizeof(*output) * kNum, output, &err));
  CL_CHECK(err);

  // Set up kernel arguments.
  int arg = 0;
  for (const auto& buf : cl_buf_in) {
    CL_CHECK(kernel.setArg(arg++, buf));
  }
  for (const auto& buf : cl_buf_out) {
    CL_CHECK(kernel.setArg(arg++, buf));
  }

  // Execute kernel.
  const auto begin = steady_clock::now();
  vector<cl::Event> compute_event(1);
  vector<cl::Event> write_event(1);
  CL_CHECK(cmd.enqueueMigrateMemObjects(
      cl_buf_in, /* flags = */ 0, nullptr, write_event.data()));
  CL_CHECK(cmd.enqueueNDRangeKernel(
      kernel, offset, global, local, &write_event, compute_event.data()));
  CL_CHECK(cmd.enqueueMigrateMemObjects(
      cl_buf_out, CL_MIGRATE_MEM_OBJECT_HOST, &compute_event, nullptr));
  CL_CHECK(cmd.flush());
  CL_CHECK(cmd.finish());
  const auto end = steady_clock::now();

  uint64_t run_time_us = duration_cast<microseconds>(end - begin).count();
  float gflops = float(kNum) * kNum * kImSize * kImSize * kKernel * kKernel * 2
                 / (run_time_us * 1e3);
  clog << "Time: " << run_time_us * 1e-6 << " s\n";
  clog << "Perf: " << gflops << " GFlops\n";
}

void VaddKernel(const float* a, const float* b, float* c, uint64_t n) {
  cl::Context context;
  cl::CommandQueue cmd;
  cl::Kernel kernel;
  cl_int err;
  OpenclSetup("VaddKernel", &context, &cmd, &kernel);

  // Map userspace memory to kernelspace.
  vector<cl::Memory> cl_buf_in;
  vector<cl::Memory> cl_buf_out;
  cl_buf_in.push_back(cl::Buffer(
      context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(*a) * n,
      const_cast<float*>(a), &err));
  CL_CHECK(err);
  cl_buf_in.push_back(cl::Buffer(
      context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(*b) * n,
      const_cast<float*>(b), &err));
  CL_CHECK(err);
  cl_buf_out.push_back(cl::Buffer(
      context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY,
      sizeof(*c) * n, c, &err));
  CL_CHECK(err);

  // Set up kernel arguments.
  int arg = 0;
  for (const auto& buf : cl_buf_in) {
    CL_CHECK(kernel.setArg(arg++, buf));
  }
  for (const auto& buf : cl_buf_out) {
    CL_CHECK(kernel.setArg(arg++, buf));
  }

  // Execute kernel.
  vector<cl::Event> compute_event(1);
  vector<cl::Event> write_event(1);
  CL_CHECK(cmd.enqueueMigrateMemObjects(
      cl_buf_in, /* flags = */ 0, nullptr, write_event.data()));
  CL_CHECK(cmd.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(n),
           cl::NullRange, &write_event, compute_event.data()));
  CL_CHECK(cmd.enqueueMigrateMemObjects(
      cl_buf_out, CL_MIGRATE_MEM_OBJECT_HOST, &compute_event, nullptr));
  CL_CHECK(cmd.flush());
  CL_CHECK(cmd.finish());
};
