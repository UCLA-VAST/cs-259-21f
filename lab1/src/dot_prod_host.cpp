#include "xcl2.hpp"
#include <algorithm>
#include <vector>
#define DATA_SIZE 4096

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cout << "Usage: " << argv[0] << " <XCLBIN File>" << std::endl;
    return EXIT_FAILURE;
  }

  std::string binaryFile = argv[1];
  size_t vector_size_bytes = sizeof(float) * DATA_SIZE;
  cl_int err;
  cl::Context context;
  cl::Kernel krnl_dot_prod;
  cl::CommandQueue q;
  // Allocate Memory in Host Memory
  // When creating a buffer with user pointer (CL_MEM_USE_HOST_PTR), under the
  // hood user ptr
  // is used if it is properly aligned. when not aligned, runtime had no choice
  // but to create
  // its own host side buffer. So it is recommended to use this allocator if
  // user wish to
  // create buffer using CL_MEM_USE_HOST_PTR to align user buffer to page
  // boundary. It will
  // ensure that user buffer is used when user create Buffer/Mem object with
  // CL_MEM_USE_HOST_PTR
  std::vector<float, aligned_allocator<float>> source_in1(DATA_SIZE);
  std::vector<float, aligned_allocator<float>> source_in2(DATA_SIZE);
  std::vector<float, aligned_allocator<float>> source_hw_results(1);
  std::vector<float, aligned_allocator<float>> source_sw_results(1);

  source_hw_results[0] = 0.f;
  source_sw_results[0] = 0.f;

  for (int i = 0; i < DATA_SIZE; ++i) {
    source_in1[i] = float(i) / DATA_SIZE;
    source_in2[i] = float(i * i) / DATA_SIZE / DATA_SIZE;
    source_sw_results[0] += source_in1[i] * source_in2[i];
  }

  // OPENCL HOST CODE AREA START
  // get_xil_devices() is a utility API which will find the xilinx
  // platforms and will return list of devices connected to Xilinx platform
  auto devices = xcl::get_xil_devices();
  // read_binary_file() is a utility API which will load the binaryFile
  // and will return the pointer to file buffer.
  auto fileBuf = xcl::read_binary_file(binaryFile);
  cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};
  bool valid_device = false;
  for (unsigned int i = 0; i < devices.size(); i++) {
    auto device = devices[i];
    // Creating Context and Command Queue for selected Device
    OCL_CHECK(err, context = cl::Context(device, NULL, NULL, NULL, &err));
    OCL_CHECK(err, q = cl::CommandQueue(context, device,
                                        CL_QUEUE_PROFILING_ENABLE, &err));
    std::cout << "Trying to program device[" << i
              << "]: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
    cl::Program program(context, {device}, bins, NULL, &err);
    if (err != CL_SUCCESS) {
      std::cout << "Failed to program device[" << i << "] with xclbin file!\n";
    } else {
      std::cout << "Device[" << i << "]: program successful!\n";
      OCL_CHECK(err, krnl_dot_prod = cl::Kernel(program, "dot_prod_kernel", &err));
      valid_device = true;
      break; // we break because we found a valid device
    }
  }
  if (!valid_device) {
    std::cout << "Failed to program any device found, exit!\n";
    exit(EXIT_FAILURE);
  }

  // Allocate Buffer in Global Memory
  // Buffers are allocated using CL_MEM_USE_HOST_PTR for efficient memory and
  // Device-to-host communication
  OCL_CHECK(err, cl::Buffer buffer_in1(
                     context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                     vector_size_bytes, source_in1.data(), &err));
  OCL_CHECK(err, cl::Buffer buffer_in2(
                     context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                     vector_size_bytes, source_in2.data(), &err));
  OCL_CHECK(err, cl::Buffer buffer_output(
                     context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY,
                     sizeof(float), source_hw_results.data(), &err));

  int size = DATA_SIZE;
  OCL_CHECK(err, err = krnl_dot_prod.setArg(0, buffer_in1));
  OCL_CHECK(err, err = krnl_dot_prod.setArg(1, buffer_in2));
  OCL_CHECK(err, err = krnl_dot_prod.setArg(2, buffer_output));
  OCL_CHECK(err, err = krnl_dot_prod.setArg(3, size));

  // Copy input data to device global memory
  OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_in1, buffer_in2},
                                                  0 /* 0 means from host*/));

  // Launch the Kernel
  // For HLS kernels global and local size is always (1,1,1). So, it is
  // recommended
  // to always use enqueueTask() for invoking HLS kernel
  OCL_CHECK(err, err = q.enqueueTask(krnl_dot_prod));

  // Copy Result from Device Global Memory to Host Local Memory
  OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_output},
                                                  CL_MIGRATE_MEM_OBJECT_HOST));
  q.finish();
  // OPENCL HOST CODE AREA END

  // Compare the results of the Device to the simulation
  bool fail = false;
  if (fabs(source_hw_results[0] - source_sw_results[0]) /
      source_sw_results[0] > 1e-4) {
    std::cout << "Error: Result mismatch" << std::endl;
    std::cout << "CPU result = " << source_sw_results[0]
        << " Device result = " << source_hw_results[0] << std::endl;
    fail = true;
  }

  std::cout << "TEST " << (fail ? "FAILED" : "PASSED") << std::endl;
  return (fail ? EXIT_FAILURE : EXIT_SUCCESS);
}
