#include <chrono>
#include <iostream>
#include <string>

#include "cnn.h"

using std::chrono::duration_cast;
using std::chrono::microseconds;
using std::chrono::steady_clock;
using std::clog;
using std::endl;
using std::string;

int main(int argc, char** argv) {
  // Allocate memory on heap to avoid stack overflow.
  static float input[kNum][kInImSize][kInImSize];
  static float weight[kNum][kNum][kKernel][kKernel];
  static float bias[kNum];
  static float output[kNum][kOutImSize][kOutImSize];

  if (argc > 2) {
    clog << "Usage: " << argv[0] << " [data dir]\n";
    return EXIT_FAILURE;
  }

  const string data_dir = argc == 2 ? string(argv[1]) + "/" : "";
  LoadData(data_dir, input, weight, bias);
  clog << "Invoke CNN computation kernel\n";

  if (getenv("SEQUENTIAL")) {
    const auto begin = steady_clock::now();
    CnnSequential(input, weight, bias, output);
    const auto end = steady_clock::now();

    uint64_t run_time_us = duration_cast<microseconds>(end - begin).count();
    float gflops = float(kNum) * kNum * kImSize * kImSize * kKernel * kKernel * 2
                   / (run_time_us * 1e3);
    clog << "Time: " << run_time_us * 1e-6 << " s\n";
    clog << "Perf: " << gflops << " GFlops\n";
  } else {
    CnnKernel(input, weight, bias, output);
  }

  int error = Verify(data_dir, output);
  if (error != 0) {
    clog << "Found " << error << " error" << (error > 1 ? "s\n" : "\n");
    clog << "FAIL" << endl;
    return EXIT_FAILURE;
  } else {
    clog << "PASS" << endl;
    return EXIT_SUCCESS;
  }
}
