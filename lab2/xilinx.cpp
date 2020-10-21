#include "ap_int.h"
//#include <iostream>
//using namespace std;

const int kNum = 256;
const int kKernel = 5;
const int kImSize = 224;
const int kInImSize = 228;
const int kOutImSize = 112;

#define weight(i, j, p, q) \
    weight[(i) * kNum * kKernel * kKernel + (j) * kKernel * kKernel + \
    (p) * kKernel + (q)]
#define input(j, h, w) \
    input[(j) * kInImSize * kInImSize + (h) * kInImSize + (w)]
#define output(i, h, w) \
    output[(i) * kOutImSize * kOutImSize + (h) * kOutImSize + (w)]

#define max(a, b) ((a) > (b) ? (a) : (b))

extern "C"{
void CnnKernel(const float* input, const float* weight,
               const float* bias, float* output) {
#pragma HLS INTERFACE m_axi port=input offset=slave bundle=gmem1 depth=13307904
#pragma HLS INTERFACE m_axi port=weight offset=slave bundle=gmem2 depth=1638400
#pragma HLS INTERFACE m_axi port=bias offset=slave bundle=gmem3 depth=256
#pragma HLS INTERFACE m_axi port=output offset=slave bundle=gmem4 depth=3211264

#pragma HLS INTERFACE s_axilite port=input bundle=control
#pragma HLS INTERFACE s_axilite port=weight bundle=control
#pragma HLS INTERFACE s_axilite port=bias bundle=control
#pragma HLS INTERFACE s_axilite port=output bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control               

    static float C[kNum][kImSize][kImSize];

    for (int i = 0; i < kNum; ++i) {
      for (int h = 0; h < kImSize; ++h) {
        for (int w = 0; w < kImSize; ++w) {
          C[i][h][w] = bias[i];
        }
      }
    }

    // Convolution
    for (int i = 0; i < kNum; ++i) {
      for (int j = 0; j < kNum; ++j) {
        for (int h = 0; h < kImSize; ++h) {
          for (int w = 0; w < kImSize; ++w) {
            for (int p = 0; p < kKernel; ++p) {
              for (int q = 0; q < kKernel; ++q)
                C[i][h][w] += weight(i, j, p, q) * input(j, h + p, w + q);
              }
            }
        }
      }
    }
	
	// ReLU
	for (int i = 0; i < kNum; ++i) {
      for (int h = 0; h < kImSize; ++h) {
        for (int w = 0; w < kImSize; ++w) {
          C[i][h][w] = max(0.f, C[i][h][w]);
        }
      }
    }
	
	// Max pooling
    for (int i = 0; i < kNum; ++i) {
      for (int h = 0; h < kOutImSize; ++h) {
        for (int w = 0; w < kOutImSize; ++w) {
          output(i, h, w) = max(
            max(C[i][h * 2][w * 2    ], C[i][h * 2 + 1][w * 2    ]),
            max(C[i][h * 2][w * 2 + 1], C[i][h * 2 + 1][w * 2 + 1]));
        }
      }
    }
  }
}
