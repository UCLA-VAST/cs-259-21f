#include <assert.h>

extern "C" {

void dot_prod_kernel(const float *a, const float *b, float *c, const int num_elems) {
#pragma HLS interface m_axi port = a offset = slave bundle = gmem
#pragma HLS interface m_axi port = b offset = slave bundle = gmem
#pragma HLS interface m_axi port = c offset = slave bundle = gmem
#pragma HLS interface s_axilite port = a bundle = control
#pragma HLS interface s_axilite port = b bundle = control
#pragma HLS interface s_axilite port = c bundle = control
#pragma HLS interface s_axilite port = num_elems bundle = control
#pragma HLS interface s_axilite port = return bundle = control
  assert(num_elems <= 4096); // this helps HLS estimate the loop trip count
  /***************************
   * your code goes here ... *
   ***************************/
}

}  // extern "C"
