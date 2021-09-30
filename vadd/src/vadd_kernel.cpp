#define DATA_SIZE 256

extern "C" {
void vadd_kernel(int *c, const int *a, const int *b) {
#pragma HLS interface m_axi port = a offset = slave bundle = gmem
#pragma HLS interface m_axi port = b offset = slave bundle = gmem
#pragma HLS interface m_axi port = c offset = slave bundle = gmem
#pragma HLS interface s_axilite port = a bundle = control
#pragma HLS interface s_axilite port = b bundle = control
#pragma HLS interface s_axilite port = c bundle = control
#pragma HLS interface s_axilite port = return bundle = control

vadd_loop:
  for (int i = 0; i < DATA_SIZE; ++i) {
    c[i] = a[i] + b[i];
  }
}
}
