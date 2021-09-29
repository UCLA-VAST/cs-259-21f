#include <ap_int.h>

#define DATA_SIZE 256
#define BLOCK_SIZE 4
#define DATA_SIZE_IN_A_BUFFER (BLOCK_SIZE * 16)

typedef ap_int<512> block_t;

extern "C" {

void load(const bool enable,
          const int i_base,
          block_t a_buf[BLOCK_SIZE],
          const block_t *a_in,
          block_t b_buf[BLOCK_SIZE],
          const block_t *b_in) {
#pragma HLS inline off
    if (enable) {
        for (int i = 0; i < BLOCK_SIZE; i++) {
            a_buf[i] = a_in[i_base + i];
        }
        for (int i = 0; i < BLOCK_SIZE; i++) {
            b_buf[i] = b_in[i_base + i];
        }
    }
}

void store(const bool enable,
           const int i_base,
           block_t c_buf[BLOCK_SIZE],
           block_t *c) {
#pragma HLS inline off
    if (enable) {
        for (int i = 0; i < BLOCK_SIZE; i++) {
            c[i_base + i] = c_buf[i];
        }
    }
}

void compute(const bool enable,
             const block_t a_buf[BLOCK_SIZE],
             const block_t b_buf[BLOCK_SIZE],
             block_t c_buf[BLOCK_SIZE]) {
#pragma HLS inline off
    int a_buf_normal[DATA_SIZE_IN_A_BUFFER], b_buf_normal[DATA_SIZE_IN_A_BUFFER];
#pragma HLS array_partition variable=a_buf_normal complete
#pragma HLS array_partition variable=b_buf_normal complete
    int c_buf_normal[DATA_SIZE_IN_A_BUFFER];
#pragma HLS array_partition variable=c_buf_normal complete
    
    if (enable) {
        copy_a_buf: for (int i = 0; i < BLOCK_SIZE; i++) {
            for (int j = 0; j < 16; j++) {
                ap_int<32> val = a_buf[i](32 * (j + 1) - 1, 32 * j);
                a_buf_normal[i * 16 + j] = (int) val;
            }
        }
        
        copy_b_buf: for (int i = 0; i < BLOCK_SIZE; i++) {
            for (int j = 0; j < 16; j++) {
                ap_int<32> val = b_buf[i](32 * (j + 1) - 1, 32 * j);
                b_buf_normal[i * 16 + j] = (int) val;
            }
        }
        
        calc_add: for (int i = 0; i < DATA_SIZE_IN_A_BUFFER; i++) {
#pragma HLS unroll factor=2
            c_buf_normal[i] = a_buf_normal[i] + b_buf_normal[i];
        }
        
        copy_c_buf: for (int i = 0; i < BLOCK_SIZE; i++) {
            for (int j = 0; j < 16; j++) {
                c_buf[i](32 * (j + 1) - 1, 32 * j) = (ap_int<32>) c_buf_normal[i * 16 + j];
            }
        }
    }
}

void vadd(block_t *c, const block_t *a, const block_t *b) {
#pragma HLS interface m_axi port = a offset = slave bundle = gmem
#pragma HLS interface m_axi port = b offset = slave bundle = gmem
#pragma HLS interface m_axi port = c offset = slave bundle = gmem
#pragma HLS interface s_axilite port = a bundle = control
#pragma HLS interface s_axilite port = b bundle = control
#pragma HLS interface s_axilite port = c bundle = control
#pragma HLS interface s_axilite port = return bundle = control

    block_t a_buf_0[BLOCK_SIZE];
    block_t a_buf_1[BLOCK_SIZE];
    
    block_t b_buf_0[BLOCK_SIZE];
    block_t b_buf_1[BLOCK_SIZE];
    
    block_t c_buf_0[BLOCK_SIZE];
    block_t c_buf_1[BLOCK_SIZE];
    
    for (int i = 0; i < DATA_SIZE/DATA_SIZE_IN_A_BUFFER + 2; ++i) {
        if (i % 2 == 0) {
            load(i < DATA_SIZE/DATA_SIZE_IN_A_BUFFER, i, a_buf_0, a, b_buf_0, b);
            compute((i > 0) && (i < DATA_SIZE/DATA_SIZE_IN_A_BUFFER + 1), a_buf_1, b_buf_1, c_buf_1);
            store(i > 1, i, c_buf_0, c);
        } else {
            load(i < DATA_SIZE/DATA_SIZE_IN_A_BUFFER, i, a_buf_1, a, b_buf_1, b);
            compute((i > 0) && (i < DATA_SIZE/DATA_SIZE_IN_A_BUFFER + 1), a_buf_0, b_buf_0, c_buf_0);
            store(i > 1, i, c_buf_1, c);
        }
    }
}
}
