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
#pragma HLS pipeline II=1
            a_buf[i] = a_in[i_base + i];
        }
        for (int i = 0; i < BLOCK_SIZE; i++) {
#pragma HLS pipeline II=1
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
#pragma HLS pipeline II=1
            c[i_base + i] = c_buf[i];
        }
    }
}

void compute(const bool enable,
             const block_t a_buf[BLOCK_SIZE],
             const block_t b_buf[BLOCK_SIZE],
             block_t c_buf[BLOCK_SIZE]) {
#pragma HLS inline off
    if (enable) {
        for (int i = 0; i < BLOCK_SIZE; i++) {
#pragma HLS pipeline II=1
            block_t a_u512 = a_buf[i];
            block_t b_u512 = b_buf[i];
            block_t c_u512;
            for (int j = 0; j < 16; j++) {
                ap_int<32> val_a = a_u512(32 * (j + 1) - 1, 32 * j);
                ap_int<32> val_b = b_u512(32 * (j + 1) - 1, 32 * j);
                int a = (int) val_a;
                int b = (int) val_b;
		int c = a + b;
                c_u512(32 * (j + 1) - 1, 32 * j) = (ap_int<32>) c;
            }
            c_buf[i] = c_u512;
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
            load(i < DATA_SIZE/DATA_SIZE_IN_A_BUFFER, i * BLOCK_SIZE, a_buf_0, a, b_buf_0, b);
            compute((i > 0) && (i < DATA_SIZE/DATA_SIZE_IN_A_BUFFER + 1), a_buf_1, b_buf_1, c_buf_1);
            store(i > 1, (i - 2) * BLOCK_SIZE, c_buf_0, c);
        } else {
            load(i < DATA_SIZE/DATA_SIZE_IN_A_BUFFER, i * BLOCK_SIZE, a_buf_1, a, b_buf_1, b);
            compute((i > 0) && (i < DATA_SIZE/DATA_SIZE_IN_A_BUFFER + 1), a_buf_0, b_buf_0, c_buf_0);
            store(i > 1, (i - 2) * BLOCK_SIZE, c_buf_1, c);
        }
    }
}
}
