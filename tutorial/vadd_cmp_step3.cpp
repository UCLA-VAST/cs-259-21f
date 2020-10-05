#include <ap_int.h>

#define DATA_SIZE 256
#define BLOCK_SIZE 16

typedef ap_int<512> block_t;

extern "C" {

void load(block_t a_buf[BLOCK_SIZE], const block_t *a_in,
          block_t b_buf[BLOCK_SIZE], const block_t *b_in) {
    for (int i = 0; i < BLOCK_SIZE; i++) {
        a_buf[i] = a_in[i];
    }
    for (int i = 0; i < BLOCK_SIZE; i++) {
        b_buf[i] = b_in[i];
    }
}

void store(block_t c_buf[BLOCK_SIZE], block_t *c) {
    for (int i = 0; i < BLOCK_SIZE; i++) {
        c[i] = c_buf[i];
    }
}

void compute(const block_t a_buf[BLOCK_SIZE], const block_t b_buf[BLOCK_SIZE], 
             block_t c_buf[BLOCK_SIZE]) {
    int a_buf_normal[DATA_SIZE], b_buf_normal[DATA_SIZE];
#pragma HLS array_partition variable=a_buf_normal complete
#pragma HLS array_partition variable=b_buf_normal complete
    int c_buf_normal[DATA_SIZE];
#pragma HLS array_partition variable=c_buf_normal complete

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

    calc_add: for (int i = 0; i < DATA_SIZE; i++) {
#pragma HLS unroll factor=2
        c_buf_normal[i] = a_buf_normal[i] + b_buf_normal[i];
    }

    copy_c_buf: for (int i = 0; i < BLOCK_SIZE; i++) {
        for (int j = 0; j < 16; j++) {
            c_buf[i](32 * (j + 1) - 1, 32 * j) = (ap_int<32>) c_buf_normal[i * 16 + j];
        }
    }

    return;
}

void vadd(block_t *c, const block_t *a, const block_t *b) {
#pragma HLS interface m_axi port = a offset = slave bundle = gmem
#pragma HLS interface m_axi port = b offset = slave bundle = gmem
#pragma HLS interface m_axi port = c offset = slave bundle = gmem
#pragma HLS interface s_axilite port = a bundle = control
#pragma HLS interface s_axilite port = b bundle = control
#pragma HLS interface s_axilite port = c bundle = control
#pragma HLS interface s_axilite port = return bundle = control

    block_t a_buf[BLOCK_SIZE], b_buf[BLOCK_SIZE];
    block_t c_buf[BLOCK_SIZE];

    load(a_buf, a, b_buf, b);
    compute(a_buf, b_buf, c_buf);
    store(c_buf, c);
}
}
