#include <ap_int.h>

#define DATA_SIZE 256

extern "C" {

void load(int a_buf[DATA_SIZE], const int *a_in,
          int b_buf[DATA_SIZE], const int *b_in) {
    for (int i = 0; i < DATA_SIZE; i++) {
        a_buf[i] = a_in[i];
    }
    for (int i = 0; i < DATA_SIZE; i++) {
        b_buf[i] = b_in[i];
    }
}

void store(int c_buf[DATA_SIZE], int *c) {
    for (int i = 0; i < DATA_SIZE; i++) {
        c[i] = c_buf[i];
    }
}

void compute(const int a_buf[DATA_SIZE], const int b_buf[DATA_SIZE], int c_buf[DATA_SIZE]) {
    int a_buf_normal[DATA_SIZE], b_buf_normal[DATA_SIZE];
#pragma HLS array_partition variable=a_buf_normal complete
#pragma HLS array_partition variable=b_buf_normal complete
    int c_buf_normal[DATA_SIZE];
#pragma HLS array_partition variable=c_buf_normal complete

    copy_a_buf: for (int i = 0; i < DATA_SIZE; i++) {
        a_buf_normal[i] = a_buf[i];
    }

    copy_b_buf: for (int i = 0; i < DATA_SIZE; i++) {
        b_buf_normal[i] = b_buf[i];
    }

    calc_add: for (int i = 0; i < DATA_SIZE; i++) {
        c_buf_normal[i] = a_buf_normal[i] + b_buf_normal[i];
    }

    copy_c_buf: for (int i = 0; i < DATA_SIZE; i++) {
        c_buf[i] = c_buf_normal[i];
    }

    return;
}

void vadd(int *c, const int *a, const int *b) {
#pragma HLS interface m_axi port = a offset = slave bundle = gmem
#pragma HLS interface m_axi port = b offset = slave bundle = gmem
#pragma HLS interface m_axi port = c offset = slave bundle = gmem
#pragma HLS interface s_axilite port = a bundle = control
#pragma HLS interface s_axilite port = b bundle = control
#pragma HLS interface s_axilite port = c bundle = control
#pragma HLS interface s_axilite port = return bundle = control

    int a_buf[DATA_SIZE], b_buf[DATA_SIZE];
    int c_buf[DATA_SIZE];

    load(a_buf, a, b_buf, b);
    compute(a_buf, b_buf, c_buf);
    store(c_buf, c);
}
}
