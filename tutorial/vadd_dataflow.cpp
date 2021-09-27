#include <ap_int.h>
#include <hls_stream.h>

#define DATA_SIZE 256
#define BLOCK_SIZE 16

typedef ap_int<512> block_t;

void load(hls::stream<block_t> & fifo_a, const block_t *a_in,
          hls::stream<block_t> & fifo_b, const block_t *b_in) {
#pragma HLS inline off
    for (int i = 0; i < BLOCK_SIZE; i++) {
#pragma HLS pipeline
        block_t a = a_in[i];
	block_t b = b_in[i];
	fifo_a.write(a);
        fifo_b.write(b);
    }
}


void store(hls::stream<block_t> & fifo_c, block_t *c) {
#pragma HLS inline off
    for (int i = 0; i < BLOCK_SIZE; i++) {
#pragma HLS pipeline
	block_t c = fifo_c.read();
	c[i] = c;
    }
}


void compute(hls::stream<block_t> & fifo_a, hls::stream<block_t> & fifo_b, 
             hls::stream<block_t> & fifo_c) {
#pragma HLS inline off
    calc_add: for (int i = 0; i < BLOCK_SIZE; i++) {
#pragma HLS pipeline
        block_t a = fifo_a.read();
        block_t b = fifo_b.read();
        block_t c;
        for (int  p = 0; p < 16; ++p) {
            ap_int<32> a_u32 = a(32 * (p + 1) - 1, 32 * p);
            ap_int<32> b_u32 = b(32 * (p + 1) - 1, 32 * p);
            int a_int = (int) a_u32;
            int b_int = (int) b_u32;
            int c_int = a_int + b_int;
            ap_int<32> c_u32 = (ap_int<32>) c_int;
            c(32 * (p + 1) - 1, 32 * p) = c_u32;
        }
        fifo_c.write(c);
    }
}



extern "C" {


void vadd(block_t *c, const block_t *a, const block_t *b) {
#pragma HLS interface m_axi port = a offset = slave bundle = gmem
#pragma HLS interface m_axi port = b offset = slave bundle = gmem
#pragma HLS interface m_axi port = c offset = slave bundle = gmem
#pragma HLS interface s_axilite port = a bundle = control
#pragma HLS interface s_axilite port = b bundle = control
#pragma HLS interface s_axilite port = c bundle = control
#pragma HLS interface s_axilite port = return bundle = control

    hls::stream<block_t> fifo_a;
    hls::stream<block_t> fifo_b;
    hls::stream<block_t> fifo_c;

#pragma HLS STREAM variable=fifo_a depth=8
#pragma HLS STREAM variable=fifo_b depth=8
#pragma HLS STREAM variable=fifo_c depth=8

#pragma HLS RESOURCE variable=fifo_a core=FIFO_SRL
#pragma HLS RESOURCE variable=fifo_b core=FIFO_SRL
#pragma HLS RESOURCE variable=fifo_c core=FIFO_SRL

    #pragma HLS dataflow
    
    load(fifo_a, a, fifo_b, b);
    compute(fifo_a, fifo_b, fifo_c);
    store(fifo_c, c);
}
}
