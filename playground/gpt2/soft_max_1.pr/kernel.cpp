
//===------------------------------------------------------------*- C++ -*-===//
//
// Automatically generated file for High-level Synthesis (HLS).
//
//===----------------------------------------------------------------------===//
#include <algorithm>
#include <ap_axi_sdata.h>
#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_math.h>
#include <hls_stream.h>
#include <math.h>
#include <stdint.h>
using namespace std;

extern "C" {

void soft_max(
  float v0[1][50257],
  float v1[1][50257]
) {	// L2
  float max;	// L3
  l_max_i: for (int i = 0; i < 50257; i++) {	// L4
    float v4 = v0[0][i];	// L5
    float analyze;	// L6
    analyze = v4;	// L7
    ap_int<33> v6 = i;	// L8
    bool v7 = v6 == 0;	// L11
    float v8 = max;	// L12
    float v9 = analyze;	// L13
    bool v10 = v9 > v8;	// L14
    bool v11 = v7 | v10;	// L15
    if (v11) {	// L16
      float v12 = analyze;	// L17
      max = v12;	// L18
    }
  }
  float base;	// L23
  base = 0.000000;	// L24
  l_base_i1: for (int i1 = 0; i1 < 50257; i1++) {	// L25
    float v15 = v0[0][i1];	// L26
    float tmp_in;	// L27
    tmp_in = v15;	// L28
    float v17 = tmp_in;	// L29
    float v18 = max;	// L30
    float v19 = v17 - v18;	// L31
    float v20 = exp(v19);	// L32
    float v21 = base;	// L33
    float v22 = v21 + v20;	// L34
    base = v22;	// L35
  }
  l_top_i2: for (int i2 = 0; i2 < 50257; i2++) {	// L38
    float v24 = v0[0][i2];	// L39
    float tmp_in1;	// L40
    tmp_in1 = v24;	// L41
    float v26 = tmp_in1;	// L42
    float v27 = max;	// L43
    float v28 = v26 - v27;	// L44
    float v29 = exp(v28);	// L45
    float v30 = base;	// L46
    float v31 = v29 / v30;	// L47
    v1[0][i2] = v31;	// L48
  }
}

void top(
  float *v32,
  float *v33
) {	// L52
  #pragma HLS interface m_axi port=v32 offset=slave bundle=gmem0
  #pragma HLS interface m_axi port=v33 offset=slave bundle=gmem1
  float buf0[1][50257];	//
  l_S_buf0_buf0_l_0: for (int buf0_l_0 = 0; buf0_l_0 < 1; buf0_l_0++) {	//
    l_buf0_l_1: for (int buf0_l_1 = 0; buf0_l_1 < 50257; buf0_l_1++) {	//
    #pragma HLS pipeline II=1 rewind
      float v37 = v32[((buf0_l_0 * 50257) + buf0_l_1)];	//
      buf0[buf0_l_0][buf0_l_1] = v37;	//
    }
  }
  float v38[1][50257];
  soft_max(buf0, v38);	// L53
  l_S_result1_result1_l_0: for (int result1_l_0 = 0; result1_l_0 < 1; result1_l_0++) {	//
    l_result1_l_1: for (int result1_l_1 = 0; result1_l_1 < 50257; result1_l_1++) {	//
    #pragma HLS pipeline II=1 rewind
      float v41 = v38[result1_l_0][result1_l_1];	//
      v33[((result1_l_0 * 50257) + result1_l_1)] = v41;	//
      if (v41 != 0)
        fprintf(stderr, "%.4e, ", v41);
    }
  }
}


} // extern "C"
