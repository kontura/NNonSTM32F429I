
#include "time_profiling.h"

extern const float32_t l1_w_o[48000];

void time_profiling(TIM_HandleTypeDef TimHandle){
  uint64_t sigmoid_results[2] = {0,0};
 // profile_sigmoid(TimHandle, sigmoid_results);

  uint64_t convolution_results[6] = {0,0,0,0,0,0,0};
  profile_convolution(TimHandle, convolution_results);

  uint64_t dot_product_results[5] = {0,0,0,0,0,0};
  profile_dot_product(TimHandle, dot_product_results);

  uint64_t offset_results[5] = {0,0,0,0,0,0};
  profile_offset(TimHandle, offset_results);

  uint64_t add_results[5] = {0,0,0,0,0,0};
  profile_add(TimHandle, add_results);

  BSP_LED_On(LED4);     
  while(1){}
}

void profile_dot_product(TIM_HandleTypeDef TimHandle, uint64_t *results){
  float32_t out1 = 0;
  float32_t out2 = 0;
  q63_t out3 = 0;
  q31_t out4 = 0;
  q15_t out5 = 0;
  q31_t l1_w_o_q31[20000] = {[0 ... 19999] = 0};
  q15_t l1_w_o_q15[20000] = {[0 ... 19999] = 0};
  q7_t l1_w_o_q7[20000] = {[0 ... 19999] = 0};
  uint64_t size = 20000;
  arm_float_to_q31(l1_w_o,  l1_w_o_q31, size);
  arm_float_to_q15(l1_w_o,  l1_w_o_q15, size);
  arm_float_to_q7(l1_w_o,  l1_w_o_q7, size);

  start_time_measure(TimHandle);
  (dot_product(l1_w_o, l1_w_o, (uint64_t) size));
  results[0] = stop_time_measure(TimHandle);

  start_time_measure(TimHandle);
  (arm_dot_prod_f32(l1_w_o, l1_w_o, size, &out2));
  results[1] = stop_time_measure(TimHandle);

  start_time_measure(TimHandle);
  arm_dot_prod_q31(l1_w_o_q31, l1_w_o_q31, size, &out3);
  results[2] = stop_time_measure(TimHandle);

  start_time_measure(TimHandle);
  (arm_dot_prod_q15(l1_w_o_q15, l1_w_o_q15, size, &out4));
  results[3] = stop_time_measure(TimHandle);

  start_time_measure(TimHandle);
  arm_dot_prod_q7(l1_w_o_q7, l1_w_o_q7, size, &out5);
  results[4] = stop_time_measure(TimHandle);
  
  start_time_measure(TimHandle);
  INF_LOOP_TOGGL_PIN(dot_product_q15(l1_w_o_q15, l1_w_o_q15, (uint64_t) size));
  results[5] = stop_time_measure(TimHandle);

}

void profile_convolution(TIM_HandleTypeDef TimHandle, uint64_t *results){
  float32_t out1[4000] = {[0 ... 3999] = 0};
  float32_t out2[4000] = {[0 ... 3999] = 0};
  float32_t out6[4000] = {[0 ... 3999] = 0};
  q7_t out3[4000] = {[0 ... 3999] = 0};
  q15_t out4[4000] = {[0 ... 3999] = 0};
  q31_t out5[4000] = {[0 ... 3999] = 0};

  q7_t l1_w_o_q7[4000] = {[0 ... 3999] = 0};
  q15_t l1_w_o_q15[4000] = {[0 ... 3999] = 0};
  q31_t l1_w_o_q31[4000] = {[0 ... 3999] = 0};
  uint32_t size = 4000;
  arm_float_to_q7(l1_w_o,  l1_w_o_q7, size);
  arm_float_to_q15(l1_w_o,  l1_w_o_q15, size);
  arm_float_to_q31(l1_w_o,  l1_w_o_q31, size);

  start_time_measure(TimHandle);
  (convolution_additive_q9_t(l1_w_o_q15, 50, out4, l1_w_o_q15, 5));
  results[0] = stop_time_measure(TimHandle);

  start_time_measure(TimHandle);
  arm_conv_f32(l1_w_o, size, l1_w_o, size, out2);
  results[1] = stop_time_measure(TimHandle);

  start_time_measure(TimHandle);
  arm_conv_q7(l1_w_o_q7, size, l1_w_o_q7, size, out3);
  results[2] = stop_time_measure(TimHandle);

  start_time_measure(TimHandle);
  arm_conv_q15(l1_w_o_q15, size, l1_w_o_q15, size, out4);
  results[3] = stop_time_measure(TimHandle);

  start_time_measure(TimHandle);
  arm_conv_q31(l1_w_o_q31, size, l1_w_o_q31, size, out5);
  results[4] = stop_time_measure(TimHandle);

  start_time_measure(TimHandle);
  (convolution_additive(l1_w_o+2000, 50, out6, l1_w_o, 5));
  results[5] = stop_time_measure(TimHandle);

  start_time_measure(TimHandle);
  (convolution_optimized(l1_w_o+3999, 50, out1, l1_w_o, 5));
  results[6] = stop_time_measure(TimHandle);
}

void profile_relu(TIM_HandleTypeDef TimHandle, uint64_t *results){
  float32_t out1[20000] = {[0 ... 19999] = 0};
  float32_t out2[20000] = {[0 ... 19999] = 0};

  start_time_measure(TimHandle);
  for(uint32_t j=0; j<20000; j++){
    out1[j] = sigmoid(l1_w_o[j]);
  }
  results[0] = stop_time_measure(TimHandle);

  start_time_measure(TimHandle);
  arm_fn_f32(l1_w_o, out2, 20000, &sigmoid);
  results[1] = stop_time_measure(TimHandle);
}

void profile_sigmoid(TIM_HandleTypeDef TimHandle, uint64_t *results){
  float32_t out1[20000] = {[0 ... 19999] = 0};
  float32_t out2[20000] = {[0 ... 19999] = 0};
  q15_t out4[20000] = {[0 ... 19999] = 0};
  q15_t l1_w_o_q15[20000] = {[0 ... 19999] = 0};

  uint32_t size = 20000;
  arm_float_to_q15(l1_w_o,  l1_w_o_q15, size);

  start_time_measure(TimHandle);
  for(uint32_t j=0; j<20000; j++){
    out1[j] = ReLU(l1_w_o[j]);
  }
  results[0] = stop_time_measure(TimHandle);

  start_time_measure(TimHandle);
  for(uint32_t j=0; j<20000; j++){
    out1[j] = ReLU_q15(l1_w_o[j]);
  }
  results[1] = stop_time_measure(TimHandle);

  start_time_measure(TimHandle);
  arm_fn_f32(l1_w_o, out2, 20000, &ReLU);
  results[2] = stop_time_measure(TimHandle);

  start_time_measure(TimHandle);
  arm_fn_f32(l1_w_o, out2, 20000, &ReLU_q15);
  results[3] = stop_time_measure(TimHandle);
}

void profile_add(TIM_HandleTypeDef TimHandle, uint64_t *results){
  float32_t out1[20000] = {[0 ... 19999] = 0};
  float32_t out2[20000] = {[0 ... 19999] = 0};
  q15_t out4[20000] = {[0 ... 19999] = 0};
  q15_t l1_w_o_q15[20000] = {[0 ... 19999] = 0};

  uint32_t size = 20000;
  arm_float_to_q15(l1_w_o,  l1_w_o_q15, size);

  start_time_measure(TimHandle);
  for(uint32_t j=0; j<size; j++){
    out1[j] = l1_w_o[j] + l1_w_o[j];
  }
  results[0] = stop_time_measure(TimHandle);

  start_time_measure(TimHandle);
  arm_add_f32(l1_w_o, l1_w_o, out2, size);
  results[1] = stop_time_measure(TimHandle);

  start_time_measure(TimHandle);
  arm_add_q15(l1_w_o_q15, l1_w_o_q15, out4, size);
  results[2] = stop_time_measure(TimHandle);
}

void profile_offset(TIM_HandleTypeDef TimHandle, uint64_t *results){
  float32_t out1[20000] = {[0 ... 19999] = 0};
  float32_t out2[20000] = {[0 ... 19999] = 0};
  q15_t out4[20000] = {[0 ... 19999] = 0};
  q15_t l1_w_o_q15[20000] = {[0 ... 19999] = 0};

  uint32_t size = 20000;
  arm_float_to_q15(l1_w_o,  l1_w_o_q15, size);

  start_time_measure(TimHandle);
  for(uint32_t j=0; j<size; j++){
    out1[j] = l1_w_o[j] + l1_w_o[j];
  }
  results[0] = stop_time_measure(TimHandle);

  start_time_measure(TimHandle);
  arm_offset_f32(l1_w_o, l1_w_o[1], out2, size);
  results[1] = stop_time_measure(TimHandle);

  start_time_measure(TimHandle);
  arm_offset_q15(l1_w_o_q15, l1_w_o_q15[1], out4, size);
  results[2] = stop_time_measure(TimHandle);
}
