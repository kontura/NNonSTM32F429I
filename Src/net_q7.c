
uint32_t net_q7_5layers(q7_t* letter){
  //float32_t l0_feature_maps[24*24*20] = {[0 ... 11519] = 0};
  q7_t l0_feature_maps[24*24*21] = {[0 ... (24*24*21)-1] = 0}; //multiply by 21, cause we need some execess space for insitu convolution
 // q7_t l0_feature_maps[18680+24*24] = {[0 ... (24*24+18680)-1] = 0};
  q7_t l0_pooled_feature_maps[12*12*20] = {[0 ... (12*12*20)-1] = 0};

  q7_t l1_feature_maps[8*8*40] = {[0 ... 2559] = 0};
  q7_t tmp_l1_f_m[200] = {[0 ... (200)-1] = 0};
  q7_t l1_pooled_feature_maps[4*4*40] = {[0 ... (4*4*40)-1] = 0};

  q7_t l2_full_connection[100] = {[0 ... 99] = 0};
  q7_t l3_full_connection[100] = {[0 ... 99] = 0};
  q7_t l4_soft_max_result[10] = {[0 ... 9] = 0};
  q7_t out[10] = {[0 ... 9] = 0};

  uint32_t l0_size = 20;
  uint32_t l1_size = 40;
  uint32_t l2_size = 100;
  uint32_t l3_size = 100;
  uint32_t l4_size = 10;

// l0_w[500]
// l1_w[20000]
// l2_w_o[64000]
// l3_w_o[10000]
// l4_w_o[1000]

  q7_t buffer_q7_t[64000] = {[0 ... 63999] = 0};
  q7_t biases_q7_t[100] = {[0 ... 99] = 0};

  //layer 0
  arm_float_to_q7(l0_w,  buffer_q7_t, 500);
  arm_float_to_q7(l0_b,  biases_q7_t, 20);
  for(uint32_t i=0; i<l0_size; i++){
    //convolution_optimized(letter, 28, l0_feature_maps+(i*24*24), l0_w_o+(i*(5*5+4*23)), 5);
    convolution_additive_q7_t(letter, 28, l0_feature_maps+(i*24*24), buffer_q7_t+(i*(5*5)), 5);
    pooling_optimized_q7_t(l0_feature_maps+(i*24*24), l0_pooled_feature_maps+(i*12*12), 24, &arm_max_q7);

   // for(uint32_t j=0; j<12*12; j++){
   //   (l0_pooled_feature_maps+(i*12*12))[j] = ReLU((l0_pooled_feature_maps+(i*12*12))[j] + l0_b[i]);
   // }
    arm_offset_q7(l0_pooled_feature_maps+(i*12*12), biases_q7_t[i], l0_pooled_feature_maps+(i*12*12), 12*12);
  }
  arm_fn_q7(l0_pooled_feature_maps, l0_pooled_feature_maps, (12*12)*l0_size, &ReLU_q7);

  
  //layer 1
  arm_float_to_q7(l1_w,  buffer_q7_t, 20000);
  arm_float_to_q7(l1_b,  biases_q7_t, 40);
  for(uint32_t i=0; i<l1_size; i++){
    for(uint32_t j=0; j<l0_size; j++){
      //convolution_optimized(l0_pooled_feature_maps+(j*12*12), 12, tmp_l1_f_m, l1_w_o+((i*(5*5+4*7)*l0_size)+(j*(5*5+4*7))), 5);
      //arm_add_f32(l1_feature_maps+(i*8*8), tmp_l1_f_m, l1_feature_maps+(i*8*8), 8*8);
      convolution_additive_q7_t(l0_pooled_feature_maps+(j*12*12), 12, l1_feature_maps+(i*8*8), buffer_q7_t+((i*(5*5)*l0_size)+(j*(5*5))), 5);
    }
    pooling_optimized_q7_t(l1_feature_maps+(i*8*8), l1_pooled_feature_maps+(i*4*4), 8, &arm_max_q7);
   // for(uint32_t j=0; j<4*4; j++){
   //   (l1_pooled_feature_maps+(i*4*4))[j] = ReLU((l1_pooled_feature_maps+(i*4*4))[j] + l1_b[i]);
   // }
    arm_offset_q7(l1_pooled_feature_maps+(i*4*4), biases_q7_t[i], l1_pooled_feature_maps+(i*4*4), 4*4);
  }
  arm_fn_q7(l1_pooled_feature_maps, l1_pooled_feature_maps, (4*4)*l1_size, &ReLU_q7);

  
  //layer 2
  arm_float_to_q7(l2_w_o,  buffer_q7_t, 64000);
  arm_float_to_q7(l2_b,  biases_q7_t, 100);
  for(uint32_t i=0; i<l2_size; i++){
   // for(uint32_t j=0; j<l1_size;j++){
   //   //l2_full_connection[i] += dot_product_with_nth_column(l1_pooled_feature_maps+(j*4*4), l2_w+((j*4*4*100)+i), 4*4, 100);

   //   arm_dot_prod_f32(l1_pooled_feature_maps+(j*4*4), l2_w_o+((i*4*4*40)+(j*4*4)) ,4*4 , &tmp);
   //   l2_full_connection[i] += tmp;
   // }
    arm_dot_prod_q7(l1_pooled_feature_maps, buffer_q7_t+(i*4*4*40), 4*4*40, l2_full_connection+i);
   // l2_full_connection[i] = ReLU(l2_full_connection[i] + l2_b[i]);
  }
  arm_add_q7(l2_full_connection, biases_q7_t, l2_full_connection, l2_size);
  arm_fn_q7(l2_full_connection, l2_full_connection, l2_size, &ReLU_q7);

  
  //layer 3
  arm_float_to_q7(l3_w_o,  buffer_q7_t, 10000);
  arm_float_to_q7(l3_b,  biases_q7_t, 100);
  for(uint32_t i=0; i<l3_size; i++){
    //l3_full_connection[i] = dot_product_with_nth_column(l2_full_connection, l3_w+i, 100, 100);
    arm_dot_prod_q7(l2_full_connection, biases_q7_t+(i*l2_size), l2_size, l3_full_connection+i);
//    l3_full_connection[i] = ReLU(l3_full_connection[i] + l3_b[i]);
  }
  arm_add_q7(l3_full_connection, biases_q7_t, l3_full_connection, l3_size);
  arm_fn_q7(l3_full_connection, l3_full_connection, l3_size, &ReLU_q7);

  //layer 4
  arm_float_to_q7(l4_w_o,  buffer_q7_t, 10000);
  arm_float_to_q7(l4_b,  biases_q7_t, 100);
  for(uint32_t i=0; i<l4_size; i++){
    //l4_soft_max_result[i] = dot_product_with_nth_column(l3_full_connection, l4_w+i, 100, 10);
    arm_dot_prod_q7(l3_full_connection, buffer_q7_t+(i*l3_size), l3_size, l4_soft_max_result+i);
    //l4_soft_max_result[i] = l4_soft_max_result[i] + l4_b[i];
  }
  arm_add_q7(l4_soft_max_result, biases_q7_t, l4_soft_max_result, l4_size);
  soft_max_q7(l4_soft_max_result, 10, out);

  return index_of_most_probable_q7(out);
}
