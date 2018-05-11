
uint32_t net_q9_5layers(q15_t* letter){
  q15_t l0_feature_maps[24*24*21] = {[0 ... (24*24*21)-1] = 0}; //multiply by 21, cause we need some execess space for insitu convolution
  q15_t l0_pooled_feature_maps[12*12*20] = {[0 ... (12*12*20)-1] = 0};

  q15_t l1_feature_maps[8*8*40] = {[0 ... 2559] = 0};
  q15_t l1_pooled_feature_maps[4*4*40] = {[0 ... (4*4*40)-1] = 0};

  q15_t l2_full_connection[100] = {[0 ... 99] = 0};
  q15_t l3_full_connection[100] = {[0 ... 99] = 0};
  q15_t l4_soft_max_result[10] = {[0 ... 9] = 0};
  q15_t out[10] = {[0 ... 9] = 0};

  uint32_t l0_size = 20;
  uint32_t l1_size = 40;
  uint32_t l2_size = 100;
  uint32_t l3_size = 100;
  uint32_t l4_size = 10;

  //layer 0
  for(uint32_t i=0; i<l0_size; i++){
    //convolution_optimized(letter, 28, l0_feature_maps+(i*24*24), l0_w_o+(i*(5*5+4*23)), 5);
    convolution_additive_q9_t(letter, 28, l0_feature_maps+(i*24*24), l0_w+(i*(5*5)), 5);
    pooling_optimized_q9_t(l0_feature_maps+(i*24*24), l0_pooled_feature_maps+(i*12*12), 24, &arm_max_q15);

    for(uint32_t j=0; j<12*12; j++){
      (l0_pooled_feature_maps+(i*12*12))[j] = ReLU_q15(__SSAT(((l0_pooled_feature_maps+(i*12*12))[j] + l0_b[i]), 16));
    }
    //arm_offset_q15(l0_pooled_feature_maps+(i*12*12), biases_q9_t[i], l0_pooled_feature_maps+(i*12*12), 12*12);
  }
  //arm_fn_q15(l0_pooled_feature_maps, l0_pooled_feature_maps, (12*12)*l0_size, &ReLU_q15);

  
  //layer 1
  for(uint32_t i=0; i<l1_size; i++){
    for(uint32_t j=0; j<l0_size; j++){
      //convolution_optimized(l0_pooled_feature_maps+(j*12*12), 12, tmp_l1_f_m, l1_w_o+((i*(5*5+4*7)*l0_size)+(j*(5*5+4*7))), 5);
      //arm_add_f32(l1_feature_maps+(i*8*8), tmp_l1_f_m, l1_feature_maps+(i*8*8), 8*8);
      convolution_additive_q9_t(l0_pooled_feature_maps+(j*12*12), 12, l1_feature_maps+(i*8*8), l1_w+((i*(5*5)*l0_size)+(j*(5*5))), 5);
    }
    pooling_optimized_q9_t(l1_feature_maps+(i*8*8), l1_pooled_feature_maps+(i*4*4), 8, &arm_max_q15);
    for(uint32_t j=0; j<4*4; j++){
      (l1_pooled_feature_maps+(i*4*4))[j] = ReLU_q15(__SSAT(((l1_pooled_feature_maps+(i*4*4))[j] + l1_b[i]), 16));
    }
   // arm_offset_q15(l1_pooled_feature_maps+(i*4*4), biases_q9_t[i], l1_pooled_feature_maps+(i*4*4), 4*4);
  }
//  arm_q9_to_float(l1_pooled_feature_maps, l1_pooled_feature_maps_f, 4*4*40);
  //arm_fn_q15(l1_pooled_feature_maps, l1_pooled_feature_maps, (4*4)*l1_size, &ReLU_q15);

  
  //layer 2
  for(uint32_t i=0; i<l2_size; i++){
    for(uint32_t j=0; j<l1_size;j++){
      //l2_full_connection[i] = (q15_t) __SSAT(l2_full_connection[i] + dot_product_with_nth_column(l1_pooled_feature_maps+(j*4*4), buffer_q9_t+((j*4*4*100)+i), 4*4, 100), 16);
      l2_full_connection[i] = (q15_t) __SSAT(l2_full_connection[i] + dot_product_q15(l1_pooled_feature_maps+(j*4*4), l2_w_o+((i*4*4*40)+(j*4*4)), (uint64_t) (4*4)), 16);

   //   arm_dot_prod_f32(l1_pooled_feature_maps+(j*4*4), l2_w_o+((i*4*4*40)+(j*4*4)) ,4*4 , &tmp);
    //  l2_full_connection[i] += tmp;
    }
   //   arm_dot_prod_q15(l1_pooled_feature_maps, buffer_q9_t+(i*4*4*40), 4*4*40, l2_full_connection+i);
    l2_full_connection[i] = ReLU_q15(__SSAT((l2_full_connection[i] + l2_b[i]), 16));
  }
  //arm_add_q15(l2_full_connection, biases_q9_t, l2_full_connection, l2_size);
  //arm_fn_q15(l2_full_connection, l2_full_connection, l2_size, &ReLU_q15);

  
  //layer 3
  for(uint32_t i=0; i<l3_size; i++){
    //l3_full_connection[i] = dot_product_with_nth_column(l2_full_connection, l3_w+i, 100, 100);
    //arm_dot_prod_q15(l2_full_connection, biases_q9_t+(i*l2_size), l2_size, l3_full_connection+i);
    l3_full_connection[i] = dot_product_q15(l2_full_connection, l3_w_o+(i*l2_size), (uint64_t) l2_size);
    l3_full_connection[i] = ReLU_q15( __SSAT((l3_full_connection[i] + l3_b[i]), 16));
  }
  //arm_add_q15(l3_full_connection, biases_q9_t, l3_full_connection, l3_size);
  //arm_fn_q15(l3_full_connection, l3_full_connection, l3_size, &ReLU_q15);

  //layer 4
  for(uint32_t i=0; i<l4_size; i++){
    //l4_soft_max_result[i] = dot_product_with_nth_column(l3_full_connection, l4_w+i, 100, 10);
    //arm_dot_prod_q15(l3_full_connection, buffer_q9_t+(i*l3_size), l3_size, l4_soft_max_result+i);
    l4_soft_max_result[i] = dot_product_q15(l3_full_connection, l4_w_o+(i*l3_size), (uint64_t) l3_size);
    l4_soft_max_result[i] = __SSAT(l4_soft_max_result[i] + l4_b[i], 16);
  }
  //arm_add_q15(l4_soft_max_result, biases_q9_t, l4_soft_max_result, l4_size);
  soft_max_q15(l4_soft_max_result, 10, out);

  return index_of_most_probable_q15(out);
}
