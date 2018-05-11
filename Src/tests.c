
#include "tests.h"

extern const float32_t num1[786];
extern const float32_t num2[786];
extern const float32_t num3[786];
extern const float32_t num4[786];
extern const float32_t num5[786];
extern const float32_t num6[786];
extern const float32_t num8[786];

extern const float32_t num10[786];
extern const float32_t num11[786];
extern const float32_t num12[786];
extern const float32_t num13[786];
extern const float32_t num14[786];
extern const float32_t num15[786];
extern const float32_t num16[786];
extern const float32_t num17[786];
extern const float32_t num18[786];
extern const float32_t num19[786];

extern const float32_t num20[786];
extern const float32_t num21[786];
extern const float32_t num22[786];
extern const float32_t num23[786];
extern const float32_t num24[786];
extern const float32_t num25[786];
extern const float32_t num26[786];
extern const float32_t num27[786];
extern const float32_t num28[786];
extern const float32_t num29[786];

extern const float32_t test_num0[786];
extern const float32_t test_num1[786];
extern const float32_t test_num2[786];
extern const float32_t test_num3[786];
extern const float32_t test_num4[786];
extern const float32_t test_num5[786];
extern const float32_t test_num6[786];
extern const float32_t test_num7[786];
extern const float32_t test_num8[786];

uint8_t test(){
  if (pooling_tests() != 1) return 0;
  if (sigmoid_tests() != 1) return 0;
  if (convolution_tests() != 1) return 0;
  if (convolution_optimized_tests() != 1) return 0;
  if (soft_max_tests() != 1) return 0;
  if (ReLU_tests() != 1) return 0;
  if (dot_product_tests() != 1) return 0;
  if (most_probable_tests() != 1) return 0;


  return 1;
}

uint8_t float_equality(float32_t a, float32_t b, float32_t eps){
  if ((a - b) > eps || (b - a) > eps){
    return 0;
  }else{
    return 1;
  }
}

uint8_t float_array_equality(float32_t* a, float32_t* b, uint8_t size, float32_t eps){
  for(uint8_t i=0; i<size; i++){
    if (!float_equality(a[i],b[i],eps)) return 0;
  }
  return 1;
}

uint8_t sigmoid_tests(){
  float32_t out = 0.880797077;
  float32_t eps = 0.0001;
  if (!float_equality(sigmoid(2),out,eps)) return 0;
  out = 0.5;
  if (!float_equality(sigmoid(0),out,eps)) return 0;
  out = 0.370051;
  if (!float_equality(sigmoid(-0.532),out,eps)) return 0;

  return 1;
}

uint8_t pooling_tests(){
  float32_t feature_map[576] = {[0 ... 575] = 5};
  float32_t output[144] = {[0 ... 143] = 0};
  pooling(feature_map, output, 24, &max);
  for(uint32_t i=0; i<144; i++){
    if (output[i] != 5) return 0;
  }


  ///second
  float32_t feature_map2[16] = {1, 2, 3, 4,
                               4, 9, 0, 3,
                               0, 3, 1, 22,
                               0, 0, 2, 13};
  float32_t output2[4] = {0};
  pooling(feature_map2, output2, 4, &max);
  if (output2[0] != 9) return 0;
  if (output2[1] != 4) return 0;
  if (output2[2] != 3) return 0;
  if (output2[3] != 22) return 0;

  return 1;
}

uint8_t convolution_tests(){
  uint32_t input_side = 28;
  uint32_t weights_side = 5;
  float32_t in[784] = {[0 ... 783] = 5};
  float32_t output[576] = {[0 ... 575] = 0};
  float32_t weights[25] = {[0 ... 24] = 1};
  float32_t bias = 0;

  convolution_with_activation(in, input_side, output,  weights, weights_side, bias, &sigmoid);

  float32_t eps = 0.0001;
  float32_t out = 0.99999999;
  for(uint32_t i=0; i<576; i++){
    if (!float_equality(output[i],out,eps)) return 0;
  }

  //second
  input_side = 4;
  weights_side = 2;
  float32_t in2[16] = {1, 2, 3, 4,
                       4, 9, 0, 3,
                       0, 3, 1, 22,
                       0, 0, 2, 13};

  float32_t output2[9] = {[0 ... 8] = 0};
  float32_t weights2[4] = {0.5, 0.1,
                           2,   0};
  bias = 0.3;
  //test1
  convolution_with_activation(in2, input_side, output2,  weights2, weights_side, bias, &sigmoid);
  float32_t correct_results[9] = {0.9998766 ,1 ,0.9002495 ,0.9608343 ,0.9999796 ,0.9308616 ,0.6456563 ,0.8698915 ,0.9990889};
  if (!float_array_equality(output2,correct_results,9,eps)) return 0;

  //test2
  float32_t output3[9] = {0, 0, 0,
                          0, 3, 0,
                          0, 0, 1};
  convolution_additive(in2, input_side, output3,  weights2, weights_side);
  float32_t correct_results2[9] = {8.7 ,19.3 ,1.9 ,2.9 ,10.5+3 ,2.3 ,0.3 ,1.6 ,6.7+1};
  if (!float_array_equality(output3,correct_results2,9,eps)) return 0;

  //test3 (subarray of inputs)
  float32_t in4[24] = {3, 5, 3, 9,
                       1, 2, 3, 4,
                       4, 9, 0, 3,
                       0, 3, 1, 22,
                       0, 0, 2, 13,
                       3, 9, 1, 888};

  float32_t output4[9] = {[0 ... 8] = 0};
  float32_t weights4[4] = {0.5, 0.1,
                           2,   0};
  convolution_additive(in4+4, input_side, output4,  weights4, weights_side);
  float32_t correct_results4[9] = {8.7 ,19.3 ,1.9 ,2.9 ,10.5 ,2.3 ,0.3 ,1.6 ,6.7};
  if (!float_array_equality(output4,correct_results4,9,eps)) return 0;

  //test4
  float32_t output5[12] = {[0 ... 11] = 0};
  float32_t correct_results5[12] = {0, 0, 0, 8.7 ,19.3 ,1.9 ,2.9 ,10.5 ,2.3 ,0.3 ,1.6 ,6.7};
  convolution_additive(in4+4, input_side, output5+3,  weights4, weights_side);
  if (!float_array_equality(output5,correct_results5,12,eps)) return 0;

  //test5 (for all inputs, we can go as we please in one dimension, but width has to be the same!!!)
  float32_t weights5[10] = {0.35, 4.1,
                           5,   7,
                           0.5, 0.1,
                           2,   0,
                           -2,   99};
  float32_t output6[12] = {[0 ... 11] = 0};
  convolution_additive(in4+4, input_side, output6+3,  weights5+4, weights_side);
  if (!float_array_equality(output6,correct_results5,12,eps)) return 0;
  return 1;
}

uint8_t convolution_optimized_tests(){
  uint32_t input_side = 28;
  uint32_t weights_side = 5;
  float32_t in[784] = {[0 ... 783] = 5};
  float32_t output[1000] = {[0 ... 999] = 0};
  float32_t weights[25] = {[0 ... 24] = 1};
  float32_t bias = 0;

  convolution_optimized(in, input_side, output,  weights, weights_side);

  float32_t eps = 0.0001;
  float32_t out = 125;
  for(uint32_t i=0; i<576; i++){
    if (!float_equality(output[i],out,eps)) return 0;
  }

  //second
  input_side = 4;
  weights_side = 2;
  float32_t in2[16] = {1, 2, 3, 4,
                       4, 9, 0, 3,
                       0, 3, 1, 22,
                       0, 0, 2, 13};

  float32_t weights2[6] = {0, 2, 0, 0, //neccesary weights extension
                           0.1, 0.5};
  float32_t output3[229] = {[0 ... 228] = 0};
  convolution_optimized(in2, input_side, output3,  weights2, weights_side);
  float32_t correct_results2[9] = {8.7 ,19.3 ,1.9 ,2.9 ,10.5 ,2.3 ,0.3 ,1.6 ,6.7};
  if (!float_array_equality(output3,correct_results2,9,eps)) return 0;

  return 1;
}


uint8_t soft_max_tests(){
  float32_t in[7] = {1, 2, 3, 4, 1, 2, 3};
  float32_t output[7];
  soft_max(in, 7, output);

  float32_t eps = 0.01;

  float32_t correct_results[7] ={0.024 ,0.064 ,0.175 ,0.475 ,0.024 ,0.064 ,0.175};
  if (!float_array_equality(output,correct_results,7,eps)) return 0;

  return 1;
}

uint8_t ReLU_tests(){
  float32_t a1 = 0.880797077;
  float32_t a2 = 0.0001;
  float32_t a3 = -0.0001;
  float32_t a4 = -333.0001;
  if (ReLU(a1) != a1) return 0;
  if (ReLU(a2) != a2) return 0;
  if (ReLU(a3) != 0) return 0;
  if (ReLU(a4) != 0) return 0;

  return 1;
}

uint8_t most_probable_tests(){
  float32_t a[10] = {0.880797077, 0.8, 0.1, 0.2, 0.3, 0.5, 0.1, 0.9, 0.333, 0.3131};
  if (index_of_most_probable(a) != 7) return 0;

  float32_t a2[10] = {0.880797077, 0.8809, 0.1, 0.2, 0.3, 0.5231, 0.10001, 0.00003, 0.333, 0.3131};
  if (index_of_most_probable(a2) != 1) return 0;
  return 1;
}

uint8_t dot_product_tests(){
  float32_t a1[6] = {1,2,3,4,5,6};
  float32_t a2[6] = {4,3,2,9,0,0};
  float32_t a3[10] = {1,3,2,9,0,0,3,2,1,9};
  float32_t a4[36] = {4,3,2,9,0,0,
                      1,3,4,6,3,2, 
                      1,3,4,6,3,2, 
                      1,3,4,6,3,2, 
                      1,3,4,6,3,2, 
                      1,3,4,6,3,2};
  if (dot_product(a1, a1,(uint64_t) 6) != 91) return 0;
  if (dot_product(a2, a1, (uint64_t)6) != 52) return 0;
  if (dot_product(a3+4, a1, (uint64_t)6) != 76) return 0;
  if (dot_product_with_nth_column(a3+4, a4, 6, 6) != 15) return 0;
  if (dot_product_with_nth_column(a3+4, a4+3, 6, 6) != 90) return 0;
  if (dot_product_with_nth_column(a3+5, a4+(2*6)+1, 2, 6) != 9) return 0;

  return 1;
}

uint8_t classifier_test(uint32_t (*classify)(const float32_t *)){
  uint8_t counter = 0;

  if (classify(num1) == 3) counter++;
  if (classify(num2) == 9) counter++;
  if (classify(num3) == 7) counter++;
  if (classify(num4) == 6) counter++;
  if (classify(num5) == 1) counter++;
  if (classify(num6) == 2) counter++;
  if (classify(num8) == 4) counter++;
  if (classify(num10) == 3) counter++;
  if (classify(num11) == 5) counter++;
  if (classify(num12) == 3) counter++;
  if (classify(num13) == 6) counter++;
  if (classify(num14) == 1) counter++;
  if (classify(num15) == 7) counter++;
  if (classify(num16) == 2) counter++;
  if (classify(num17) == 8) counter++;
  if (classify(num18) == 6) counter++;
  if (classify(num19) == 9) counter++;
  if (classify(num20) == 4) counter++;
  if (classify(num21) == 0) counter++;
  if (classify(num22) == 9) counter++;
  if (classify(num23) == 1) counter++;
  if (classify(num24) == 1) counter++;
  if (classify(num25) == 2) counter++;
  if (classify(num26) == 4) counter++;
  if (classify(num27) == 3) counter++;
  if (classify(num28) == 2) counter++;
  if (classify(num29) == 7) counter++;
  if (classify(test_num0) == 7) counter++;
  if (classify(test_num1) == 2) counter++;
  if (classify(test_num2) == 1) counter++;
  if (classify(test_num3) == 0) counter++;
  if (classify(test_num4) == 4) counter++;
  if (classify(test_num5) == 1) counter++;
  if (classify(test_num6) == 4) counter++;
  if (classify(test_num7) == 9) counter++;
  if (classify(test_num8) == 5) counter++;

  return counter;
}

uint8_t classifier_test_q7_t(uint32_t (*classify)(const float32_t *)){
  uint8_t counter = 0;
  q7_t letter_q7_t[784] = {[0 ... 783] = 0};

  arm_float_to_q7(num1, letter_q7_t, 784);

  if (classify(letter_q7_t) == 3) counter++;
  arm_float_to_q7(num2, letter_q7_t, 784);
  if (classify(letter_q7_t) == 9) counter++;
  arm_float_to_q7(num3, letter_q7_t, 784);
  if (classify(letter_q7_t) == 7) counter++;
  arm_float_to_q7(num4, letter_q7_t, 784);
  if (classify(letter_q7_t) == 6) counter++;
  arm_float_to_q7(num5, letter_q7_t, 784);
  if (classify(letter_q7_t) == 1) counter++;
  arm_float_to_q7(num6, letter_q7_t, 784);
  if (classify(letter_q7_t) == 2) counter++;
  arm_float_to_q7(num8, letter_q7_t, 784);
  if (classify(letter_q7_t) == 4) counter++;
  arm_float_to_q7(num10, letter_q7_t, 784);
  if (classify(letter_q7_t) == 3) counter++;
  arm_float_to_q7(num11, letter_q7_t, 784);
  if (classify(letter_q7_t) == 5) counter++;
  arm_float_to_q7(num12, letter_q7_t, 784);
  if (classify(letter_q7_t) == 3) counter++;
  arm_float_to_q7(num13, letter_q7_t, 784);
  if (classify(letter_q7_t) == 6) counter++;
  arm_float_to_q7(num14, letter_q7_t, 784);
  if (classify(letter_q7_t) == 1) counter++;
  arm_float_to_q7(num15, letter_q7_t, 784);
  if (classify(letter_q7_t) == 7) counter++;
  arm_float_to_q7(num16, letter_q7_t, 784);
  if (classify(letter_q7_t) == 2) counter++;
  arm_float_to_q7(num17, letter_q7_t, 784);
  if (classify(letter_q7_t) == 8) counter++;
  arm_float_to_q7(num18, letter_q7_t, 784);
  if (classify(letter_q7_t) == 6) counter++;
  arm_float_to_q7(num19, letter_q7_t, 784);
  if (classify(letter_q7_t) == 9) counter++;
  arm_float_to_q7(num20, letter_q7_t, 784);
  if (classify(letter_q7_t) == 4) counter++;
  arm_float_to_q7(num21, letter_q7_t, 784);
  if (classify(letter_q7_t) == 0) counter++;
  arm_float_to_q7(num22, letter_q7_t, 784);
  if (classify(letter_q7_t) == 9) counter++;
  arm_float_to_q7(num23, letter_q7_t, 784);
  if (classify(letter_q7_t) == 1) counter++;
  arm_float_to_q7(num24, letter_q7_t, 784);
  if (classify(letter_q7_t) == 1) counter++;
  arm_float_to_q7(num25, letter_q7_t, 784);
  if (classify(letter_q7_t) == 2) counter++;
  arm_float_to_q7(num26, letter_q7_t, 784);
  if (classify(letter_q7_t) == 4) counter++;
  arm_float_to_q7(num27, letter_q7_t, 784);
  if (classify(letter_q7_t) == 3) counter++;
  arm_float_to_q7(num28, letter_q7_t, 784);
  if (classify(letter_q7_t) == 2) counter++;
  arm_float_to_q7(num29, letter_q7_t, 784);
  if (classify(letter_q7_t) == 7) counter++;

  arm_float_to_q7(test_num0, letter_q7_t, 784);
  if (classify(letter_q7_t) == 7) counter++;

  arm_float_to_q7(test_num1, letter_q7_t, 784);
  if (classify(letter_q7_t) == 2) counter++;

  arm_float_to_q7(test_num2, letter_q7_t, 784);
  if (classify(letter_q7_t) == 1) counter++;

  arm_float_to_q7(test_num3, letter_q7_t, 784);
  if (classify(letter_q7_t) == 0) counter++;

  arm_float_to_q7(test_num4, letter_q7_t, 784);
  if (classify(letter_q7_t) == 4) counter++;

  arm_float_to_q7(test_num5, letter_q7_t, 784);
  if (classify(letter_q7_t) == 1) counter++;

  arm_float_to_q7(test_num6, letter_q7_t, 784);
  if (classify(letter_q7_t) == 4) counter++;

  arm_float_to_q7(test_num7, letter_q7_t, 784);
  if (classify(letter_q7_t) == 9) counter++;

  arm_float_to_q7(test_num8, letter_q7_t, 784);
  if (classify(letter_q7_t) == 5) counter++;

  return counter;
}

uint8_t classifier_test_q9_t(uint32_t (*classify)(const float32_t *)){
  uint8_t counter = 0;
  q15_t letter_q9_t[784] = {[0 ... 783] = 0};

  arm_float_to_q9(num1, letter_q9_t, 784);

  if (classify(letter_q9_t) == 3) counter++;
  arm_float_to_q9(num2, letter_q9_t, 784);
  if (classify(letter_q9_t) == 9) counter++;
  arm_float_to_q9(num3, letter_q9_t, 784);
  if (classify(letter_q9_t) == 7) counter++;
  arm_float_to_q9(num4, letter_q9_t, 784);
  if (classify(letter_q9_t) == 6) counter++;
  arm_float_to_q9(num5, letter_q9_t, 784);
  if (classify(letter_q9_t) == 1) counter++;
  arm_float_to_q9(num6, letter_q9_t, 784);
  if (classify(letter_q9_t) == 2) counter++;
  arm_float_to_q9(num8, letter_q9_t, 784);
  if (classify(letter_q9_t) == 4) counter++;
  arm_float_to_q9(num10, letter_q9_t, 784);
  if (classify(letter_q9_t) == 3) counter++;
  arm_float_to_q9(num11, letter_q9_t, 784);
  if (classify(letter_q9_t) == 5) counter++;
  arm_float_to_q9(num12, letter_q9_t, 784);
  if (classify(letter_q9_t) == 3) counter++;
  arm_float_to_q9(num13, letter_q9_t, 784);
  if (classify(letter_q9_t) == 6) counter++;
  arm_float_to_q9(num14, letter_q9_t, 784);
  if (classify(letter_q9_t) == 1) counter++;
  arm_float_to_q9(num15, letter_q9_t, 784);
  if (classify(letter_q9_t) == 7) counter++;
  arm_float_to_q9(num16, letter_q9_t, 784);
  if (classify(letter_q9_t) == 2) counter++;
  arm_float_to_q9(num17, letter_q9_t, 784);
  if (classify(letter_q9_t) == 8) counter++;
  arm_float_to_q9(num18, letter_q9_t, 784);
  if (classify(letter_q9_t) == 6) counter++;
  arm_float_to_q9(num19, letter_q9_t, 784);
  if (classify(letter_q9_t) == 9) counter++;
  arm_float_to_q9(num20, letter_q9_t, 784);
  if (classify(letter_q9_t) == 4) counter++;
  arm_float_to_q9(num21, letter_q9_t, 784);
  if (classify(letter_q9_t) == 0) counter++;
  arm_float_to_q9(num22, letter_q9_t, 784);
  if (classify(letter_q9_t) == 9) counter++;
  arm_float_to_q9(num23, letter_q9_t, 784);
  if (classify(letter_q9_t) == 1) counter++;
  arm_float_to_q9(num24, letter_q9_t, 784);
  if (classify(letter_q9_t) == 1) counter++;
  arm_float_to_q9(num25, letter_q9_t, 784);
  if (classify(letter_q9_t) == 2) counter++;
  arm_float_to_q9(num26, letter_q9_t, 784);
  if (classify(letter_q9_t) == 4) counter++;
  arm_float_to_q9(num27, letter_q9_t, 784);
  if (classify(letter_q9_t) == 3) counter++;
  arm_float_to_q9(num28, letter_q9_t, 784);
  if (classify(letter_q9_t) == 2) counter++;
  arm_float_to_q9(num29, letter_q9_t, 784);
  if (classify(letter_q9_t) == 7) counter++;

  arm_float_to_q9(test_num0, letter_q9_t, 784);
  if (classify(letter_q9_t) == 7) counter++;

  arm_float_to_q9(test_num1, letter_q9_t, 784);
  if (classify(letter_q9_t) == 2) counter++;

  arm_float_to_q9(test_num2, letter_q9_t, 784);
  if (classify(letter_q9_t) == 1) counter++;

  arm_float_to_q9(test_num3, letter_q9_t, 784);
  if (classify(letter_q9_t) == 0) counter++;

  arm_float_to_q9(test_num4, letter_q9_t, 784);
  if (classify(letter_q9_t) == 4) counter++;

  arm_float_to_q9(test_num5, letter_q9_t, 784);
  if (classify(letter_q9_t) == 1) counter++;

  arm_float_to_q9(test_num6, letter_q9_t, 784);
  if (classify(letter_q9_t) == 4) counter++;

  arm_float_to_q9(test_num7, letter_q9_t, 784);
  if (classify(letter_q9_t) == 9) counter++;

  arm_float_to_q9(test_num8, letter_q9_t, 784);
  if (classify(letter_q9_t) == 5) counter++;

  return counter;
}
