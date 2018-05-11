

#include "conv.h"

float32_t max(float32_t a[], float32_t n){
  float32_t max = a[0];

  for(uint32_t i = 1; i<n; i++){
    if (a[i] > max) max = a[i];
  }

  return max;
}

float32_t dot_product(const float32_t w[], const float32_t a[], uint64_t vector_size) {
  float32_t result = 0.0f;
  for (uint64_t i = 0; i < vector_size; i++){
    result += (float32_t)(w[i]*a[i]);
  }
  return result;
}

q15_t dot_product_q15(q15_t w[], q15_t a[], uint64_t vector_size) {
  q15_t result = 0;
  q15_t tmp = 0;
  for (uint64_t i = 0; i < vector_size; i++){
    tmp = (q15_t) __SSAT((((q31_t) w[i] * a[i]) >> 9 ), 16);
    result = (q15_t) __SSAT((q31_t) (result + tmp), 16);
    //result += (float32_t)(w[i]*a[i]);
  }
  return result;
}

float32_t dot_product_with_nth_column(const float32_t a[], const float32_t b[], uint32_t vector_size, uint32_t n) {
  float32_t result = 0.0f;
  for (uint32_t i = 0; i < vector_size; i++){
    result += b[i*n] * a[i];
  }
  return result;
}

q15_t dot_product_with_nth_column_q15(const q15_t a[], const q15_t b[], uint32_t vector_size, uint32_t n) {
  q15_t result = 0;
  q15_t tmp = 0;
  for (uint32_t i = 0; i < vector_size; i++){
    result += b[i*n] * a[i];
    tmp = (q15_t) __SSAT((((q31_t) (b[i*n]) * (a[i])) >> 9 ), 16);
    result = (q15_t) __SSAT((q31_t) (result + tmp), 16);
  }
  return result;
}


uint32_t coords(uint32_t x, uint32_t y, uint32_t side){
  return((x*side)+y);
}

//for 2x2 region
void pooling(float32_t in[], float32_t out[], uint32_t side, float32_t(*pooling_function)(float32_t*, float32_t)){
  /*          j
   *   ----------------
   *  | x x x x x x x x       
   *  | x x x x x x x x       
   *  | x x a b x x x x       x x x x 
   * i| x x c d x x x x  -->  x a x x 
   *  | x x x x x x x x       x x x x 
   *  | x x x x x x x x       x x x x 
   *  | x x x x x x x x        
   *  | x x x x x x x x        
   */

  float32_t region[4];
  uint32_t output_side = side/2;

  for(uint32_t i = 0; i<side; i=i+2){
    for(uint32_t j = 0; j<side; j=j+2){
      region[0] = in[coords(i,j,side)];
      region[1] = in[coords(i+1,j,side)];
      region[2] = in[coords(i,j+1,side)];
      region[3] = in[coords(i+1,j+1,side)];

      out[coords(i/2,j/2,output_side)] = pooling_function(region, 4);

    }
  }
}

void pooling_optimized(float32_t in[], float32_t out[], uint32_t side, void(*pooling_function)(float32_t*, uint32_t, float32_t*, uint32_t*)){
  /*          j
   *   ----------------
   *  | x x x x x x x x       
   *  | x x x x x x x x       
   *  | x x a b x x x x       x x x x 
   * i| x x c d x x x x  -->  x a x x 
   *  | x x x x x x x x       x x x x 
   *  | x x x x x x x x       x x x x 
   *  | x x x x x x x x        
   *  | x x x x x x x x        
   */

  float32_t region[4];
  uint32_t output_side = side/2;
  float32_t result;
  uint32_t result_pos;

  for(uint32_t i = 0; i<side; i=i+2){
    for(uint32_t j = 0; j<side; j=j+2){
      region[0] = in[coords(i,j,side)];
      region[1] = in[coords(i+1,j,side)];
      region[2] = in[coords(i,j+1,side)];
      region[3] = in[coords(i+1,j+1,side)];

      pooling_function(region, 4, &result, &result_pos);
      out[coords(i/2,j/2,output_side)] = result;
    }
  }
}

void pooling_optimized_q7_t(q7_t in[], q7_t out[], uint32_t side, void(*pooling_function)(q7_t*, uint32_t, q7_t*, uint32_t*)){
  q7_t region[4];
  uint32_t output_side = side/2;
  q7_t result;
  uint32_t result_pos;

  for(uint32_t i = 0; i<side; i=i+2){
    for(uint32_t j = 0; j<side; j=j+2){
      region[0] = in[coords(i,j,side)];
      region[1] = in[coords(i+1,j,side)];
      region[2] = in[coords(i,j+1,side)];
      region[3] = in[coords(i+1,j+1,side)];

      pooling_function(region, 4, &result, &result_pos);
      out[coords(i/2,j/2,output_side)] = result;
    }
  }
}

void pooling_optimized_q9_t(q15_t in[], q15_t out[], uint32_t side, void(*pooling_function)(q15_t*, uint32_t, q15_t*, uint32_t*)){
  q15_t region[4];
  uint32_t output_side = side/2;
  q15_t result;
  uint32_t result_pos;

  for(uint32_t i = 0; i<side; i=i+2){
    for(uint32_t j = 0; j<side; j=j+2){
      region[0] = in[coords(i,j,side)];
      region[1] = in[coords(i+1,j,side)];
      region[2] = in[coords(i,j+1,side)];
      region[3] = in[coords(i+1,j+1,side)];

      pooling_function(region, 4, &result, &result_pos);
      out[coords(i/2,j/2,output_side)] = result;
    }
  }
}

void convolution_with_activation(const float32_t in[], uint32_t input_side, float32_t out[], const float32_t weights[], uint32_t weights_side, float32_t bias, float32_t(*activation_fn)(float32_t)){
  /*          j
   *   ----------------
   *  | x a b c x x x x       x z x x x x x
   *  | x d e f x x x x       x x x x x x x
   *  | x g h i x x x x       x x x x x x x
   * i| x x x x x x x x  -->  x x x x x x x
   *  | x x x x x x x x       x x x x x x x
   *  | x x x x x x x x       x x x x x x x
   *  | x x x x x x x x       x x x x x x x
   *  | x x x x x x x x       
   */
  //weights_side == kernel size
  //weights + bias == kernel --> shared among all local receptive fields

  uint32_t stride = 1;
  float32_t res;
  uint32_t output_side = (input_side/stride) - (weights_side - 1);


  for(uint32_t i = 0; i<(input_side-(weights_side-1)); i=i+stride){
    for(uint32_t j = 0; j<(input_side-(weights_side-1)); j=j+stride){
      res = 0;

      for(uint32_t sub_i = 0; sub_i<weights_side; sub_i++){
        for(uint32_t sub_j = 0; sub_j<weights_side; sub_j++){
          res += weights[coords(sub_i, sub_j, weights_side)] * in[coords(i+sub_i, j+sub_j, input_side)];
        }
      }

      res += bias;
      out[coords(i,j,output_side)] = (float32_t) activation_fn(res);
    }
  }
}

void convolution_additive_q7_t(q7_t in[], uint32_t input_side, q7_t out[], q7_t weights[], uint32_t weights_side){
  uint32_t stride = 1;
  q7_t res;
  uint32_t output_side = (input_side/stride) - (weights_side - 1);


  for(uint32_t i = 0; i<(input_side-(weights_side-1)); i=i+stride){
    for(uint32_t j = 0; j<(input_side-(weights_side-1)); j=j+stride){
      res = 0;

      for(uint32_t sub_i = 0; sub_i<weights_side; sub_i++){
        for(uint32_t sub_j = 0; sub_j<weights_side; sub_j++){
          res = (q7_t) __SSAT(res + (q7_t) __SSAT((((q15_t) (weights[coords(sub_i, sub_j, weights_side)]) * (in[coords(i+sub_i, j+sub_j, input_side)])) >> 7 ), 8), 8);
        }
      }
      out[coords(i,j,output_side)] = (q7_t) __SSAT((res + out[coords(i,j,output_side)]), 8);
    }
  } 
}

void convolution_additive_q9_t(q15_t in[], uint32_t input_side, q15_t out[], q15_t weights[], uint32_t weights_side){
  uint32_t stride = 1;
  q15_t tmp;
  q15_t res;
  uint32_t output_side = (input_side/stride) - (weights_side - 1);


  for(uint32_t i = 0; i<(input_side-(weights_side-1)); i=i+stride){
    for(uint32_t j = 0; j<(input_side-(weights_side-1)); j=j+stride){
      res = 0;

      for(uint32_t sub_i = 0; sub_i<weights_side; sub_i++){
        for(uint32_t sub_j = 0; sub_j<weights_side; sub_j++){
          tmp = (q15_t) __SSAT((((q31_t) (weights[coords(sub_i, sub_j, weights_side)]) * (in[coords(i+sub_i, j+sub_j, input_side)])) >> 9 ), 16);
          res = (q15_t) __SSAT((q31_t) (res + tmp), 16);
        }
      }
      out[coords(i,j,output_side)] = (q15_t) __SSAT((q31_t)(res + out[coords(i,j,output_side)]), 16);
    }
  } 
}


void convolution_additive(const float32_t in[], uint32_t input_side, float32_t out[], const float32_t weights[], uint32_t weights_side){
  /*          j
   *   ----------------
   *  | x a b c x x x x       x z x x x x x
   *  | x d e f x x x x       x x x x x x x
   *  | x g h i x x x x       x x x x x x x
   * i| x x x x x x x x  -->  x x x x x x x
   *  | x x x x x x x x       x x x x x x x
   *  | x x x x x x x x       x x x x x x x
   *  | x x x x x x x x       x x x x x x x
   *  | x x x x x x x x       
   */
  //weights_side == kernel size
  //weights + bias == kernel --> shared among all local receptive fields

  uint32_t stride = 1;
  float32_t res;
  uint32_t output_side = (input_side/stride) - (weights_side - 1);


  for(uint32_t i = 0; i<(input_side-(weights_side-1)); i=i+stride){
    for(uint32_t j = 0; j<(input_side-(weights_side-1)); j=j+stride){
      res = 0;

      for(uint32_t sub_i = 0; sub_i<weights_side; sub_i++){
        for(uint32_t sub_j = 0; sub_j<weights_side; sub_j++){
          res += weights[coords(sub_i, sub_j, weights_side)] * in[coords(i+sub_i, j+sub_j, input_side)];
        }
      }

      out[coords(i,j,output_side)] += res;
    }
  }

}

void convolution_optimized(const float32_t in[], uint32_t input_side, float32_t out[], const float32_t weights[], uint32_t weights_side){
  uint32_t stride = 1;
  uint32_t output_side = (input_side/stride) - (weights_side - 1);
  uint32_t output_size = (output_side)*(output_side+weights_side-1);
  uint32_t weights_size = weights_side * weights_side + ((input_side-weights_side)*(weights_side-1));

  //arm_conv_f32(in, input_side*input_side, weights, weights_size, conv_out);
  uint32_t start = weights_size-1;
  //arm_conv_partial_f32(in, input_side*input_side, weights, weights_size, conv_out, start ,output_size);
  arm_conv_partial_f32(in, input_side*input_side, weights, weights_size, out, start ,output_size);

  for(uint32_t i = 0; i<output_side; i++){
    arm_copy_f32(out+(i*(input_side))+(weights_size-1), out+(i*(output_side)), output_side);
  }
}

void convolution_optimized_one_go(const float32_t in[], uint32_t input_side, float32_t out[], const float32_t weights[], uint32_t weights_size){
  uint32_t stride = 1;
  uint32_t output_side = (input_side/stride) - (5 - 1);
  uint32_t output_size = (output_side)*(output_side+5-1);

  //TODO here could be alloc and then free, yet I dont know how, so far
  //float32_t conv_out[28*28+116] = {[0 ... 899] = 0};
  //float32_t conv_out[1001] = calloc(10 * sizeof(float32_t), 0);

  arm_conv_f32(in, input_side*input_side, weights, weights_size, out);
  //arm_conv_partial_f32(in, input_side*input_side, weights, weights_size, conv_out, start ,output_size);
  //arm_conv_partial_f32(in, input_side*input_side, weights, weights_size, out, start ,output_size);

  for(uint32_t i = 0; i<output_side; i++){
    arm_copy_f32(out+(i*(input_side))+(weights_size-1), out+(i*(output_side)), output_side);
  }
}

void convolution_additive_optimized(const float32_t in[], uint32_t input_side, float32_t out[], const float32_t weights[], uint32_t weights_side){
  uint32_t stride = 1;
  uint32_t output_side = (input_side/stride) - (weights_side - 1);
  uint32_t output_size = (output_side)*(output_side)+(16); //only for filter of size 5x5
  float32_t conv_out[201] = {[0 ... 200] = 0};

  arm_conv_partial_f32(in, input_side, weights, weights_side, conv_out, 5*5+4*23-1, output_size);
  arm_copy_f32(conv_out+23, conv_out+27, 24);
  arm_copy_f32(conv_out+46, conv_out+54, 24);
  arm_copy_f32(conv_out+69, conv_out+81, 24);
  arm_copy_f32(conv_out+92, conv_out+108, 24);

  arm_matrix_instance_f32 a = {output_side, output_side, conv_out};
  arm_matrix_instance_f32 b = {output_side, output_side, out};
  arm_matrix_instance_f32 c = {output_side, output_side, out};
  arm_mat_add_f32(&a, &b, &c);
  arm_copy_f32(c.pData, out, output_side*output_side);
}
