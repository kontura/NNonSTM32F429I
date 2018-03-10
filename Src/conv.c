
#include "conv.h"

float32_t max(float32_t a[], float32_t n){
  float32_t max = a[0];

  for(uint32_t i = 1; i<n; i++){
    if (a[i] > max) max = a[i];
  }

  return max;
}

float32_t dot_product(const float32_t w[], const float32_t a[], uint32_t vector_size) {
  float32_t result = 0.0f;
  for (uint32_t i = 0; i < vector_size; i++)
    result += w[i]*a[i];
  return result;
}

float32_t dot_product_with_nth_column(const float32_t a[], const float32_t b[], uint32_t vector_size, uint32_t n) {
  float32_t result = 0.0f;
  for (uint32_t i = 0; i < vector_size; i++){
    result += b[i*n] * a[i];
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
