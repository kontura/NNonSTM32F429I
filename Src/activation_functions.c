
#include "activation_functions.h"

//TAYLOR SEries
float32_t exponential(uint32_t n, float32_t x) {
  float32_t exp = 1.0f; // initialize sum of series
  float32_t new_x = x;

  if (x==0) return 1;
  if (x<0) new_x = -x;

  for (uint32_t i = n - 1; i > 0; --i )
    exp = 1 + new_x * exp / i;

  if(x<0) return 1/exp;

  return exp;
}

float32_t sigmoid(float32_t z){
  return 1.0/(1.0+exponential(100, -z));
}

float32_t ReLU(float32_t z){
  if (z > 0)
    return z;
  else
    return 0;
}

void soft_max(float32_t* input, uint32_t input_size, float32_t* output){
  // a = e^(zi)/SUMi(e^zi)
  float32_t suma=0;
  for(uint32_t i=0; i<input_size; i++){
    suma += exponential(100, input[i]);
  }
  if (isinf(suma)) suma = FLOAT32_T_MAX;

  for(uint32_t i=0; i<input_size; i++){
    output[i] = exponential(100, input[i])/suma;
    //if (output[i] != output[i]) output[i] = 1; // check for NaN
  }

}
