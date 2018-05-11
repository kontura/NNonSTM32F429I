
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

q7_t ReLU_q7(q7_t z){
  if (z > 0)
    return z;
  else
    return 0;
}

q15_t ReLU_q15(q15_t z){
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
void soft_max_q15(q15_t* input, uint32_t input_size, q15_t* output){
  // a = e^(zi)/SUMi(e^zi)
  float32_t input_f[100] = {[0 ... 100-1] = 0};
  float32_t output_f[100] = {[0 ... 100-1] = 0};
  arm_q15_to_float(input, input_f, input_size);
  float32_t suma=0;
  for(uint32_t i=0; i<input_size; i++){
    suma += exponential(100, input_f[i]);
  }
  if (isinf(suma)) suma = FLOAT32_T_MAX;

  for(uint32_t i=0; i<input_size; i++){
    output_f[i] = exponential(100, input_f[i])/suma;
  //if (output[i] != output[i]) output[i] = 1; // check for NaN
  }

  arm_float_to_q15(output_f, output, input_size);
}

void soft_max_q9(q15_t* input, uint32_t input_size, q15_t* output){
  // a = e^(zi)/SUMi(e^zi)
  float32_t input_f[100] = {[0 ... 100-1] = 0};
  float32_t output_f[100] = {[0 ... 100-1] = 0};
  arm_q9_to_float(input, input_f, input_size);
  float32_t suma=0;
  for(uint32_t i=0; i<input_size; i++){
    suma += exponential(100, input_f[i]);
  }
  if (isinf(suma)) suma = FLOAT32_T_MAX;

  for(uint32_t i=0; i<input_size; i++){
    output_f[i] = exponential(100, input_f[i])/suma;
  //if (output[i] != output[i]) output[i] = 1; // check for NaN
  }

  arm_float_to_q9(output_f, output, input_size);
}

void soft_max_q7(q7_t* input, uint32_t input_size, q7_t* output){
  // a = e^(zi)/SUMi(e^zi)
  float32_t input_f[100] = {[0 ... 100-1] = 0};
  float32_t output_f[100] = {[0 ... 100-1] = 0};
  arm_q7_to_float(input, input_f, input_size);
  float32_t suma=0;
  for(uint32_t i=0; i<input_size; i++){
    suma += exponential(100, input_f[i]);
  }
  if (isinf(suma)) suma = FLOAT32_T_MAX;

  for(uint32_t i=0; i<input_size; i++){
    output_f[i] = exponential(100, input_f[i])/suma;
  //if (output[i] != output[i]) output[i] = 1; // check for NaN
  }

  arm_float_to_q7(output_f, output, input_size);
}
