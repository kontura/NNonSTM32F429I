

#ifndef __activations_fce_H
#define __activations_fce_H

#include "stm32f429i_discovery.h"
#include "arm_math.h"

#define FLOAT32_T_MAX 1.7014116317805962808001687976863e38

float32_t exponential(uint32_t n, float32_t x);
float32_t sigmoid(float32_t z);
float32_t ReLU(float32_t z);
q7_t ReLU_q7(q7_t z);
q15_t ReLU_q15(q15_t z);
void soft_max(float32_t* input, uint32_t input_size, float32_t* output);

#endif /* __activations_fce_H */
