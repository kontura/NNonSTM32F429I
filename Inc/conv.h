
#ifndef __conv_H
#define __conv_H

#include "stm32f429i_discovery.h"
#include "arm_math.h"
#include "activation_functions.h"

//float32_t (*pooling_function)(float32_t, float32_t, float32_t, float32_t);

float32_t max(float32_t a[], float32_t n);
void pooling(float32_t in[], float32_t out[], uint32_t side, float32_t(*pooling_function)(float32_t*, float32_t));
uint32_t coords(uint32_t x, uint32_t y, uint32_t side);
void convolution_with_activation(const float32_t in[], uint32_t input_side, float32_t out[], const float32_t weights[], uint32_t weights_side, float32_t bias, float32_t(*activation_fn)(float32_t));
void convolution_additive(const float32_t in[], uint32_t input_side, float32_t out[], const float32_t weights[], uint32_t weights_side);
void convolution_additive_optimized(const float32_t in[], uint32_t input_side, float32_t out[], const float32_t weights[], uint32_t weights_side);
void convolution_optimized(const float32_t in[], uint32_t input_side, float32_t out[], const float32_t weights[], uint32_t weights_side);
float32_t dot_product(const float32_t w[], const float32_t a[], uint64_t vector_size);
float32_t dot_product_with_nth_column(const float32_t w[], const float32_t a[], uint32_t vector_size, uint32_t n);

void convolution_optimized_one_go(const float32_t in[], uint32_t input_side, float32_t out[], const float32_t weights[], uint32_t weights_size);

/*
float32_t feed_forward(const float32_t* a, const sourceLayer* b, const targetLayer* w);
float32_t feed_forward_basic_net(const float32_t* a, const basicNetB* b, const basicNetW* w);
*/

#endif /* __conv_H */
