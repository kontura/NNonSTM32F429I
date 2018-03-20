
#ifndef __TEST_H
#define __TEST_H

/* Includes ------------------------------------------------------------------*/
#include "stm32f429i_discovery.h"
#include "arm_math.h"
#include "conv.h"
#include "utility.h"
#include "activation_functions.h"


uint8_t test();
uint8_t pooling_tests();
uint8_t sigmoid_tests();
uint8_t convolution_tests();
uint8_t convolution_optimized_tests();
uint8_t soft_max_tests();
uint8_t ReLU_tests();
uint8_t dot_product_tests();
uint8_t float_equality(float32_t a, float32_t b, float32_t eps);
uint8_t float_array_equality(float32_t* a, float32_t* b, uint8_t size, float32_t eps);
uint8_t most_probable_tests();
uint8_t classifier_test(uint32_t (*classify)(const float32_t *));

#endif /* __TEST_H */
