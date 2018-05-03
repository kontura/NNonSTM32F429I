#ifndef __utility_H
#define __utility_H

#include "stm32f429i_discovery.h"
#include "arm_math.h"
#include <math.h>
#include "activation_functions.h"

void time_profiling(TIM_HandleTypeDef TimHandle);
uint64_t time_counter;
uint32_t index_of_most_probable(float32_t probabilities[10]);
void arm_fn_f32( float32_t * pSrc, float32_t * pDst, uint32_t blockSize, float32_t(*fn)(float32_t));
void Error_Handler(void);
void start_time_measure(TIM_HandleTypeDef TimHandle);
uint64_t stop_time_measure(TIM_HandleTypeDef TimHandle);

#endif /* __utility_H */
