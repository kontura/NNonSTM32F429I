#ifndef __utility_H
#define __utility_H

#include "stm32f429i_discovery.h"
#include "arm_math.h"
#include <math.h>
#include "activation_functions.h"

void time_profiling(TIM_HandleTypeDef TimHandle);
uint64_t time_counter;
uint32_t index_of_most_probable(float32_t probabilities[10]);
uint32_t index_of_most_probable15(q15_t probabilities[10]);
void arm_fn_f32( float32_t * pSrc, float32_t * pDst, uint32_t blockSize, float32_t(*fn)(float32_t));
void arm_fn_q7( q7_t * pSrc, q7_t * pDst, uint32_t blockSize, q7_t(*fn)(q7_t));
void arm_fn_q15( q15_t * pSrc, q15_t * pDst, uint32_t blockSize, q15_t(*fn)(q15_t));
void Error_Handler(void);
void start_time_measure(TIM_HandleTypeDef TimHandle);
uint64_t stop_time_measure(TIM_HandleTypeDef TimHandle);

#define INF_LOOP_TOGGL_PIN(call) do{\
  call;\
  HAL_GPIO_TogglePin(GPIOC, GPIO_PIN_4);\
  BSP_LED_Toggle(LED4);\
}while(1)


void arm_q9_to_float( q15_t * pSrc, float32_t * pDst, uint32_t blockSize);
#endif /* __utility_H */
