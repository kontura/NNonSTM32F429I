
#ifndef __time_profiling_H
#define __time_profiling_H

#include "utility.h"

#define INF_LOOP_TOGGL_PIN(call) do{\
  call;\
  HAL_GPIO_TogglePin(GPIOC, GPIO_PIN_4);\
  BSP_LED_Toggle(LED4);\
}while(1)

uint64_t time_counter;
void time_profiling(TIM_HandleTypeDef TimHandle);
void profile_sigmoid(TIM_HandleTypeDef TimHandle, uint64_t *results);

#endif /* __time_profiling_H */
