
#ifndef __time_profiling_H
#define __time_profiling_H

#include "utility.h"

uint64_t time_counter;
void time_profiling(TIM_HandleTypeDef TimHandle);
void profile_sigmoid(TIM_HandleTypeDef TimHandle, uint64_t *results);

#endif /* __time_profiling_H */
