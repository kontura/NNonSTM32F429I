#ifndef __utility_H
#define __utility_H

#include "stm32f429i_discovery.h"
#include "arm_math.h"
#include <math.h>

uint32_t index_of_most_probable(float32_t probabilities[10]);
void arm_fn_f32( float32_t * pSrc, float32_t * pDst, uint32_t blockSize, float32_t(*fn)(float32_t));

#endif /* __utility_H */
