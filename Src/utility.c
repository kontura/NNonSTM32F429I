

#include "utility.h"

/**
  * @brief  This function is executed in case of error occurrence.
  * @param  None
  * @retval None
  */
void Error_Handler(void)
{
  /* Turn LED3 on */
  BSP_LED_On(LED3);
  while(1)
  {
  }
}

void start_time_measure(TIM_HandleTypeDef TimHandle){
  time_counter = 0;
  if(HAL_TIM_Base_Start_IT(&TimHandle) != HAL_OK)
  {
    /* Starting Error */
    Error_Handler();
  }
}

uint64_t stop_time_measure(TIM_HandleTypeDef TimHandle){
  if(HAL_TIM_Base_Stop_IT(&TimHandle) != HAL_OK)
  {
    /* Starting Error */
    Error_Handler();
  }
  return time_counter;
}

uint32_t index_of_most_probable_q7(q7_t probabilities[10]){
  uint32_t index_of_max = 0;
  for(uint32_t i=0; i<10;i++){
    if (probabilities[i] > probabilities[index_of_max]) index_of_max = i;
  }

  return index_of_max;
}

uint32_t index_of_most_probable_q15(q15_t probabilities[10]){
  uint32_t index_of_max = 0;
  for(uint32_t i=0; i<10;i++){
    if (probabilities[i] > probabilities[index_of_max]) index_of_max = i;
  }

  return index_of_max;
}

uint32_t index_of_most_probable(float32_t probabilities[10]){
  uint32_t index_of_max = 0;
  for(uint32_t i=0; i<10;i++){
    if (probabilities[i] > probabilities[index_of_max]) index_of_max = i;
  }

  return index_of_max;
}

void arm_fn_f32(
  float32_t * pSrc,
  float32_t * pDst,
  uint32_t blockSize,
  float32_t(*fn)(float32_t))
{
  uint32_t blkCnt;                               /* loop counter */

#ifndef ARM_MATH_CM0_FAMILY

  /* Run the below code for Cortex-M4 and Cortex-M3 */
  float32_t in1, in2, in3, in4;                  /* temporary variables */

  /*loop Unrolling */
  blkCnt = blockSize >> 2u;

  /* First part of the processing with loop unrolling.  Compute 4 outputs at a time.    
   ** a second loop below computes the remaining 1 to 3 samples. */
  while(blkCnt > 0u)
  {
    /* C = |A| */
    /* Calculate absolute and then store the results in the destination buffer. */
    /* read sample from source */
    in1 = *pSrc;
    in2 = *(pSrc + 1);
    in3 = *(pSrc + 2);

    /* find absolute value */
    in1 = fn(in1);

    /* read sample from source */
    in4 = *(pSrc + 3);

    /* find absolute value */
    in2 = fn(in2);

    /* read sample from source */
    *pDst = in1;

    /* find absolute value */
    in3 = fn(in3);

    /* find absolute value */
    in4 = fn(in4);

    /* store result to destination */
    *(pDst + 1) = in2;

    /* store result to destination */
    *(pDst + 2) = in3;

    /* store result to destination */
    *(pDst + 3) = in4;


    /* Update source pointer to process next sampels */
    pSrc += 4u;

    /* Update destination pointer to process next sampels */
    pDst += 4u;

    /* Decrement the loop counter */
    blkCnt--;
  }

  /* If the blockSize is not a multiple of 4, compute any remaining output samples here.    
   ** No loop unrolling is used. */
  blkCnt = blockSize % 0x4u;

#else

  /* Run the below code for Cortex-M0 */

  /* Initialize blkCnt with number of samples */
  blkCnt = blockSize;

#endif /*   #ifndef ARM_MATH_CM0_FAMILY   */

  while(blkCnt > 0u)
  {
    /* C = |A| */
    /* Calculate absolute and then store the results in the destination buffer. */
    *pDst++ = fn(*pSrc++);

    /* Decrement the loop counter */
    blkCnt--;
  }
}

void arm_fn_q7(
  q7_t * pSrc,
  q7_t * pDst,
  uint32_t blockSize,
  q7_t(*fn)(q7_t))
{
  uint32_t blkCnt;                               /* loop counter */
  q7_t in;                                       /* Input value1 */

#ifndef ARM_MATH_CM0_FAMILY

  /* Run the below code for Cortex-M4 and Cortex-M3 */
  q31_t in1, in2, in3, in4;                      /* temporary input variables */
  q31_t out1, out2, out3, out4;                  /* temporary output variables */

  /*loop Unrolling */
  blkCnt = blockSize >> 2u;

  /* First part of the processing with loop unrolling.  Compute 4 outputs at a time.    
   ** a second loop below computes the remaining 1 to 3 samples. */
  while(blkCnt > 0u)
  {
    /* C = |A| */
    /* Read inputs */
    in1 = (q31_t) * pSrc;
    in2 = (q31_t) * (pSrc + 1);
    in3 = (q31_t) * (pSrc + 2);

    /* find absolute value */
    out1 = fn(in1);

    /* read input */
    in4 = (q31_t) * (pSrc + 3);

    /* find absolute value */
    out2 = fn(in2);

    /* store result to destination */
    *pDst = (q7_t) out1;

    /* find absolute value */
    out3 = fn(in3);

    /* find absolute value */
    out4 = fn(in4);

    /* store result to destination */
    *(pDst + 1) = (q7_t) out2;

    /* store result to destination */
    *(pDst + 2) = (q7_t) out3;

    /* store result to destination */
    *(pDst + 3) = (q7_t) out4;

    /* update pointers to process next samples */
    pSrc += 4u;
    pDst += 4u;

    /* Decrement the loop counter */
    blkCnt--;
  }

  /* If the blockSize is not a multiple of 4, compute any remaining output samples here.    
   ** No loop unrolling is used. */
  blkCnt = blockSize % 0x4u;
#else

  /* Run the below code for Cortex-M0 */
  blkCnt = blockSize;

#endif /* #define ARM_MATH_CM0_FAMILY */

  while(blkCnt > 0u)
  {
    /* C = |A| */
    /* Read the input */
    in = *pSrc++;

    /* Store the Absolute result in the destination buffer */
    *pDst++ = fn(in);

    /* Decrement the loop counter */
    blkCnt--;
  }
}

void arm_fn_q15(
  q15_t * pSrc,
  q15_t * pDst,
  uint32_t blockSize,
  q15_t(*fn)(q15_t))
{
  uint32_t blkCnt;                               /* loop counter */
  q15_t in;                                       /* Input value1 */

#ifndef ARM_MATH_CM0_FAMILY

  /* Run the below code for Cortex-M4 and Cortex-M3 */
  q31_t in1, in2, in3, in4;                      /* temporary input variables */
  q31_t out1, out2, out3, out4;                  /* temporary output variables */

  /*loop Unrolling */
  blkCnt = blockSize >> 2u;

  /* First part of the processing with loop unrolling.  Compute 4 outputs at a time.    
   ** a second loop below computes the remaining 1 to 3 samples. */
  while(blkCnt > 0u)
  {
    /* C = |A| */
    /* Read inputs */
    in1 = (q31_t) * pSrc;
    in2 = (q31_t) * (pSrc + 1);
    in3 = (q31_t) * (pSrc + 2);

    /* find absolute value */
    out1 = fn(in1);

    /* read input */
    in4 = (q31_t) * (pSrc + 3);

    /* find absolute value */
    out2 = fn(in2);

    /* store result to destination */
    *pDst = (q7_t) out1;

    /* find absolute value */
    out3 = fn(in3);

    /* find absolute value */
    out4 = fn(in4);

    /* store result to destination */
    *(pDst + 1) = (q15_t) out2;

    /* store result to destination */
    *(pDst + 2) = (q15_t) out3;

    /* store result to destination */
    *(pDst + 3) = (q15_t) out4;

    /* update pointers to process next samples */
    pSrc += 4u;
    pDst += 4u;

    /* Decrement the loop counter */
    blkCnt--;
  }

  /* If the blockSize is not a multiple of 4, compute any remaining output samples here.    
   ** No loop unrolling is used. */
  blkCnt = blockSize % 0x4u;
#else

  /* Run the below code for Cortex-M0 */
  blkCnt = blockSize;

#endif /* #define ARM_MATH_CM0_FAMILY */

  while(blkCnt > 0u)
  {
    /* C = |A| */
    /* Read the input */
    in = *pSrc++;

    /* Store the Absolute result in the destination buffer */
    *pDst++ = fn(in);

    /* Decrement the loop counter */
    blkCnt--;
  }
}

void arm_add_q9(
  q15_t * pSrcA,
  q15_t * pSrcB,
  q15_t * pDst,
  uint32_t blockSize)
{
  uint32_t blkCnt;                               /* loop counter */

  /* Initialize blkCnt with number of samples */
  blkCnt = blockSize;

  while(blkCnt > 0u)
  {
    /* C = A + B */
    /* Add and then store the results in the destination buffer. */
    *pDst++ = (q15_t) __SSAT(((q31_t) * pSrcA++ + *pSrcB++), 9);

    /* Decrement the loop counter */
    blkCnt--;
  }
}

void arm_q9_to_float(
  q15_t * pSrc,
  float32_t * pDst,
  uint32_t blockSize)
{
  q15_t *pIn = pSrc;                             /* Src pointer */
  uint32_t blkCnt;                               /* loop counter */


#ifndef ARM_MATH_CM0_FAMILY

  /* Run the below code for Cortex-M4 and Cortex-M3 */

  /*loop Unrolling */
  blkCnt = blockSize >> 2u;

  /* First part of the processing with loop unrolling.  Compute 4 outputs at a time.    
   ** a second loop below computes the remaining 1 to 3 samples. */
  while(blkCnt > 0u)
  {
    /* C = (float32_t) A / 32768 */
    /* convert from q15 to float and then store the results in the destination buffer */
    *pDst++ = ((float32_t) * pIn++ / 512.0f);
    *pDst++ = ((float32_t) * pIn++ / 512.0f);
    *pDst++ = ((float32_t) * pIn++ / 512.0f);
    *pDst++ = ((float32_t) * pIn++ / 512.0f);

    /* Decrement the loop counter */
    blkCnt--;
  }

  /* If the blockSize is not a multiple of 4, compute any remaining output samples here.    
   ** No loop unrolling is used. */
  blkCnt = blockSize % 0x4u;

#else

  /* Run the below code for Cortex-M0 */

  /* Loop over blockSize number of values */
  blkCnt = blockSize;

#endif /* #ifndef ARM_MATH_CM0_FAMILY */

  while(blkCnt > 0u)
  {
    /* C = (float32_t) A / 32768 */
    /* convert from q15 to float and then store the results in the destination buffer */
    *pDst++ = ((float32_t) * pIn++ / 512.0f);

    /* Decrement the loop counter */
    blkCnt--;
  }
}
