

#include "utility.h"


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
