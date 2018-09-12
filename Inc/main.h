#ifndef __MAIN_H
#define __MAIN_H

/* Includes ------------------------------------------------------------------*/
#include "stm32f429i_discovery.h"
#include "arm_math.h"
#include "stm32f4xx_hal.h"

#include "conv.h"
#include "tests.h"
#include "utility.h"
#include "activation_functions.h"

#ifdef PROFILE
  #include "time_profiling.h"
#endif


uint32_t net_2layers(const float32_t* letter);
uint32_t net_3layers(const float32_t* letter);
uint32_t net_5layers(const float32_t* letter);
uint32_t net_5layers_optimized(const float32_t* letter);
uint32_t net_5layers_optimized_max(const float32_t* letter);

/* Exported types ------------------------------------------------------------*/
/* Exported constants --------------------------------------------------------*/
#define SDRAM_BANK_ADDR                 ((uint32_t)0xD0000000)

/* #define SDRAM_MEMORY_WIDTH            FMC_SDRAM_MEM_BUS_WIDTH_8 */
#define SDRAM_MEMORY_WIDTH            FMC_SDRAM_MEM_BUS_WIDTH_16

/* #define SDCLOCK_PERIOD                   FMC_SDRAM_CLOCK_PERIOD_2 */
#define SDCLOCK_PERIOD                FMC_SDRAM_CLOCK_PERIOD_3

#define SDRAM_TIMEOUT     ((uint32_t)0xFFFF) 

#define TIMx  TIM3
#define TIMx_CLK_ENABLE  __HAL_RCC_TIM3_CLK_ENABLE
//for TIMx's NVIC
#define TIMx_IRQn  TIM3_IRQn
#define TIMx_IRQHandler  TIM3_IRQHandler

#endif /* __MAIN_H */

/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
