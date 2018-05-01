/**
  ******************************************************************************
  * @file    FMC/FMC_SDRAM/Src/main.c 
  * @author  MCD Application Team
  * @brief   This sample code shows how to use STM32F4xx FMC HAL API to access 
  *          by read and write operation the SDRAM external memory device.
  ******************************************************************************
  * @attention
  *
  * <h2><center>&copy; COPYRIGHT(c) 2017 STMicroelectronics</center></h2>
  *
  * Redistribution and use in source and binary forms, with or without modification,
  * are permitted provided that the following conditions are met:
  *   1. Redistributions of source code must retain the above copyright notice,
  *      this list of conditions and the following disclaimer.
  *   2. Redistributions in binary form must reproduce the above copyright notice,
  *      this list of conditions and the following disclaimer in the documentation
  *      and/or other materials provided with the distribution.
  *   3. Neither the name of STMicroelectronics nor the names of its contributors
  *      may be used to endorse or promote products derived from this software
  *      without specific prior written permission.
  *
  * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
  * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
  * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
  * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
  * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
  * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
  * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
  * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
  * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
  *
  ******************************************************************************
  */

/* Includes ------------------------------------------------------------------*/
#include "main.h"

#include "letter.h"
#include "letters10_30.h"
#include "test_letters.h"
//#include "tmp_out.h"
#include "exported_for_test17.h"

#include "tmp_out_s.h"
#include "tmp_out_ss.h"
/** @addtogroup STM32F4xx_HAL_Examples
  * @{
  */

/** @addtogroup FMC_SDRAM_Basic
  * @{
  */ 

/* Private typedef -----------------------------------------------------------*/
/* Private define ------------------------------------------------------------*/
#define BUFFER_SIZE         ((uint32_t)0x0100)
#define WRITE_READ_ADDR     ((uint32_t)0x0800)
#define REFRESH_COUNT       ((uint32_t)0x056A)   /* SDRAM refresh counter (90MHz SDRAM clock) */
    
float32_t *dynamic_letter = (float32_t*) 0x8180000;
uint32_t *one_spot = (uint32_t*) 0x81A0000;

/* Private macro -------------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*/
TIM_HandleTypeDef TimHandle;
uint16_t uwPrescalerValue = 0;
/* SDRAM handler declaration */
SDRAM_HandleTypeDef hsdram;
FMC_SDRAM_TimingTypeDef SDRAM_Timing;
FMC_SDRAM_CommandTypeDef command;

/* Read/Write Buffers */
uint32_t aTxBuffer[BUFFER_SIZE];
float32_t aRxBuffer[BUFFER_SIZE];

/* Status variables */
__IO uint32_t uwWriteReadStatus = 0;

/* Counter index */
uint32_t uwIndex = 0;

uint64_t time_counter = 0;

/* Private function prototypes -----------------------------------------------*/
static void SystemClock_Config(void);
static void Error_Handler(void);
static void SDRAM_Initialization_Sequence(SDRAM_HandleTypeDef *hsdram, FMC_SDRAM_CommandTypeDef *Command);
static void Fill_Buffer(uint32_t *pBuffer, uint32_t uwBufferLenght, uint32_t uwOffset);

/* Private functions ---------------------------------------------------------*/

/**
  * @brief  Main program
  * @param  None
  * @retval None
  */
int main(void)
{    
  /* STM32F4xx HAL library initialization:
       - Configure the Flash prefetch, instruction and Data caches
       - Configure the Systick to generate an interrupt each 1 msec
       - Set NVIC Group Priority to 4
       - Global MSP (MCU Support Package) initialization
     */
  HAL_Init();
  
  /* Configure LED3 and LED4 */
  BSP_LED_Init(LED3);
  BSP_LED_Init(LED4);
  
  /* Configure the system clock to 180 MHz */
  SystemClock_Config();
  
  /* Compute the prescaler value to have TIM3 counter clock equal to 10 KHz */
  uwPrescalerValue = (uint32_t) ((SystemCoreClock /2) / 10000) - 1;
  
  /* Set TIMx instance */
  TimHandle.Instance = TIMx;
  /* Initialize TIM3 peripheral as follows:
       + Period = 10000 - 1
       + Prescaler = ((SystemCoreClock/2)/10000) - 1
       + ClockDivision = 0
       + Counter direction = Up
  */
  //TimHandle.Init.Period = 10000 - 1;  //1sec counter, cause I have freq 10KHz, counting to 10K -> 1s
  TimHandle.Init.Period = 10 - 1;       //0.25 sec counter, cause I have freq 10KHz, counting to 2,5K -> 1ms
  TimHandle.Init.Prescaler = uwPrescalerValue;
  TimHandle.Init.ClockDivision = 0;
  TimHandle.Init.CounterMode = TIM_COUNTERMODE_UP;
  if(HAL_TIM_Base_Init(&TimHandle) != HAL_OK)
  {
    /* Initialization Error */
    Error_Handler();
  }
  

  /*##-1- Configure the SDRAM device #########################################*/
  /* SDRAM device configuration */ 
  hsdram.Instance = FMC_SDRAM_DEVICE;
  
  /* Timing configuration for 90 MHz of SDRAM clock frequency (180MHz/2) */
  /* TMRD: 2 Clock cycles */
  SDRAM_Timing.LoadToActiveDelay    = 2;
  /* TXSR: min=70ns (6x11.90ns) */
  SDRAM_Timing.ExitSelfRefreshDelay = 7;
  /* TRAS: min=42ns (4x11.90ns) max=120k (ns) */
  SDRAM_Timing.SelfRefreshTime      = 4;
  /* TRC:  min=63 (6x11.90ns) */        
  SDRAM_Timing.RowCycleDelay        = 7;
  /* TWR:  2 Clock cycles */
  SDRAM_Timing.WriteRecoveryTime    = 2;
  /* TRP:  15ns => 2x11.90ns */
  SDRAM_Timing.RPDelay              = 2;
  /* TRCD: 15ns => 2x11.90ns */
  SDRAM_Timing.RCDDelay             = 2;

  hsdram.Init.SDBank             = FMC_SDRAM_BANK2;
  hsdram.Init.ColumnBitsNumber   = FMC_SDRAM_COLUMN_BITS_NUM_8;
  hsdram.Init.RowBitsNumber      = FMC_SDRAM_ROW_BITS_NUM_12;
  hsdram.Init.MemoryDataWidth    = SDRAM_MEMORY_WIDTH;
  hsdram.Init.InternalBankNumber = FMC_SDRAM_INTERN_BANKS_NUM_4;
  hsdram.Init.CASLatency         = FMC_SDRAM_CAS_LATENCY_3;
  hsdram.Init.WriteProtection    = FMC_SDRAM_WRITE_PROTECTION_DISABLE;
  hsdram.Init.SDClockPeriod      = SDCLOCK_PERIOD;
  hsdram.Init.ReadBurst          = FMC_SDRAM_RBURST_DISABLE;
  hsdram.Init.ReadPipeDelay      = FMC_SDRAM_RPIPE_DELAY_1;

  /* Initialize the SDRAM controller */
  if(HAL_SDRAM_Init(&hsdram, &SDRAM_Timing) != HAL_OK)
  {
    /* Initialization Error */
    Error_Handler(); 
  }
  
  /* Program the SDRAM external device */
  SDRAM_Initialization_Sequence(&hsdram, &command);   
    
  /*##-2- SDRAM memory read/write access #####################################*/  
  
  /* Fill the buffer to write */
  Fill_Buffer(aTxBuffer, BUFFER_SIZE, 0xA244250F);   

/* Write data to the SDRAM memory */
  for (uwIndex = 0; uwIndex < 20000; uwIndex++)
  {
    *(__IO float32_t*) (SDRAM_BANK_ADDR + WRITE_READ_ADDR + 4*uwIndex) = l1_w[uwIndex];
  }    
  for (uwIndex = 20000; uwIndex < 84000; uwIndex++)
  {
    *(__IO float32_t*) (SDRAM_BANK_ADDR + WRITE_READ_ADDR + 4*uwIndex) = l2_w[uwIndex];
  }    

/* Read back data from the SDRAM memory */
  for (uwIndex = 0; uwIndex < BUFFER_SIZE; uwIndex++)
  {
    aRxBuffer[uwIndex] = *(__IO float32_t*) (SDRAM_BANK_ADDR + WRITE_READ_ADDR + 4*uwIndex);
   } 

/*##-3- Checking data integrity ############################################*/    

  for (uwIndex = 0; (uwIndex < BUFFER_SIZE) && (uwWriteReadStatus == 0); uwIndex++)
  {
    if (aRxBuffer[uwIndex] != l1_w[uwIndex])
    {
      uwWriteReadStatus++;
    }
  }	

  BSP_LED_Off(LED3);
  BSP_LED_Off(LED3);
  if (!test())
  {
    /* KO */
    /* Turn on LED4 */
    BSP_LED_On(LED4);     
  }
  else
  { 
    /* OK */
    /* Turn on LED3 */
    BSP_LED_On(LED3);
  }

 // float32_t l0_w_all_in_one[15013] = {[0 ... 15012] = 0};
 // for(uint32_t i=0; i<20; i++){
 //   arm_copy_f32(l0_w_o+(i*117), l0_w_all_in_one+((19-i)*(28*28)), 117);
 // }

  /*##-2- Start the TIM Base generation in interrupt mode ####################*/
  /* Start Channel1 */
  if(HAL_TIM_Base_Start_IT(&TimHandle) != HAL_OK)
  {
    /* Starting Error */
    Error_Handler();
  }

  //currently 36 tests
  // its starting to take a while
  //
  //uint32_t out = net_2layers(num1);
 // uint32_t out2 = classifier_test(&net_2layers);
 // uint32_t out3 = classifier_test(&net_3layers);
  //uint32_t out5 = classifier_test(&net_5layers);
  uint32_t out5_o = classifier_test(&net_5layers_optimized);
  /*##-2- Start the TIM Base generation in interrupt mode ####################*/
  /* Start Channel1 */
  if(HAL_TIM_Base_Stop_IT(&TimHandle) != HAL_OK)
  {
    /* Starting Error */
    Error_Handler();
  }

  GPIO_InitTypeDef GPIO_InitStruct;
  __HAL_RCC_GPIOC_CLK_ENABLE();
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_PULLUP;
  GPIO_InitStruct.Pin = GPIO_PIN_8;
  HAL_GPIO_Init(GPIOC, &GPIO_InitStruct);

  /* Infinite loop */  
  BSP_LED_On(LED4);     
  *one_spot = net_5layers_optimized(dynamic_letter);
  while (1)
  {
  }
}

void HAL_TIM_PeriodElapsedCallback(TIM_HandleTypeDef *htim)
{
  time_counter++;
  BSP_LED_Toggle(LED3);
}

uint32_t net_2layers(const float32_t* letter){
  float32_t l0_feature_maps[24*24*20] = {[0 ... 11519] = 0};
  float32_t l0_pooled_feature_maps[12*12*20] = {[0 ... (12*12*20)-1] = 0};

  float32_t l4_soft_max_result[10] = {[0 ... 9] = 0};
  float32_t out[10] = {[0 ... 9] = 0};

  uint32_t l0_size = 20;
  uint32_t l4_size = 10;

  //layer 0
  for(uint32_t i=0; i<l0_size; i++){
    convolution_with_activation(letter, 28, l0_feature_maps+(i*24*24), L2_l0_w+(i*5*5), 5, L2_l0_b[i], &ReLU);
    pooling(l0_feature_maps+(i*24*24), l0_pooled_feature_maps+(i*12*12), 24, &max);
  }

  //layer 2
  for(uint32_t i=0; i<l4_size; i++){ //for each neuron
    for(uint32_t j=0; j<l0_size;j++){ //for each feature
      l4_soft_max_result[i] += dot_product_with_nth_column(l0_pooled_feature_maps+(j*12*12), L2_l1_w+((j*12*12*10)+i), 12*12, 10);
    }
    l4_soft_max_result[i] = l4_soft_max_result[i] + L2_l1_b[i];
  }
  soft_max(l4_soft_max_result, 10, out);

  return index_of_most_probable(out);

}

uint32_t net_3layers(const float32_t* letter){

  float32_t l0_feature_maps[24*24*20] = {[0 ... 11519] = 0};
  float32_t l0_pooled_feature_maps[12*12*20] = {[0 ... (12*12*20)-1] = 0};

  float32_t l1_feature_maps[8*8*40] = {[0 ... 2559] = 0};
  float32_t l1_pooled_feature_maps[4*4*40] = {[0 ... (4*4*40)-1] = 0};

  float32_t l4_soft_max_result[10] = {[0 ... 9] = 0};
  float32_t out[10] = {[0 ... 9] = 0};

  uint32_t l0_size = 20;
  uint32_t l1_size = 40;
  uint32_t l4_size = 10;

  //layer 0
  for(uint32_t i=0; i<l0_size; i++){
    convolution_with_activation(letter, 28, l0_feature_maps+(i*24*24), L3_l0_w+(i*5*5), 5, L3_l0_b[i], &ReLU);
    pooling(l0_feature_maps+(i*24*24), l0_pooled_feature_maps+(i*12*12), 24, &max);
  }

  //layer 1
  for(uint32_t i=0; i<l1_size; i++){
   for(uint32_t j=0; j<l0_size; j++){
     convolution_additive(l0_pooled_feature_maps+(j*12*12), 12, l1_feature_maps+(i*8*8), L3_l1_w+((i*5*5*l0_size)+(j*5*5)), 5);
   }
   for(uint32_t j=0; j<8*8; j++){
     (l1_feature_maps+(i*8*8))[j] = ReLU((l1_feature_maps+(i*8*8))[j] + L3_l1_b[i]);
   }
   pooling(l1_feature_maps+(i*8*8), l1_pooled_feature_maps+(i*4*4), 8, &max);
  }

  //layer 2
  for(uint32_t i=0; i<l4_size; i++){ //for each neuron
    for(uint32_t j=0; j<l1_size;j++){ //for each feature
      l4_soft_max_result[i] += dot_product_with_nth_column(l1_pooled_feature_maps+(j*4*4), L3_l2_w+((j*4*4*10)+i), 4*4, 10);
    }
    l4_soft_max_result[i] = l4_soft_max_result[i] + L3_l2_b[i];
  }
  soft_max(l4_soft_max_result, 10, out);

  return index_of_most_probable(out);

}

uint32_t net_5layers(const float32_t* letter){
  float32_t l0_feature_maps[24*24*20] = {[0 ... 11519] = 0};
  float32_t l0_pooled_feature_maps[12*12*20] = {[0 ... (12*12*20)-1] = 0};

  float32_t l1_feature_maps[8*8*40] = {[0 ... 2559] = 0};
  float32_t l1_pooled_feature_maps[4*4*40] = {[0 ... (4*4*40)-1] = 0};

  float32_t l2_full_connection[100] = {[0 ... 99] = 0};
  float32_t l3_full_connection[100] = {[0 ... 99] = 0};
  float32_t l4_soft_max_result[10] = {[0 ... 9] = 0};
  float32_t out[10] = {[0 ... 9] = 0};

  uint32_t l0_size = 20;
  uint32_t l1_size = 40;
  uint32_t l2_size = 100;
  uint32_t l3_size = 100;
  uint32_t l4_size = 10;

  //layer 0
  for(uint32_t i=0; i<l0_size; i++){
    convolution_additive(letter, 28, l0_feature_maps+(i*24*24), l0_w+(i*5*5), 5);
    pooling(l0_feature_maps+(i*24*24), l0_pooled_feature_maps+(i*12*12), 24, &max);
    for(uint32_t j=0; j<12*12; j++){
      (l0_pooled_feature_maps+(i*12*12))[j] = ReLU((l0_pooled_feature_maps+(i*12*12))[j] + l0_b[i]);
    }
  }

  //layer 1
  for(uint32_t i=0; i<l1_size; i++){
   for(uint32_t j=0; j<l0_size; j++){
     convolution_additive(l0_pooled_feature_maps+(j*12*12), 12, l1_feature_maps+(i*8*8), l1_w+((i*5*5*l0_size)+(j*5*5)), 5);
   }
   pooling(l1_feature_maps+(i*8*8), l1_pooled_feature_maps+(i*4*4), 8, &max);
   for(uint32_t j=0; j<4*4; j++){
     (l1_pooled_feature_maps+(i*4*4))[j] = ReLU((l1_pooled_feature_maps+(i*4*4))[j] + l1_b[i]);
   }
  }

  //layer 2
  for(uint32_t i=0; i<l2_size; i++){
    for(uint32_t j=0; j<l1_size;j++){
      l2_full_connection[i] += dot_product_with_nth_column(l1_pooled_feature_maps+(j*4*4), l2_w+((j*4*4*100)+i), 4*4, 100);
    }
    l2_full_connection[i] = ReLU(l2_full_connection[i] + l2_b[i]);
  }

  //layer 3
  for(uint32_t i=0; i<l3_size; i++){
    l3_full_connection[i] = dot_product_with_nth_column(l2_full_connection, l3_w+i, 100, 100);
    l3_full_connection[i] = ReLU(l3_full_connection[i] + l3_b[i]);
  }

  //layer 4
  for(uint32_t i=0; i<l4_size; i++){
    l4_soft_max_result[i] = dot_product_with_nth_column(l3_full_connection, l4_w+i, 100, 10);
    l4_soft_max_result[i] = l4_soft_max_result[i] + l4_b[i];
  }
  soft_max(l4_soft_max_result, 10, out);

  return index_of_most_probable(out);
}

uint32_t net_5layers_optimized(const float32_t* letter){
  //float32_t l0_feature_maps[24*24*20] = {[0 ... 11519] = 0};
  float32_t l0_feature_maps[24*24*21] = {[0 ... (24*24*21)-1] = 0}; //multiply by 21, cause we need some execess space for insitu convolution
 // float32_t l0_feature_maps[18680+24*24] = {[0 ... (24*24+18680)-1] = 0};
  float32_t l0_pooled_feature_maps[12*12*20] = {[0 ... (12*12*20)-1] = 0};

  float32_t l1_feature_maps[8*8*40] = {[0 ... 2559] = 0};
  float32_t tmp_l1_f_m[200] = {[0 ... (200)-1] = 0};
  float32_t l1_pooled_feature_maps[4*4*40] = {[0 ... (4*4*40)-1] = 0};

  float32_t l2_full_connection[100] = {[0 ... 99] = 0};
  float32_t l3_full_connection[100] = {[0 ... 99] = 0};
  float32_t l4_soft_max_result[10] = {[0 ... 9] = 0};
  float32_t out[10] = {[0 ... 9] = 0};

  uint32_t l0_size = 20;
  uint32_t l1_size = 40;
  uint32_t l2_size = 100;
  uint32_t l3_size = 100;
  uint32_t l4_size = 10;

  //layer 0

  //convolution_optimized_one_go(letter, 28, l0_feature_maps, l0_w_o_in_one, 15680);
  for(uint32_t i=0; i<l0_size; i++){
    convolution_optimized(letter, 28, l0_feature_maps+(i*24*24), l0_w_o+(i*(5*5+4*23)), 5);
    pooling_optimized(l0_feature_maps+(i*24*24), l0_pooled_feature_maps+(i*12*12), 24, &arm_max_f32);

//    for(uint32_t j=0; j<12*12; j++){
//      (l0_pooled_feature_maps+(i*12*12))[j] = ReLU((l0_pooled_feature_maps+(i*12*12))[j] + l0_b[i]);
//    }
    arm_offset_f32(l0_pooled_feature_maps+(i*12*12), l0_b[i], l0_pooled_feature_maps+(i*12*12), 12*12);
  }
  arm_fn_f32(l0_pooled_feature_maps, l0_pooled_feature_maps, (12*12)*l0_size, &ReLU);

  //layer 1
  for(uint32_t i=0; i<l1_size; i++){
   for(uint32_t j=0; j<l0_size; j++){
     convolution_optimized(l0_pooled_feature_maps+(j*12*12), 12, tmp_l1_f_m, l1_w_o+((i*(5*5+4*7)*l0_size)+(j*(5*5+4*7))), 5);
     arm_add_f32(l1_feature_maps+(i*8*8), tmp_l1_f_m, l1_feature_maps+(i*8*8), 8*8);
   }
   pooling_optimized(l1_feature_maps+(i*8*8), l1_pooled_feature_maps+(i*4*4), 8, &arm_max_f32);
   for(uint32_t j=0; j<4*4; j++){
     (l1_pooled_feature_maps+(i*4*4))[j] = ReLU((l1_pooled_feature_maps+(i*4*4))[j] + l1_b[i]);
   }
  }

  //layer 2
  for(uint32_t i=0; i<l2_size; i++){
   // for(uint32_t j=0; j<l1_size;j++){
   //   //l2_full_connection[i] += dot_product_with_nth_column(l1_pooled_feature_maps+(j*4*4), l2_w+((j*4*4*100)+i), 4*4, 100);

   //   arm_dot_prod_f32(l1_pooled_feature_maps+(j*4*4), l2_w_o+((i*4*4*40)+(j*4*4)) ,4*4 , &tmp);
   //   l2_full_connection[i] += tmp;
   // }
    arm_dot_prod_f32(l1_pooled_feature_maps, l2_w_o+(i*4*4*40), 4*4*40, l2_full_connection+i);
//    l2_full_connection[i] = ReLU(l2_full_connection[i] + l2_b[i]);
  }
  arm_add_f32(l2_full_connection, l2_b, l2_full_connection, l2_size);
  arm_fn_f32(l2_full_connection, l2_full_connection, l2_size, &ReLU);

  //layer 3
  for(uint32_t i=0; i<l3_size; i++){
    //l3_full_connection[i] = dot_product_with_nth_column(l2_full_connection, l3_w+i, 100, 100);
    arm_dot_prod_f32(l2_full_connection, l3_w_o+(i*l2_size), l2_size, l3_full_connection+i);
    //l2_full_connection[i] += tmp;


    l3_full_connection[i] = ReLU(l3_full_connection[i] + l3_b[i]);
  }

  //layer 4
  for(uint32_t i=0; i<l4_size; i++){
    //l4_soft_max_result[i] = dot_product_with_nth_column(l3_full_connection, l4_w+i, 100, 10);
    arm_dot_prod_f32(l3_full_connection, l4_w_o+(i*l3_size), l3_size, l4_soft_max_result+i);
    l4_soft_max_result[i] = l4_soft_max_result[i] + l4_b[i];
  }
  soft_max(l4_soft_max_result, 10, out);

  return index_of_most_probable(out);
}

/**
  * @brief  System Clock Configuration
  *         The system Clock is configured as follow : 
  *            System Clock source            = PLL (HSE)
  *            SYSCLK(Hz)                     = 180000000
  *            HCLK(Hz)                       = 180000000
  *            AHB Prescaler                  = 1
  *            APB1 Prescaler                 = 4
  *            APB2 Prescaler                 = 2
  *            HSE Frequency(Hz)              = 8000000
  *            PLL_M                          = 8
  *            PLL_N                          = 360
  *            PLL_P                          = 2
  *            PLL_Q                          = 7
  *            VDD(V)                         = 3.3
  *            Main regulator output voltage  = Scale1 mode
  *            Flash Latency(WS)              = 5
  * @param  None
  * @retval None
  */
static void SystemClock_Config(void)
{
  RCC_ClkInitTypeDef RCC_ClkInitStruct;
  RCC_OscInitTypeDef RCC_OscInitStruct;

  /* Enable Power Control clock */
  __HAL_RCC_PWR_CLK_ENABLE();
  
  /* The voltage scaling allows optimizing the power consumption when the device is 
     clocked below the maximum system frequency, to update the voltage scaling value 
     regarding system frequency refer to product datasheet.  */
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);
  
  /* Enable HSE Oscillator and activate PLL with HSE as source */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSE;
  RCC_OscInitStruct.HSEState = RCC_HSE_ON;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSE;
  RCC_OscInitStruct.PLL.PLLM = 8;
  RCC_OscInitStruct.PLL.PLLN = 360;
  RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV2;
  RCC_OscInitStruct.PLL.PLLQ = 7;
  HAL_RCC_OscConfig(&RCC_OscInitStruct);

  /* Activate the Over-Drive mode */
  HAL_PWREx_EnableOverDrive();
  
  /* Select PLL as system clock source and configure the HCLK, PCLK1 and PCLK2 
     clocks dividers */
  RCC_ClkInitStruct.ClockType = (RCC_CLOCKTYPE_SYSCLK | RCC_CLOCKTYPE_HCLK | RCC_CLOCKTYPE_PCLK1 | RCC_CLOCKTYPE_PCLK2);
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV4;  
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV2;  
  HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_5);
}

/**
  * @brief  This function is executed in case of error occurrence.
  * @param  None
  * @retval None
  */
static void Error_Handler(void)
{
  /* Turn LED3 on */
  BSP_LED_On(LED3);
  while(1)
  {
  }
}

/**
  * @brief  Perform the SDRAM exernal memory inialization sequence
  * @param  hsdram: SDRAM handle
  * @param  Command: Pointer to SDRAM command structure
  * @retval None
  */
static void SDRAM_Initialization_Sequence(SDRAM_HandleTypeDef *hsdram, FMC_SDRAM_CommandTypeDef *Command)
{
  __IO uint32_t tmpmrd =0;
  /* Step 3:  Configure a clock configuration enable command */
  Command->CommandMode 			 = FMC_SDRAM_CMD_CLK_ENABLE;
  Command->CommandTarget 		 = FMC_SDRAM_CMD_TARGET_BANK2;
  Command->AutoRefreshNumber 	 = 1;
  Command->ModeRegisterDefinition = 0;

  /* Send the command */
  HAL_SDRAM_SendCommand(hsdram, Command, 0x1000);

  /* Step 4: Insert 100 ms delay */
  HAL_Delay(100);
    
  /* Step 5: Configure a PALL (precharge all) command */ 
  Command->CommandMode 			 = FMC_SDRAM_CMD_PALL;
  Command->CommandTarget 	     = FMC_SDRAM_CMD_TARGET_BANK2;
  Command->AutoRefreshNumber 	 = 1;
  Command->ModeRegisterDefinition = 0;

  /* Send the command */
  HAL_SDRAM_SendCommand(hsdram, Command, 0x1000);  
  
  /* Step 6 : Configure a Auto-Refresh command */ 
  Command->CommandMode 			 = FMC_SDRAM_CMD_AUTOREFRESH_MODE;
  Command->CommandTarget 		 = FMC_SDRAM_CMD_TARGET_BANK2;
  Command->AutoRefreshNumber 	 = 4;
  Command->ModeRegisterDefinition = 0;

  /* Send the command */
  HAL_SDRAM_SendCommand(hsdram, Command, 0x1000);
  
  /* Step 7: Program the external memory mode register */
  tmpmrd = (uint32_t)SDRAM_MODEREG_BURST_LENGTH_2          |
                     SDRAM_MODEREG_BURST_TYPE_SEQUENTIAL   |
                     SDRAM_MODEREG_CAS_LATENCY_3           |
                     SDRAM_MODEREG_OPERATING_MODE_STANDARD |
                     SDRAM_MODEREG_WRITEBURST_MODE_SINGLE;
  
  Command->CommandMode = FMC_SDRAM_CMD_LOAD_MODE;
  Command->CommandTarget 		 = FMC_SDRAM_CMD_TARGET_BANK2;
  Command->AutoRefreshNumber 	 = 1;
  Command->ModeRegisterDefinition = tmpmrd;

  /* Send the command */
  HAL_SDRAM_SendCommand(hsdram, Command, 0x1000);
  
  /* Step 8: Set the refresh rate counter */
  /* (15.62 us x Freq) - 20 */
  /* Set the device refresh counter */
  HAL_SDRAM_ProgramRefreshRate(hsdram, REFRESH_COUNT); 
}
                  
/**
  * @brief  Fills buffer with user predefined data.
  * @param  pBuffer: pointer on the buffer to fill
  * @param  uwBufferLenght: size of the buffer to fill
  * @param  uwOffset: first value to fill on the buffer
  * @retval None
  */
static void Fill_Buffer(uint32_t *pBuffer, uint32_t uwBufferLenght, uint32_t uwOffset)
{
  uint32_t tmpIndex = 0;

  /* Put in global buffer different values */
  for (tmpIndex = 0; tmpIndex < uwBufferLenght; tmpIndex++ )
  {
    pBuffer[tmpIndex] = tmpIndex + uwOffset;
  }
}                  

#ifdef  USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t* file, uint32_t line)
{ 
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */

  /* Infinite loop */
  while (1)
  {
  }
}
#endif

/**
  * @}
  */ 

/**
  * @}
  */ 

/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
