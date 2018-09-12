/* Includes ------------------------------------------------------------------*/
#include "main.h"

#include "letter.h"
#include "letters10_30.h"
#include "test_letters.h"

#ifdef PROFILE
  #include "exported_for_test_prof.h"
#else
  //#include "exported_for_test1_q9.h"
  //#include "exported_for_test17_only_opt.h"
  #include "exported_for_test17.h"
#endif

//#include "net_q7.c"
//#include "net_q9.c"

#ifdef PROFILE
  #include "tmp_out_s.h"
  #include "tmp_out_ss.h"
#endif
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
volatile unsigned int *one_spot = (volatile unsigned int *)0x81A0000;

/* Private macro -------------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*/
TIM_HandleTypeDef TimHandle;
uint16_t uwPrescalerValue = 0;

/* Status variables */
__IO uint32_t uwWriteReadStatus = 0;


/* Private function prototypes -----------------------------------------------*/
static void SystemClock_Config(void);

/* Private functions ---------------------------------------------------------*/
extern uint64_t time_counter = 0;

int main(void)
{    
  HAL_Init();
  
  /* Configure LED3 and LED4 */
  BSP_LED_Init(LED3);
  BSP_LED_Init(LED4);
  
  /* Configure the system clock to 180 MHz */
  SystemClock_Config();
  
  /* Compute the prescaler value to have TIM3 counter clock equal to 10 KHz */
  //uwPrescalerValue = (uint32_t) ((SystemCoreClock /2) / 10000) - 1; //orinal setting, with which most measurments were done
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

  GPIO_InitTypeDef GPIO_InitStruct;
  __HAL_RCC_GPIOC_CLK_ENABLE();
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_PULLUP;
  GPIO_InitStruct.Pin = GPIO_PIN_4;
  HAL_GPIO_Init(GPIOC, &GPIO_InitStruct);
  HAL_GPIO_WritePin(GPIOC, GPIO_PIN_4, GPIO_PIN_RESET);
  
  BSP_LED_Off(LED3);
  BSP_LED_Off(LED3);

  if (!test()) {
    BSP_LED_On(LED4);     
  } else { 
    BSP_LED_On(LED3);
  }

  q15_t letter_q9_t[784] = {[0 ... 783] = 0};
  arm_float_to_q9(num1, letter_q9_t, 784);
  uint32_t out5 = 0;

  start_time_measure(TimHandle);

  //currently 36 tests
  
  //uint32_t out = net_2layers(num1);
  //uint32_t out2 = classifier_test(&net_2layers);
  //uint32_t out3 = classifier_test(&net_3layers);
  //net_5layers_optimized(num1);
  //out5 = classifier_test(&net_5layers);
  out5 = classifier_test(&net_5layers_optimized);
 // out5 = classifier_test_q9_t(&net_q9_5layers);
  
 // for(uint32_t i=0; i<36;i++){
 //   net_q9_5layers(letter_q9_t);
 // }
  //out5_o = classifier_test(&net_5layers_optimized_max);

  uint64_t time = stop_time_measure(TimHandle);



  *one_spot = 42;

#ifdef PROFILE
  time_profiling(TimHandle);
#endif

  /* Infinite loop */  
  BSP_LED_Off(LED3);
  BSP_LED_Off(LED3);
  if (out5== 35){
    BSP_LED_On(LED3);     
    while(1) {}
  }else{
    BSP_LED_On(LED4);     
    while(1) {}
  }

}


void HAL_TIM_PeriodElapsedCallback(TIM_HandleTypeDef *htim)
{
  time_counter++;
  BSP_LED_Toggle(LED3);
}

#ifdef PROFILE
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
#endif

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
      //l2_full_connection[i] += dot_product_with_nth_column(l1_pooled_feature_maps+(j*4*4), l2_w+((j*4*4*100)+i), 4*4, 100);
      l2_full_connection[i] += dot_product(l1_pooled_feature_maps+(j*4*4), l2_w_o+((i*4*4*40)+(j*4*4)), 4*4);
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

uint32_t net_5layers_optimized_max(const float32_t* letter){
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
   // for(uint32_t j=0; j<4*4; j++){
   //   (l1_pooled_feature_maps+(i*4*4))[j] = ReLU((l1_pooled_feature_maps+(i*4*4))[j] + l1_b[i]);
   // }
    arm_offset_f32(l1_pooled_feature_maps+(i*4*4), l1_b[i], l1_pooled_feature_maps+(i*4*4), 4*4);
  }
  arm_fn_f32(l1_pooled_feature_maps, l1_pooled_feature_maps, (4*4)*l1_size, &ReLU);

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
    //l3_full_connection[i] = ReLU(l3_full_connection[i] + l3_b[i]);
  }
  arm_add_f32(l3_full_connection, l3_b, l3_full_connection, l3_size);
  arm_fn_f32(l3_full_connection, l3_full_connection, l3_size, &ReLU);

  //layer 4
  for(uint32_t i=0; i<l4_size; i++){
    //l4_soft_max_result[i] = dot_product_with_nth_column(l3_full_connection, l4_w+i, 100, 10);
    arm_dot_prod_f32(l3_full_connection, l4_w_o+(i*l3_size), l3_size, l4_soft_max_result+i);
    //l4_soft_max_result[i] = l4_soft_max_result[i] + l4_b[i];
  }
  arm_add_f32(l4_soft_max_result, l4_b, l4_soft_max_result, l4_size);
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
  for(uint32_t i=0; i<l0_size; i++){
    //convolution_optimized(letter, 28, l0_feature_maps+(i*24*24), l0_w_o+(i*(5*5+4*23)), 5);
    convolution_additive(letter, 28, l0_feature_maps+(i*24*24), l0_w+(i*(5*5)), 5);
    pooling_optimized(l0_feature_maps+(i*24*24), l0_pooled_feature_maps+(i*12*12), 24, &arm_max_f32);

   // for(uint32_t j=0; j<12*12; j++){
   //   (l0_pooled_feature_maps+(i*12*12))[j] = ReLU((l0_pooled_feature_maps+(i*12*12))[j] + l0_b[i]);
   // }
    arm_offset_f32(l0_pooled_feature_maps+(i*12*12), l0_b[i], l0_pooled_feature_maps+(i*12*12), 12*12);
  }
  arm_fn_f32(l0_pooled_feature_maps, l0_pooled_feature_maps, (12*12)*l0_size, &ReLU);

  //layer 1
  for(uint32_t i=0; i<l1_size; i++){
    for(uint32_t j=0; j<l0_size; j++){
      //convolution_optimized(l0_pooled_feature_maps+(j*12*12), 12, tmp_l1_f_m, l1_w_o+((i*(5*5+4*7)*l0_size)+(j*(5*5+4*7))), 5);
      //arm_add_f32(l1_feature_maps+(i*8*8), tmp_l1_f_m, l1_feature_maps+(i*8*8), 8*8);
      convolution_additive(l0_pooled_feature_maps+(j*12*12), 12, l1_feature_maps+(i*8*8), l1_w+((i*(5*5)*l0_size)+(j*(5*5))), 5);
    }
    pooling_optimized(l1_feature_maps+(i*8*8), l1_pooled_feature_maps+(i*4*4), 8, &arm_max_f32);
  //  for(uint32_t j=0; j<4*4; j++){
  //    (l1_pooled_feature_maps+(i*4*4))[j] = ReLU((l1_pooled_feature_maps+(i*4*4))[j] + l1_b[i]);
  //  }
    arm_offset_f32(l1_pooled_feature_maps+(i*4*4), l1_b[i], l1_pooled_feature_maps+(i*4*4), 4*4);
  }
  arm_fn_f32(l1_pooled_feature_maps, l1_pooled_feature_maps, (4*4)*l1_size, &ReLU);

  //layer 2
  for(uint32_t i=0; i<l2_size; i++){
   // for(uint32_t j=0; j<l1_size;j++){
   //   //l2_full_connection[i] += dot_product_with_nth_column(l1_pooled_feature_maps+(j*4*4), l2_w+((j*4*4*100)+i), 4*4, 100);

   //   arm_dot_prod_f32(l1_pooled_feature_maps+(j*4*4), l2_w_o+((i*4*4*40)+(j*4*4)) ,4*4 , &tmp);
   //   l2_full_connection[i] += tmp;
   // }
    arm_dot_prod_f32(l1_pooled_feature_maps, l2_w_o+(i*4*4*40), 4*4*40, l2_full_connection+i);
   // l2_full_connection[i] = ReLU(l2_full_connection[i] + l2_b[i]);
  }
  arm_add_f32(l2_full_connection, l2_b, l2_full_connection, l2_size);
  arm_fn_f32(l2_full_connection, l2_full_connection, l2_size, &ReLU);

  //layer 3
  for(uint32_t i=0; i<l3_size; i++){
    //l3_full_connection[i] = dot_product_with_nth_column(l2_full_connection, l3_w+i, 100, 100);
    arm_dot_prod_f32(l2_full_connection, l3_w_o+(i*l2_size), l2_size, l3_full_connection+i);
   // l3_full_connection[i] = ReLU(l3_full_connection[i] + l3_b[i]);
  }
  arm_add_f32(l3_full_connection, l3_b, l3_full_connection, l3_size);
  arm_fn_f32(l3_full_connection, l3_full_connection, l3_size, &ReLU);

  //layer 4
  for(uint32_t i=0; i<l4_size; i++){
    //l4_soft_max_result[i] = dot_product_with_nth_column(l3_full_connection, l4_w+i, 100, 10);
    arm_dot_prod_f32(l3_full_connection, l4_w_o+(i*l3_size), l3_size, l4_soft_max_result+i);
  //  l4_soft_max_result[i] = l4_soft_max_result[i] + l4_b[i];
  }
  arm_add_f32(l4_soft_max_result, l4_b, l4_soft_max_result, l4_size);
  soft_max(l4_soft_max_result, 10, out);

  return index_of_most_probable(out);
}

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

#ifdef  USE_FULL_ASSERT
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
