# put your *.o targets here, make should handle the rest!
cube=/home/alex/usr/school/DIP/example/STM32Cube_FW_F4_V1.19.0

LIBS = stm32f4xx_hal_tim.c stm32f4xx_hal_tim_ex.c stm32f4xx_hal.c stm32f4xx_hal_gpio.c  \
			 stm32f4xx_hal_rcc.c stm32f4xx_hal_cortex.c stm32f4xx_hal_rcc_ex.c stm32f4xx_hal_pwr_ex.c \
			 stm32f429i_discovery.c stm32f4xx_hal_i2c.c stm32f4xx_hal_spi.c stm32f4xx_hal_dma.c \
			 arm_max_f32.c arm_dot_prod_f32.c  arm_conv_f32.c arm_conv_partial_f32.c \
			 arm_copy_f32.c arm_mat_add_f32.c arm_add_f32.c arm_offset_f32.c arm_offset_q15.c\
			 stm32f4xx_ll_fmc.c arm_float_to_q15.c arm_max_q15.c arm_add_q15.c arm_dot_prod_q15.c\
			 arm_q15_to_float.c arm_float_to_q7.c arm_q7_to_float.c

SRCS = ili9341.c syscalls.c conv.c activation_functions.c tests.c utility.c  math_helper.c main.c \
			 stm32f4xx_hal_msp.c system_stm32f4xx.c stm32f4xx_it.c

PROFILE_LIBS = arm_dot_prod_q31.c arm_conv_q31.c \
							 arm_float_to_q31.c arm_conv_q15.c arm_dot_prod_q7.c \
							 arm_conv_q7.c
							 
PROFILE_SRCS = time_profiling.c


PROJ_NAME=main

CC=arm-none-eabi-gcc
OBJCOPY=arm-none-eabi-objcopy

CFLAGS  = -DNDEBUG -g -O3 -Wall -T./SW4STM32/STM32F429I-Discovery/STM32F429ZITx_FLASH.ld -DSTM32F429xx -DARM_MATH_CM4 -D__FPU_PRESENT=1
CFLAGS += -mlittle-endian -mthumb -mcpu=cortex-m4 -mthumb-interwork
CFLAGS += -mfpu=fpv4-sp-d16 -mfloat-abi=hard

###################################################

vpath %.c Src \
					${cube}/Drivers/STM32F4xx_HAL_Driver/Src/ 						 	 \
					${cube}/Drivers/BSP/STM32F429I-Discovery/								 \
					${cube}/Drivers/CMSIS/DSP_Lib/Source/FilteringFunctions/ \
					${cube}/Drivers/CMSIS/DSP_Lib/Source/BasicMathFunctions/ \
					${cube}/Drivers/CMSIS/DSP_Lib/Source/SupportFunctions/	 \
					${cube}/Drivers/CMSIS/DSP_Lib/Source/MatrixFunctions/		 \
					${cube}/Drivers/CMSIS/DSP_Lib/Source/StatisticsFunctions/		 \
					${cube}/Drivers/BSP/Components/ili9341/


ROOT=$(shell pwd)

CFLAGS += -IInc 
CFLAGS += -I${cube}/Drivers/STM32F4xx_HAL_Driver/Inc/
CFLAGS += -I${cube}/Drivers/CMSIS/Device/ST/STM32F4xx/Include/
CFLAGS += -I${cube}/Drivers/CMSIS/Include/
CFLAGS += -I${cube}/Drivers/BSP/STM32F429I-Discovery/

SRCS += SW4STM32/startup_stm32f429xx.s # add startup file to build

#OBJS = $(SRCS:.c=.o)
OBJS = $(patsubst %.c, obj/%.o, $(SRCS))
PROFILE_OBJS = $(patsubst %.c, obj/%.o, $(PROFILE_SRCS))

LIBS_OBJS = $(patsubst %.c, obj/%.o, $(LIBS))
PROFILE_LIBS_OBJS = $(patsubst %.c, obj/%.o, $(PROFILE_LIBS))

###################################################

.PHONY: proj

proj: 	$(PROJ_NAME).elf

obj/%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

$(PROJ_NAME).elf: $(OBJS) $(LIBS_OBJS)
	$(CC) $(CFLAGS) $(OBJS) $(LIBS_OBJS) -o $@ -lm
	$(OBJCOPY) -O ihex $(PROJ_NAME).elf $(PROJ_NAME).hex
	$(OBJCOPY) -O binary $(PROJ_NAME).elf $(PROJ_NAME).bin

profile: CFLAGS += -DPROFILE 
profile: rm_profile_dependent $(OBJS) $(LIBS_OBJS) $(PROFILE_OBJS) $(PROFILE_LIBS_OBJS)
	$(CC) $(CFLAGS) -DPROFILE $(PROFILE_OBJS) $(LIBS_OBJS) $(PROFILE_LIBS_OBJS) $(OBJS) -o $(PROJ_NAME).elf -lm
	$(OBJCOPY) -O ihex $(PROJ_NAME).elf $(PROJ_NAME).hex
	$(OBJCOPY) -O binary $(PROJ_NAME).elf $(PROJ_NAME).bin
	
rm_profile_dependent:
	rm -f obj/main.o

clean:
	rm -f $(PROJ_NAME).elf
	rm -f $(PROJ_NAME).hex
	rm -f $(PROJ_NAME).bin
	rm -f obj/*.o
