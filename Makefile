# put your *.o targets here, make should handle the rest!
cube=/home/alex/usr/school/DIP/example/STM32Cube_FW_F4_V1.19.0

SRCS = system_stm32f4xx.c stm32f429i_discovery_sdram.c stm32f4xx_hal_sdram.c stm32f4xx_it.c \
			 stm32f4xx_hal_msp.c stm32f4xx_hal_tim.c stm32f4xx_hal_tim_ex.c stm32f4xx_hal.c stm32f4xx_hal_gpio.c stm32f4xx_hal_ltdc.c \
			 stm32f4xx_hal_rcc.c stm32f4xx_hal_cortex.c stm32f4xx_hal_rcc_ex.c stm32f4xx_hal_pwr_ex.c \
			 stm32f429i_discovery.c stm32f4xx_hal_i2c.c stm32f4xx_hal_spi.c stm32f4xx_hal_dma.c \
			 arm_conv_f32.c arm_conv_partial_f32.c arm_copy_f32.c arm_mat_add_f32.c\
			 stm32f4xx_ll_fmc.c ili9341.c syscalls.c conv.c activation_functions.c tests.c utility.c main.c 
# all the files will be generated with this name (main.elf, main.bin, main.hex, etc)

PROJ_NAME=main


# that's it, no need to change anything below this line!

###################################################

CC=arm-none-eabi-gcc
OBJCOPY=arm-none-eabi-objcopy

CFLAGS  = -g -O0 -Wall -T./SW4STM32/STM32F429I-Discovery/STM32F429ZITx_FLASH.ld -DSTM32F429xx -DARM_MATH_CM4 -D__FPU_PRESENT=1
CFLAGS += -mlittle-endian -mthumb -mcpu=cortex-m4 -mthumb-interwork
CFLAGS += -mfloat-abi=hard -mfpu=fpv4-sp-d16

###################################################

vpath %.c Src \
					${cube}/Drivers/STM32F4xx_HAL_Driver/Src/ 						  \
					${cube}/Drivers/BSP/STM32F429I-Discovery/								 \
					${cube}/Drivers/CMSIS/DSP_Lib/Source/FilteringFunctions/ \
					${cube}/Drivers/CMSIS/DSP_Lib/Source/SupportFunctions/ \
					${cube}/Drivers/CMSIS/DSP_Lib/Source/MatrixFunctions/ \
					${cube}/Drivers/BSP/Components/ili9341/


ROOT=$(shell pwd)

CFLAGS += -IInc 
CFLAGS += -I${cube}/Drivers/STM32F4xx_HAL_Driver/Inc/
CFLAGS += -I${cube}/Drivers/CMSIS/Device/ST/STM32F4xx/Include/
CFLAGS += -I${cube}/Drivers/CMSIS/Include/
CFLAGS += -I${cube}/Drivers/BSP/STM32F429I-Discovery/

SRCS += SW4STM32/startup_stm32f429xx.s # add startup file to build

OBJS = $(SRCS:.c=.o)

###################################################

.PHONY: proj

all: proj

proj: 	$(PROJ_NAME).elf

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

$(PROJ_NAME).elf: $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -o $@ 
	$(OBJCOPY) -O ihex $(PROJ_NAME).elf $(PROJ_NAME).hex
	$(OBJCOPY) -O binary $(PROJ_NAME).elf $(PROJ_NAME).bin

clean:
	rm -f $(PROJ_NAME).elf
	rm -f $(PROJ_NAME).hex
	rm -f $(PROJ_NAME).bin
	rm -f *.o
