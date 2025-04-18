#ifndef __SENSOR_H__
#define __SENSOR_H__

#include "stm32f10x_conf.h"

#define Trig(x) gpioB_pin_set(0, x);
#define Echo() GPIO_ReadInputDataBit(GPIOA, GPIO_Pin_2)


//处理智能传感器功能


void loop_sensor(void);

void ceju_jiaqu(void);
void carry_wood(void);
 

#endif