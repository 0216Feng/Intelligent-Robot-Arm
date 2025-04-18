#include "z_rcc.h"		//配置时钟文件
#include "z_gpio.h"		//配置IO口文件
#include "z_global.h"	//存放全局变量
#include "z_delay.h"	//存放延时函数
#include "z_type.h"		//存放类型定义
#include "z_usart.h"	//存放串口功能文件
#include "z_timer.h"	//存放定时器功能文件
#include "z_ps2.h"		//存放索尼手柄
#include "z_w25q64.h"	//存储芯片的操作
#include "z_adc.h"		//ADC初始化
#include <stdio.h>		//标准库文件
#include <string.h>		//标准库文件
#include <math.h>		//标准库文件
#include "z_kinematics.h"	//逆运动学算法
#include "stm32f10x_iwdg.h"
#include "z_sensor.h"
#include "z_actiongroupcontrol.h"

u16 kms_y = 0;
u8 get_count = 0;
u8 carry_step = 0;
int category = -1;

// 用于串口接收的外部变量声明
extern u8 uart_receive_buf[1024];
extern u8 uart1_get_ok;


void loop_sensor(void) {
    ceju_jiaqu();
}

/*************************************************************
函数名称：parse_data()
功能介绍：从接收到的串口字符串中解析距离值和垃圾分类类别
函数参数：无
返回值：  解析到的距离值和类别数字，解析失败返回-1  
*************************************************************/
int parse_data(float *distance, int *category) {
    char *d_ptr = NULL;
    char *c_ptr = NULL;
    char *end_ptr = NULL;
    float temp_dist = 0.0;
    int temp_cat = -1;
    
    // 初始化输出参数
    *distance = 0.0;
    *category = -1;

	sprintf((char*)cmd_return, "Raw input: %s\r\n", uart_receive_buf);
	uart1_send_str(cmd_return);
    
    // 查找格式标识
    if(strstr((char*)uart_receive_buf, "%D") != NULL) {
        d_ptr = strstr((char*)uart_receive_buf, "%D");
    } else {
        d_ptr = strstr((char*)uart_receive_buf, "D");
    }
    
    if(d_ptr == NULL) {
        return 0; // 未找到距离标识
    }
    
    c_ptr = strstr(d_ptr, "C");
    if(c_ptr == NULL) {
        return 0; // 未找到类别标识
    }
    
    end_ptr = strstr(c_ptr, "*");
    if(end_ptr == NULL) {
        // 如果没有找到星号，尝试继续解析（兼容旧格式）
    }
    
    // 解析格式：%D15.0C13* - D后面是距离，C后面是类别，*是结束符
    if(d_ptr[0] == '%') {
        if(sscanf(d_ptr, "%%D%f", &temp_dist) != 1) {
            return 0; // 距离解析失败
        }
    } else {
        if(sscanf(d_ptr, "D%f", &temp_dist) != 1) {
            return 0; // 距离解析失败
        }
    }
    
    if(sscanf(c_ptr, "C%d", &temp_cat) != 1) {
        return 0; // 类别解析失败
    }
    
    // 验证解析结果有效性
    if(temp_dist <= 0.0 || temp_cat < 0) {
        return 0; // 解析值不合理
    }
    
    // 解析成功，设置输出参数
    *distance = temp_dist;
    *category = temp_cat;
    
    return 1; // 解析成功
}

/*************************************************************
函数名称：ceju_jiaqu()
功能介绍：从串口获取距离，换算成逆运动学坐标，执行测距夹取动作
函数参数：无
返回值：  无  
*************************************************************/
void ceju_jiaqu(void) {
    static float dis = 0.0;
    static u32 last_state_change = 0;
    
    if(group_do_ok == 0) return; //有动作执行，直接返回

    // 检查串口是否接收到数据
    if(uart1_get_ok) {
        float distance = 0.0;
        int cat = -1;
        
        if(parse_data(&distance, &cat)) {
            dis = distance;
            category = cat;
            
            // 根据接收到的数据触发动作
            if(dis != 0.0 && category != -1) {
                kms_y = dis*10 + 120;
                sprintf((char*)cmd_return, "dis=%f cm, category=%d, kms_y=%d\r\n", dis, category, kms_y);
                uart1_send_str(cmd_return); // 发送目标位置和类别信息
                // 重置状态机，开始自动执行
                carry_step = 1;
                last_state_change = millis();
            }
        }
        
        uart1_get_ok = 0; // 清除接收标志
    }
    
    // 自动执行状态机的下一步
    if(carry_step != 0 && millis() - last_state_change >= 2500) { // 防抖
        carry_wood();
        last_state_change = millis();
    }
}

void carry_wood(void) {	
	//张开爪子
	if(carry_step == 1){
		set_servo(5,1200,1000);		
		//mdelay(500);
		carry_step = 2;
		uart1_send_str("carry_wood step 1");
	//运行到目标位置
	}else if(carry_step == 2){
        uart1_send_str("carry_wood step 2");
		if(kinematics_move(0,kms_y,15,1500)){
            uart1_send_str("move test");
			beep_on_times(1,100);
			//mdelay(2000);	
			carry_step = 3;
            uart1_send_str("carry_wood step 2");							
		}else{
			carry_step = 0;
			return;
		}		
	//夹取
	}else if(carry_step == 3){
        uart1_send_str("pick test");
		set_servo(5,1800,1000);		
		//mdelay(500);
		carry_step = 4;
	//分类
	}else if(carry_step == 4){
        uart1_send_str("up test");
        set_servo(1,1200,1500);
        carry_step = 5;
    }else if(carry_step == 5){
        uart1_send_str("classify test");
		switch(category){
			case 0:
				//可回收
				do_group_once(39);
				break;
			case 1:
				//厨余垃圾
				do_group_once(40);
				break;
			case 2:
				//有害垃圾
				do_group_once(41);
				break;
			case 3:
				//其他垃圾
				do_group_once(42);
				break;
		}
		carry_step = 6;
	}else if(carry_step == 6){
        uart1_send_str("drop test");
        set_servo(5,1200,1000);
        carry_step = 7;
    }else if(carry_step == 7){
		parse_group_cmd("$RST!");
		if(group_do_ok == 1){
			carry_step = 0;						
		}
	}
}