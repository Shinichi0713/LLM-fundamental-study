#ifndef MOTOR_H
#define MOTOR_H

#include <stdint.h>

// モーター初期化（PWM・GPIO設定）
void Motor_Init(void);

// 左右モーター速度設定
// speed_l, speed_r: -1000 ～ +1000 （正が前進、負が後退）
void Motor_SetSpeed(int16_t speed_l, int16_t speed_r);

// モーター停止（ブレーキ or 慣性停止）
void Motor_Stop(void);

#endif