#include "motor.h"

// 仮のPWMレジスタ定義（使用するMCUに合わせて変更）
#define PWM_LEFT_REG  (*(volatile uint32_t *)0x40002000)
#define PWM_RIGHT_REG (*(volatile uint32_t *)0x40002004)
#define MAX_PWM       1000

void Motor_Init(void)
{
    // PWMクロック、GPIO、タイマー設定は別途実施
}

void Motor_SetSpeed(int16_t speed_l, int16_t speed_r)
{
    // 飽和処理
    if (speed_l > MAX_PWM)  speed_l = MAX_PWM;
    if (speed_l < -MAX_PWM) speed_l = -MAX_PWM;
    if (speed_r > MAX_PWM)  speed_r = MAX_PWM;
    if (speed_r < -MAX_PWM) speed_r = -MAX_PWM;

    // 回転方向制御（GPIOなど）とPWM値設定
    // 実際のレジスタ書き込みはMCU仕様に合わせて実装
    PWM_LEFT_REG  = (uint32_t)speed_l;
    PWM_RIGHT_REG = (uint32_t)speed_r;
}

void Motor_Stop(void)
{
    Motor_SetSpeed(0, 0);
}

#include "sensor.h"
#include "spi.h"

// IMUレジスタ定義例（仮想的なIMU）
#define IMU_REG_WAVE_H  0x20  // ヨー角高位バイト読み出し
#define IMU_REG_WAVE_L  0x21  // ヨー角低位バイト読み出し

bool Sensor_Init(void)
{
    SPI_Init();
    // IMUのキャリブレーション待ちなどが必要ならここで実施
    return true;
}

int16_t Sensor_ReadYaw(void)
{
    uint8_t tx_buf[2];
    uint8_t rx_buf[2];
    int16_t yaw;

    SPI_ChipSelect(true);

    // 高位バイト読み出し
    tx_buf[0] = IMU_REG_WAVE_H | 0x80;  // 読み出しビットセット（IMUによる）
    tx_buf[1] = 0x00;
    SPI_Send(tx_buf, 2);
    SPI_Receive(rx_buf, 2);
    yaw = ((int16_t)rx_buf[1]) << 8;

    SPI_ChipSelect(false);

    // 小さなディレイが必要な場合はここに挿入（IMU仕様による）
    for (volatile int i = 0; i < 100; i++);

    SPI_ChipSelect(true);

    // 低位バイト読み出し
    tx_buf[0] = IMU_REG_WAVE_L | 0x80;
    tx_buf[1] = 0x00;
    SPI_Send(tx_buf, 2);
    SPI_Receive(rx_buf, 2);
    yaw |= rx_buf[1];

    SPI_ChipSelect(false);

    return yaw;  // 符号付き16bitとして返す
}


#include "robot_control.h"
#include "sensor.h"
#include "motor.h"
#include <math.h>

static float pid_integral = 0.0f;
static float pid_last_error = 0.0f;
static int16_t target_yaw = 0;  // 目標方位角（直進なら0度とする）

void RobotControl_Init(void)
{
    Sensor_Init();
    Motor_Init();

    // 起動時の現在方位を目標方位に設定（直進基準）
    target_yaw = Sensor_ReadYaw();
    pid_integral = 0.0f;
    pid_last_error = 0.0f;
}

void RobotControl_Update(void)
{
    // 1. センサー読み取り
    int16_t current_yaw = Sensor_ReadYaw();

    // 2. 方位偏差計算（-18000 ～ +18000度×100の範囲に正規化）
    int32_t error_raw = (int32_t)target_yaw - (int32_t)current_yaw;
    while (error_raw > 18000)  error_raw -= 36000;
    while (error_raw < -18000) error_raw += 36000;
    float error = (float)error_raw / 100.0f;  // 度単位に変換

    // 3. PID計算
    pid_integral += error * CONTROL_DT;
    float derivative = (error - pid_last_error) / CONTROL_DT;
    float output = (PID_KP * error) + (PID_KI * pid_integral) + (PID_KD * derivative);
    pid_last_error = error;

    // 4. モーター速度指令値計算（差動駆動）
    // 方位が右に傾いていれば左を速く、左に傾いていれば右を速く
    int16_t speed_l = BASE_SPEED - (int16_t)output;
    int16_t speed_r = BASE_SPEED + (int16_t)output;

    // 5. モーター駆動
    Motor_SetSpeed(speed_l, speed_r);
}