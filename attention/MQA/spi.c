#include "spi.h"

// ============================================
//  以下のレジスタ・ビット定義は使用するMCUに合わせて変更してください
// ============================================
#define SPI_STAT_REG     (*(volatile uint32_t *)0x40000000)  // 状態レジスタ
#define SPI_DATA_REG     (*(volatile uint32_t *)0x40000004)  // データレジスタ
#define SPI_CTRL_REG     (*(volatile uint32_t *)0x40000008)  // 制御レジスタ

#define STAT_TX_EMPTY    (1U << 0)  // 送信バッファ空フラグ
#define STAT_RX_FULL     (1U << 1)  // 受信バッファ満フラグ
#define STAT_BUSY        (1U << 2)  // バスビジーフラグ

#define CS_PIN_PORT      (*(volatile uint32_t *)0x40001000)
#define CS_PIN_BIT       (1U << 3)

void SPI_Init(void)
{
    // クロック供給、GPIO代替機能設定などは別途実施してください
    
    // SPI制御レジスタ設定例（マスター、モード0、分周比設定など）
    // この値は実際のMCUレジスタ仕様に合わせて変更が必要です
    SPI_CTRL_REG = 0x00000015;  // 例: Master, CPOL=0, CPHA=0, DIV=16
}

void SPI_ChipSelect(bool select)
{
    if (select) {
        CS_PIN_PORT &= ~CS_PIN_BIT;  // CS Low（有効）
    } else {
        CS_PIN_PORT |= CS_PIN_BIT;   // CS High（無効）
    }
}

uint8_t SPI_TransferByte(uint8_t tx_data)
{
    // 送信バッファに書き込み
    SPI_DATA_REG = tx_data;

    // 受信完了（または送信完了/バスアイドル）待ち
    while ((SPI_STAT_REG & STAT_RX_FULL) == 0) {
        // ビジーウェイト。割り込み版にする場合はここでスレッド譲渡など
    }

    // 受信データを読み出し
    return (uint8_t)(SPI_DATA_REG & 0xFF);
}

void SPI_Send(const uint8_t *data, uint16_t length)
{
    for (uint16_t i = 0; i < length; i++) {
        SPI_TransferByte(data[i]);
    }
}

void SPI_Receive(uint8_t *buffer, uint16_t length)
{
    for (uint16_t i = 0; i < length; i++) {
        buffer[i] = SPI_TransferByte(0xFF);  // ダミーデータを送信しながら受信
    }
}

#include "spi.h"

int main(void)
{
    uint8_t tx_buf[] = {0x01, 0x80, 0x00};  // 例: センサーのレジスタ読み出しコマンド
    uint8_t rx_buf[3] = {0};

    SPI_Init();

    SPI_ChipSelect(true);               // CS = Low（通信開始）
    SPI_Send(tx_buf, sizeof(tx_buf));   // コマンド送信
    SPI_ChipSelect(false);              // CS = High（通信終了）

    // 受信が必要な場合
    SPI_ChipSelect(true);
    SPI_Receive(rx_buf, 3);             // 3バイト受信
    SPI_ChipSelect(false);

    while (1) {
        // メイン処理
    }

    return 0;
}