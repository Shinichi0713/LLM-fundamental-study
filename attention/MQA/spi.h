#ifndef SPI_H
#define SPI_H

#include <stdint.h>
#include <stdbool.h>

// SPI初期化（モード0、MSBファースト、分周比などは適宜変更）
void SPI_Init(void);

// 1バイト送受信（全二重、マスター）
// 送信しながら受信した1バイトを返す
uint8_t SPI_TransferByte(uint8_t tx_data);

// 複数バイト送信
void SPI_Send(const uint8_t *data, uint16_t length);

// 複数バイト受信（ダミー送信しながら受信）
void SPI_Receive(uint8_t *buffer, uint16_t length);

// CS（チップセレクト）制御（低アクティブ想定）
void SPI_ChipSelect(bool select);

#endif /* SPI_H */