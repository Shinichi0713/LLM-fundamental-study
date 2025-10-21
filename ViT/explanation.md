# ViT

## 目的

ViTの構造を理解して、モデル構築→実際に学習して出力を確認する。

```objectivec
┌──────────────┐
│  画像(224×224×3) │
└──────────────┘
         ↓ パッチ分割 (16×16)
┌───────────────────────┐
│ 14×14=196 パッチ (各 768次元ベクトル) │
└───────────────────────┘
         ↓ 線形埋め込み
┌───────────────────────┐
│ [CLS] + 196トークン (各768次元) │
└───────────────────────┘
         ↓ 位置エンコーディング追加
         ↓
┌───────────────────────┐
│ Transformer Encoder (12層など) │
└───────────────────────┘
         ↓
     [CLS]トークンの出力 → 分類など

```

## エンベディング

Position Embedding
positional encoding
位置埋め込み
ViTの構造に含まれるSelf-Attentionは，画像の情報を並列に計算する．その際に，ここと，ここが近い位置にあることや遠い位置にあることは計算に考慮してしない．近くの位置でも遠くの位置でも関係なく計算する．そのため，画像の位置情報を完全に無視して計算することになる．

これは，性能低下になる可能性があるため，位置情報を付与する．位置情報の数値を，各トークンに加算される．位置情報はクラストークンにも加算される．

## あるべき学習曲線

https://wandb.ai/arutema47/cifar10-challange/reports/Untitled-Report--VmlldzoxNjU3MTU2?accessToken=3y3ib62e8b9ed2m2zb22dze8955fwuhljl5l4po1d5a3u9b7yzek1tz7a0d4i57r

![1760856380717](image/explanation/1760856380717.png)


## ViTの構成

patch: 画像の並び替え

Transformer：Attentionブロック×6


```
ViT(
  (to_patch_embedding): Sequential(
    (0): Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=4, p2=4)
    (1): Linear(in_features=48, out_features=192, bias=True)
  )
  (dropout): Dropout(p=0.1, inplace=False)
  (transformer): Transformer(
    (layers): ModuleList(
      (0-5): 6 x ModuleList(
        (0): PreNorm(
          (norm): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
          (fn): Attention(
            (attend): Softmax(dim=-1)
            (to_qkv): Linear(in_features=192, out_features=576, bias=False)
            (to_out): Sequential(
              (0): Linear(in_features=192, out_features=192, bias=True)
              (1): Dropout(p=0.1, inplace=False)
            )
          )
        )
        (1): PreNorm(
          (norm): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
          (fn): FeedForward(
            (net): Sequential(
              (0): Linear(in_features=192, out_features=512, bias=True)
              (1): GELU(approximate='none')
              (2): Dropout(p=0.1, inplace=False)
              (3): Linear(in_features=512, out_features=192, bias=True)
              (4): Dropout(p=0.1, inplace=False)
            )
          )
        )
      )
    )
  )
  (to_latent): Identity()
  (mlp_head): Sequential(
    (0): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
    (1): Linear(in_features=192, out_features=10, bias=True)
  )
)
```


良くないViT

```
ViT(
  (patch_embed): PatchEmbedding(
    (proj): Conv2d(3, 192, kernel_size=(4, 4), stride=(4, 4))
  )
  (pos_drop): Dropout(p=0.1, inplace=False)
  (blocks): ModuleList(
    (0-5): 6 x TransformerEncoderBlock(
      (norm1): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
      (attn): MultiheadAttention(
        (out_proj): NonDynamicallyQuantizableLinear(in_features=192, out_features=192, bias=True)
      )
      (norm2): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
      (mlp): MLP(
        (fc1): Linear(in_features=192, out_features=512, bias=True)
        (fc2): Linear(in_features=512, out_features=192, bias=True)
        (act): GELU(approximate='none')
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (dropout): Dropout(p=0.1, inplace=False)
    )
  )
  (norm): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
  (head): Linear(in_features=192, out_features=10, bias=True)
)
```
