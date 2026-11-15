OpenCode（sst/opencode）は、**75以上のLLMプロバイダに対応するオープンソースのAIコーディングエージェントCLI**です。Google Gemini APIの他に、無料または無料枠で利用できる主要なプロバイダは以下の通りです。

---

## 1. OpenCode Zen（APIキー不要・推奨）

OpenCodeが自社運営するAPI中継サービスで、**APIキーもアカウントも不要**で無料モデルが使えます。

| 無料モデル例 | 備考 |
|---|---|
| GLM-5.3-Flash Free | 期間限定無料 |
| DeepSeek V4.1 Flash Free | 期間限定無料 |
| MiMo-V2.6-Flash | 期間限定無料 |
| Big Pickle | 常時無料枠 |
| Iceman Free | 常時無料枠 |

> OpenCode内でしか使えないキャンペーン中の無料モデルも含まれます。  
> [joingonka.ai](https://joingonka.ai/ja/knowledge/opencode/) [note.com](https://note.com/bright_hosta5/n/nb1f00d9fd603)

---

## 2. 集約型プロバイダ（OpenRouter / Kilo Gateway）

### OpenRouter
- **無料ティアあり**: レート制限付きで複数モデルを無料利用可能
- Gemma 4 31B IT 等が 20リクエスト/分・50リクエスト/日で利用可能
- OpenCodeから `/connect` で接続

### Kilo Gateway
- **APIキー不要**: 匿名でIPごとに1時間200リクエストまで無料
- `:free` で終わるモデルIDが対象
- [usemagpie.ai](https://usemagpie.ai/docs/ja/free-models)

---

## 3. クラウド・GPUプロバイダ（無料枠・プレビュー）

| プロバイダ | 特徴 |
|---|---|
| **NVIDIA NIM** | プレビュー版エンドポイントが無料で提供中 |
| **Cloudflare** | Workers AIの無料枠あり |
| **Groq** | 高速推論、無料クレジットあり |
| **Cerebras** | 専用HWによる高速推論、無料枠あり |

---

## 4. 中国・アジア圏プロバイダ

| プロバイダ | 特徴 |
|---|---|
| **SenseNova** | 商湯科技（SenseTime）のモデル |
| **SiliconFlow** | 中国系モデル集約プロバイダ |
| **ModelScope** | Alibaba系のモデルハブ |
| **Agnes AI** | 中国語・日本語対応モデル |
| **Kimi K3** | Moonshot AI、ZenMux経由で無料利用可能 |
| **GLM-5.2** | Zhipu AI、ZenMux経由で無料利用可能 |

---

## 5. その他の主要プロバイダ（無料枠あり）

| プロバイダ | 無料枠の内容 |
|---|---|
| **Pollinations** | 無料の画像・テキスト生成API |
| **Mistral** | 無料ティアあり |
| **Cohere** | 無料クレジットあり |
| **DeepSeek** | 公式APIに無料枠あり |

---

## 6. ローカル実行（完全無料・API不要）

| 方法 | 説明 |
|---|---|
| **Ollama** | ローカルPCでLLMを実行し、OpenCodeからOpenAI互換APIとして接続 |
| **llama.cpp** | 同上、自前のGPU/CPUで推論 |

---

## OpenCodeでの接続方法

```bash
# OpenRouterに接続
opencode /connect openrouter

# OpenCode Zenはデフォルトで利用可能（キー不要）
# 無料モデル一覧の確認
opencode /models

# ローカルのOllamaに接続
opencode /connect ollama
```

---

## まとめ

> Google Gemini API以外に、OpenCodeで無料利用できる主なプロバイダは **OpenCode Zen（キー不要）・OpenRouter・Kilo Gateway・NVIDIA NIM・Cloudflare・Groq・Ollama（ローカル）** 等があります。  
> 特に **OpenCode Zen** はAPIキー不要で即座に使い始められるため、OpenCode入門時の最も手軽な選択肢です。ただし、Zenの無料モデルの一部は期間限定キャンペーンである点に注意が必要です。