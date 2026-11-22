---
title: "AgentScope 2.0 で始める AI エージェント管理入門"
emoji: "🤖"
type: "tech"
topics: ["agentscope", "aiagent", "llm", "python", "multiagent"]
published: false
---

## AgentScope とは

[AgentScope](https://github.com/agentscope-ai/agentscope) は、Alibaba 系のチームが開発している **AI エージェントを構築・実行・管理するためのオープンソースフレームワーク** です。ライセンスは Apache 2.0、Python 3.11 以上で動作します。

「エージェントを管理する」というと、タスク管理ツールをイメージするかもしれませんが、AgentScope が管理対象にしているのはもう少し技術的なレイヤーです。

| 管理したいもの | AgentScope の対応 |
| --- | --- |
| エージェントの思考・行動ループ | ReAct ベースの `Agent`（`reply` / `reply_stream`） |
| ツール（関数・MCP・Skill） | `Toolkit` による登録・JSON Schema 化・ディスパッチ |
| 実行権限 | ルール・モードによる ALLOW / DENY / ASK と人間承認（HITL） |
| 実行環境 | Workspace / Sandbox（ローカル、Docker、E2B、K8s など） |
| 会話・セッション | コンテキスト圧縮、長期記憶（Mem0 / ReMe）、マルチテナント・マルチセッション |
| 運用・公開 | Agent Service（FastAPI バックエンド + Web UI） |

つまり **「1 体のエージェントを書くライブラリ」から「複数ユーザー・複数セッションでエージェントを動かす基盤」までを一気通貫で提供するフレームワーク** です。

> 本記事のコードは **AgentScope 2.0.12**（記事執筆時点の安定版）を前提にしています。2.0 は 1.x から破壊的変更が入っているため、古い記事のコードはそのまま動きません。必ずバージョンを確認してください。

### 2.0 の特徴

公式ドキュメントでは、2.0 の特徴が次の 4 つに整理されています。

- **Secure**: ツールごとの権限チェック、人間承認、サンドボックスの 3 層で保護
- **Efficient**: ツールの性質に応じて並列 / 逐次実行を自動で切り替え
- **Flexible**: ミドルウェアでエージェントの動作を外から差し替え可能
- **Complete**: SDK、Web UI、マルチテナント・マルチセッション基盤、分散デプロイまでカバー

LangGraph や AutoGen と比べると、「ツール実行の承認・隔離」と「そのまま Web アプリとして公開できる Service 層」が最初から揃っている点が AgentScope らしさです。

## 導入方法

### 1. Python 3.11 以上を用意する

```bash
python --version
# Python 3.11.x 以降であること
```

AgentScope 自体は 3.11 以上が必要です。ここではプロジェクト用の仮想環境を作り、公式が推奨する `uv` でインストールします。

```bash
# uv が無い場合
python -m pip install uv

uv venv
source .venv/bin/activate   # Windows PowerShell の場合: .venv\Scripts\activate
```

### 2. AgentScope をインストールする

安定版を明示して入れるのがおすすめです（1.x が入ると API が違うため）。

```bash
uv pip install "agentscope==2.0.12"
```

追加機能（モデル API・ツールの依存一式）が必要な場合はソースから入れます。

```bash
git clone --branch v2.0.12 https://github.com/agentscope-ai/agentscope
cd agentscope
uv pip install -e ".[full]"
```

### 3. インストール確認

```bash
python -c "import agentscope; print(agentscope.__version__)"
# 2.0.12
```

### 4. API キーを設定する

この記事では OpenAI を使います。お好みで DashScope / Gemini / Anthropic / DeepSeek / Ollama などに差し替えられます。

```bash
# macOS / Linux
export OPENAI_API_KEY="sk-..."

# Windows PowerShell
$env:OPENAI_API_KEY = "sk-..."
```

鍵をソースに書かず、必ず環境変数経由で渡してください。AgentScope では「認証情報（Credential）」と「モデル（ChatModel）」を別に作るのが 2.0 の流儀です。

## はじめてのエージェントを動かす

まずはツールなしのシンプルなエージェントです。`hello_agent.py` として保存します。

```python
import asyncio
import os

from agentscope.agent import Agent
from agentscope.credential import OpenAICredential
from agentscope.message import UserMsg
from agentscope.model import OpenAIChatModel


async def main() -> None:
    agent = Agent(
        name="Friday",
        system_prompt="あなたは簡潔に答えるアシスタントです。",
        model=OpenAIChatModel(
            credential=OpenAICredential(api_key=os.environ["OPENAI_API_KEY"]),
            model="gpt-4.1",
        ),
    )

    reply = await agent.reply(
        UserMsg(name="user", content="こんにちは、あなたは誰？"),
    )

    # AssistantMsg.content はブロックのリスト。
    # テキストだけを取り出すヘルパーもある。
    print(reply.get_text_content())


if __name__ == "__main__":
    asyncio.run(main())
```

```bash
python hello_agent.py
# こんにちは！私は Friday です。何かお手伝いできることはありますか？
```

ここで出てくる要素を整理しておきます。

- `Credential` : API キーなどプロバイダー固有の認証情報
- `ChatModel` : モデル名・ストリーミング有無・パラメータを保持するラッパー
- `UserMsg` / `AssistantMsg` : 送信・応答メッセージ。中身は `TextBlock` などの**ブロック列**
- `Agent.reply()` : 推論とツール実行を回して、最終メッセージを返す
- `Agent.reply_stream()` : 同じ処理をイベント単位で逐次返す（UI 向け）

`reply()` は「完成した答え」が欲しいとき、`reply_stream()` は「考えている途中も表示したい」ときに使い分けます。

### ストリーミングで表示する

Web UI や CLI を作るなら `reply_stream()` を使います。イベントの種類で処理を分岐させます。

```python
from agentscope.event import TextBlockDeltaEvent, ToolCallStartEvent

async for event in agent.reply_stream(
    UserMsg(name="user", content="1 から 5 まで数えて"),
):
    if isinstance(event, TextBlockDeltaEvent):
        # モデルが生成した text の差分（event.delta は str）
        print(event.delta, end="", flush=True)
    elif isinstance(event, ToolCallStartEvent):
        # これからツールを呼ぶという通知
        print(f"\n[tool] {event.tool_call_name}", flush=True)
```

イベントは `start → delta → end` の組で流れてきます。全部を `Msg.append_event(event)` に渡せば、ストリームから完全なメッセージを組み立て直せるので、SSE 配信とフロントエンド描画が自然に繋がります。

> イベントのフィールド名はバージョンで変わり得るので、実装時は使用しているバージョンのドキュメントを確認してください。まずは `launch_console()` で実際のイベントを眺めるのが簡単です。

## ツールを追加してみる（最初の「操作」）

エージェントに「現在時刻」を聞くと、モデルは自分の学習データの日付を答えてしまいます。そこで Python 関数をツールとして渡してみましょう。

```python
from datetime import datetime, timedelta, timezone

from agentscope.tool import FunctionTool, Toolkit


def now_jst() -> str:
    """現在の日本時刻（JST）を返す。時刻を聞かれたときに使う。"""
    return datetime.now(timezone(timedelta(hours=9))).strftime(
        "%Y-%m-%d %H:%M:%S JST",
    )


toolkit = Toolkit(
    tools=[
        FunctionTool(now_jst, is_read_only=True),
    ],
)
```

`FunctionTool` は関数名をツール名に、docstring を説明に、型ヒントを引数の JSON Schema に自動変換してくれます。あとは先ほどの `Agent` に `toolkit=toolkit` を渡すだけです。

```python
agent = Agent(
    name="Friday",
    system_prompt="あなたは簡潔に答えるアシスタントです。",
    model=OpenAIChatModel(
        credential=OpenAICredential(api_key=os.environ["OPENAI_API_KEY"]),
        model="gpt-4.1",
    ),
    toolkit=toolkit,
)

reply = await agent.reply(UserMsg(name="user", content="今の日本時間は？"))
```

```text
2026-11-22 15:04:31 JST です。
```

### ここで `is_read_only=True` が重要

AgentScope は **すべてのツール呼び出しを権限システムでチェック** します。

- `is_read_only=True` のツールは毎回の呼び出しで読み取り専用と判定され、どのモードでも自動 ALLOW
- 省略するとデフォルトは ASK（ユーザー承認待ち）になり、素の `reply()` では応答が止まってしまう
- 処理を伴うツールは `ToolBase` を継承して `check_permissions()` を実装するか、承認イベントを拾う UI を使う

権限モードは `DEFAULT` / `ACCEPT_EDITS` / `EXPLORE` / `BYPASS` / `DONT_ASK` が用意されています。無人実行なら `BYPASS` より `DONT_ASK` の方が安全です（deny ルールを効かせつつ、ユーザー確認を ASK ではなく DENY に変換してくれます）。

```python
from agentscope.agent import AgentState
from agentscope.permission import PermissionContext, PermissionMode

agent = Agent(
    name="explorer",
    system_prompt="読み取りだけ行う調査役です。",
    model=model,
    state=AgentState(
        permission_context=PermissionContext(mode=PermissionMode.EXPLORE),
    ),
)
```

## ターミナルで対話する（いちばん簡単な入門操作）

自分でイベント処理を書く前に、公式のコンソールで動かすのがおすすめです。ツール呼び出しの承認（`y` / `n`）、`Ctrl+C` による割り込み、`exit` / `quit` での終了を全部面倒みてくれます。

```python
import asyncio
import os

from agentscope.agent import Agent
from agentscope.console import launch_console
from agentscope.credential import OpenAICredential
from agentscope.model import OpenAIChatModel


async def main() -> None:
    agent = Agent(
        name="Friday",
        system_prompt="あなたは役に立つアシスタントです。",
        model=OpenAIChatModel(
            credential=OpenAICredential(api_key=os.environ["OPENAI_API_KEY"]),
            model="gpt-4.1",
        ),
    )
    await launch_console(agent)


asyncio.run(main())
```

```text
Chat with the agent. Type 'exit' (or Ctrl+D) to quit.

user> 今日の東京の天気を調べて
...
Allow 'WebFetch'? [y]es / [N]o / [a]lways  y
```

`verbosity="debug"` を渡すと、ふだんは見えないライフサイクルイベントまで表示されます。エージェントの内部挙動を学ぶときに便利です。

## 組み込みツールと MCP

`Toolkit` には最初から便利なツールが揃っています。

| ツール | 内容 |
| --- | --- |
| `Bash` / `PowerShell` | シェルコマンド実行 |
| `Read` / `Write` / `Edit` | ファイル操作（書き込み前に `Read` が必須） |
| `Glob` / `Grep` | ファイル名・内容検索 |
| `TaskCreate` / `TaskGet` / `TaskList` / `TaskUpdate` | 計画タスクの管理 |
| `AskUser` | ユーザーに選択肢を提示（HITL） |

```python
import os

from agentscope.tool import Bash, Grep, Glob, PowerShell, Read, Toolkit

# Windows では PowerShell、それ以外では Bash を使う
tools = [PowerShell(), Read(), Glob(), Grep()] if os.name == "nt" else [Bash(), Read(), Glob(), Grep()]
toolkit = Toolkit(tools=tools)
```

MCP サーバーも `Toolkit` に登録できます。STDIO / HTTP 接続に対応しており、外部のツールをエージェントのツールとして扱えます。Skill（`SKILL.md` 形式の手順書）も同じく Toolkit 経由で読み込みます。

> Windows の `PowerShell` ツールは、現状すべてのコマンドで ASK になります（静的な安全判定が未実装のため）。コンソールなしで自動実行したい場合は、allow ルールを明示的に設定してください。

## Web アプリとして公開する（Agent Service）

SDK で作ったエージェントは、そのままマルチテナント・マルチセッションな Web アプリとして動かせます。2.x では旧来の Runtime + Studio を組み立てるより、同梱の Agent Service と Web UI を使うのが公式の推奨です。

```bash
git clone -b main https://github.com/agentscope-ai/agentscope.git
cd agentscope/examples/agent_service
python main.py
```

別ターミナルでフロントエンドを起動します。

```bash
cd agentscope/examples/web_ui
pnpm install
pnpm dev
```

Agent Service 側には次の機能が最初から入っています。

- ユーザー・セッションの分離と永続化（SQL / NoSQL、Redis など差し替え可）
- Agent Team（リーダーがワーカーを生成して協調）
- スケジュール実行、バックグラウンドタスク、エージェントのウェイクアップ
- MCP / Skill Hub、チャンネル連携（Feishu / Discord など）
- 認証は利用側アプリの責務（サンプルの user-header は本番では差し替える）

> サンプルコードは **インストールした SDK と同じチェックアウト** のものを使ってください。バージョンがずれると API が合いません。

## ハマりどころ

1. **1.x の記事を参照してしまう**
   2.0 は Agent / Message / Tool / State / Deployment すべてが変わっています。ドキュメントのバージョン selector を `2.0.12` に合わせましょう。
2. **Python 3.10 でインストールできない**
   2.0 は 3.11 以上が必須です。
3. **ツール呼び出しで応答が止まる**
   `FunctionTool` の既定は ASK です。`is_read_only=True` にするか、承認イベントを処理する `launch_console()` / Web UI を使いましょう。
4. **API キー周りのエラー**
   `Credential` と `ChatModel` の providers を揃えてください（例: `OpenAICredential` + `OpenAIChatModel`）。Ollama なら `OllamaCredential` + `OllamaChatModel` です。

## まとめ

AgentScope 2.0 は、「エージェントを作る」だけでなく「安全に動かし続ける」ところまで面倒を見てくれるフレームワークです。

まずはこの順番で触ってみるのが最短です。

1. `uv pip install "agentscope==2.0.12"`
2. `Agent` + `reply()` で 1 往復
3. `FunctionTool` でツールを 1 つ追加
4. `launch_console()` で対話しながらイベントを眺める
5. 必要になったら Agent Service / 権限モード / Sandbox へ進む

公式ドキュメントには各パーツの詳細な例が揃っているので、次はそこから自分のユースケースに近いページを読んでみてください。

## 参考リンク

- [AgentScope GitHub](https://github.com/agentscope-ai/agentscope)
- [AgentScope ドキュメント（安定版）](https://docs.agentscope.io/stable/en/index)
- [Quickstart](https://docs.agentscope.io/stable/en/quickstart)
- [AgentScope 2.0 FAQ](https://agentscope.io/blog/agentscope-faq)
- [AgentScope 1.0 論文 (arXiv:2508.16279)](https://arxiv.org/abs/2508.16279)
- [AgentScope 論文 (arXiv:2402.14034)](https://arxiv.org/abs/2402.14034)
