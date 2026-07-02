# models.py
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class Document(BaseModel):
    id: str
    filename: str
    content: str
    uploaded_at: datetime

class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime

class ChatSession(BaseModel):
    id: str
    messages: List[ChatMessage]
    created_at: datetime

class QualityCheckRequest(BaseModel):
    document_id: str
    check_type: str  # "code", "doc", "general"

class QualityCheckResult(BaseModel):
    document_id: str
    score: int
    feedback: str
    violations: List[str]


# storage.py
import json
import os
from typing import List, Optional
from models import Document, ChatSession

class DocumentStorage:
    def __init__(self, base_path: str = "./data/documents"):
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)

    def save(self, document: Document) -> None:
        filepath = os.path.join(self.base_path, f"{document.id}.json")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(document.json())

    def get(self, doc_id: str) -> Optional[Document]:
        filepath = os.path.join(self.base_path, f"{doc_id}.json")
        if not os.path.exists(filepath):
            return None
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return Document(**data)

    def list_all(self) -> List[Document]:
        docs = []
        for filename in os.listdir(self.base_path):
            if filename.endswith(".json"):
                doc = self.get(filename[:-5])  # remove .json
                if doc:
                    docs.append(doc)
        return docs

class ChatStorage:
    def __init__(self, base_path: str = "./data/chats"):
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)

    def save(self, session: ChatSession) -> None:
        filepath = os.path.join(self.base_path, f"{session.id}.json")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(session.json())

    def get(self, session_id: str) -> Optional[ChatSession]:
        filepath = os.path.join(self.base_path, f"{session_id}.json")
        if not os.path.exists(filepath):
            return None
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return ChatSession(**data)
    
# services.py
import openai
from typing import List
from models import Document, ChatMessage, QualityCheckResult

class LLMService:
    def __init__(self, api_key: str, model: str = "gpt-4"):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model

    def chat(self, messages: List[ChatMessage]) -> str:
        # ChatMessageをOpenAIの形式に変換
        openai_messages = [
            {"role": "system", "content": "あなたは親切で正確なアシスタントです。"}
        ]
        for msg in messages:
            openai_messages.append({"role": msg.role, "content": msg.content})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=openai_messages,
            temperature=0.1,
        )
        return response.choices[0].message.content

    def summarize_document(self, document: Document) -> str:
        prompt = f"""
以下のドキュメントを要約してください。
タイトル: {document.filename}
内容:
{document.content}

出力形式:
- 要約（3〜5行）
- キーポイント（箇条書き）
"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "あなたは優秀な要約アシスタントです。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
        )
        return response.choices[0].message.content

    def check_quality(self, document: Document, check_type: str) -> QualityCheckResult:
        if check_type == "code":
            prompt = f"""
以下のコードの品質を評価し、改善提案をしてください。
評価観点:
- セキュリティ
- パフォーマンス
- 保守性
- 可読性

コード:

出力形式:
- 評価サマリー（1〜3行）
- 改善提案（箇条書き）
- リスク指摘（あれば）
"""
        elif check_type == "doc":
            prompt = f"""
以下のドキュメントの品質を評価し、改善提案をしてください。
評価観点:
- 明確さ
- 完全性
- 一貫性
- 実装への導きやすさ

ドキュメント:

出力形式:
- 評価サマリー（1〜3行）
- 改善提案（箇条書き）
"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "あなたはソフトウェア品質管理の専門家です。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
        )
        feedback = response.choices[0].message.content

        # 簡易スコアリング（例）
        score = 10
        negative_keywords = ["危険", "問題", "改善必須", "重大"]
        for kw in negative_keywords:
            if kw in feedback:
                score -= 1
        score = max(0, score)

        # ルール違反の簡易検出（例）
        violations = []
        if "password" in document.content.lower() and "平文" in document.content:
            violations.append("パスワードが平文で扱われている可能性があります")

        return QualityCheckResult(
            document_id=document.id,
            score=score,
            feedback=feedback,
            violations=violations
        )

# main.py
import os
import uuid
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from models import Document, ChatMessage, ChatSession, QualityCheckRequest
from storage import DocumentStorage, ChatStorage
from services import LLMService

# 環境変数からAPIキーを取得（本番ではシークレット管理を推奨）
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required")

app = FastAPI(title="LLM業務支援アプリ")

# CORS設定（フロントエンドからアクセス可能にする）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 本番では適切に制限する
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ストレージ・サービスの初期化
doc_storage = DocumentStorage()
chat_storage = ChatStorage()
llm_service = LLMService(api_key=OPENAI_API_KEY)

# チャットセッション管理（簡易版）
active_sessions = {}

@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    content = (await file.read()).decode("utf-8")
    doc_id = str(uuid.uuid4())
    document = Document(
        id=doc_id,
        filename=file.filename,
        content=content,
        uploaded_at=datetime.now()
    )
    doc_storage.save(document)
    return {"document_id": doc_id, "filename": file.filename}

@app.get("/documents")
async def list_documents():
    docs = doc_storage.list_all()
    return [{"id": doc.id, "filename": doc.filename, "uploaded_at": doc.uploaded_at} for doc in docs]

@app.get("/documents/{doc_id}")
async def get_document(doc_id: str):
    doc = doc_storage.get(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc

@app.post("/documents/{doc_id}/summarize")
async def summarize_document(doc_id: str):
    doc = doc_storage.get(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    summary = llm_service.summarize_document(doc)
    return {"summary": summary}

@app.post("/quality/check")
async def check_quality(request: QualityCheckRequest):
    doc = doc_storage.get(request.document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    result = llm_service.check_quality(doc, request.check_type)
    return result

@app.post("/chat/{session_id}")
async def chat(session_id: str, message: str):
    # セッションの取得または作成
    if session_id not in active_sessions:
        session = ChatSession(
            id=session_id,
            messages=[],
            created_at=datetime.now()
        )
        active_sessions[session_id] = session
    else:
        session = active_sessions[session_id]

    # ユーザーメッセージの追加
    user_msg = ChatMessage(
        role="user",
        content=message,
        timestamp=datetime.now()
    )
    session.messages.append(user_msg)

    # LLMによる応答生成
    assistant_content = llm_service.chat(session.messages)

    # アシスタントメッセージの追加
    assistant_msg = ChatMessage(
        role="assistant",
        content=assistant_content,
        timestamp=datetime.now()
    )
    session.messages.append(assistant_msg)

    # セッションの保存（簡易版）
    chat_storage.save(session)

    return {"role": "assistant", "content": assistant_content}

@app.get("/chat/{session_id}/history")
async def get_chat_history(session_id: str):
    session = chat_storage.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found")
    return session.messages

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)