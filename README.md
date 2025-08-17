# 統合測定システム (LangChain Agent + RAG)

Function Calling と RAG機能を統合した自然言語測定システム

## 機能

### 🔧 Function Calling
- C#サーバーの測定関数を自然言語で実行
- 距離測定、角度計算などの数学的処理

### 📚 RAG (文書検索)
- 測定器の説明書やマニュアルから機能検索
- "どの機能を使えばよい？"などの質問に回答

### 🤖 自動判断Agent
- 質問内容を分析して適切なツールを自動選択
- 複合処理（検索→実行）の自動化

## セットアップ

### 1. 依存関係のインストール
```bash
pip install -r requirements.txt
```

### 2. 環境設定
`.env`ファイルを作成：
```
AZURE_OPENAI_ENDPOINT=https://your-endpoint.cognitiveservices.azure.com/
AZURE_OPENAI_DEPLOYMENT=gpt-4.1
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-ada-002
AZURE_OPENAI_API_KEY=your_api_key_here
```

### 3. C#サーバー起動
測定関数サーバーを `http://localhost:8080` で起動

## 使用方法

```bash
python integrated_agent.py
```

### 使用例

#### 機能検索
```
質問: "距離測定機能について教えて"
回答: [RAG] DistanceMeasurement関数が利用可能です...
```

#### 直接実行
```
質問: "点(1,1)と点(5,4)の距離を測って"
回答: [Function Calling] 距離は5.0です
```

#### 複合処理
```
質問: "角度測定の方法を調べて、その後実際に測って"
回答: [RAG + Function Calling] 角度測定はAngleMeasurement関数で...実行結果: 45度
```

## システム構成

```
統合Agent
├── Memory (統一された会話履歴)
├── Tools
│   ├── C# Function Tools
│   │   ├── DistanceMeasurement
│   │   ├── AngleMeasurement
│   │   └── その他測定関数
│   └── RAG Tool
│       └── DocumentationSearch
└── LLM (GPT-4.1) - 自動判断・ルーティング
```

## ファイル構成

- `integrated_agent.py`: メインシステム
- `rag_tool.py`: RAG機能のToolラッパー
- `pdf_rag_core.py`: RAGコア機能
- `csharp_tools.py`: C#関数ツール
- `requirements.txt`: 依存関係
- `.env`: Azure設定