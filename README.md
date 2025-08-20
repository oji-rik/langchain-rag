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

## ファイル構成と機能説明

### 🚀 メインプログラム
- **`integrated_agent.py`** - **統合システムの心臓部**
  - 全機能を統合したメインプログラム
  - LangChain Agentが質問内容を自動判断
  - RAG検索・文書追加・測定実行を適切にルーティング
  - **実行方法**: `python integrated_agent.py`

### 📚 RAG（文書検索）システム
- **`pdf_rag_core.py`** - **RAGエンジン本体**
  - PDF/Word/PowerPointからテキストを抽出
  - 文書を1000文字ずつのチャンクに自動分割
  - Azure OpenAIでベクトル化（意味をベクトルで表現）
  - FAISSで高速検索（類似度でマッチング）
  - **初心者向け説明**: 「文書の中身を理解して質問に答えるAI」

- **`rag_tool.py`** - **RAG機能をAgentツール化**
  - `DocumentationSearchTool`: 文書検索機能
  - `DocumentAddTool`: チャット中に新文書追加機能
  - LangChain AgentがRAGを簡単に呼び出せる形に変換
  - **初心者向け説明**: 「RAGをツールとして使えるようにする変換器」

### ⚙️ 測定関数システム
- **`csharp_tools.py`** - **C#サーバーとの橋渡し**
  - C#で書かれた測定関数（距離、角度など）をPythonから呼び出し
  - HTTPリクエストで関数実行をリクエスト
  - 結果をLangChain Agentが理解できる形で返却
  - **初心者向け説明**: 「計算専用サーバーとチャットシステムをつなぐ橋」

### 🔧 設定・依存関係
- **`requirements.txt`** - **必要なライブラリ一覧**
  - LangChain: AI Agent機能
  - FAISS: ベクトル検索エンジン 
  - pypdf/python-docx: ファイル読み込み
  - requests: HTTP通信
  - **初心者向け説明**: 「このシステムが動くのに必要な材料リスト」

- **`.env`** - **秘密の設定ファイル**
  - Azure OpenAI APIキー（絶対秘密）
  - エンドポイントURL（接続先情報）
  - デプロイメント名（使用するAIモデル名）
  - **初心者向け説明**: 「AIサービスに接続するための身分証明書」

### 📖 ドキュメント
- **`README.md`** - **このファイル（システム説明書）**
- **`LANGCHAIN_RAG_BENEFITS.md`** - **技術選択の理由**
- **`RAG_ARCHITECTURE_NOTES.md`** - **RAGの仕組み詳細**

## 🔍 RAG初心者向け補足説明

### RAG（Retrieval-Augmented Generation）とは？
1. **Retrieval（検索）**: 関連する文書を見つける
2. **Augmented（拡張）**: 見つけた文書を参考資料として追加
3. **Generation（生成）**: 参考資料を元に回答を生成

### このシステムでのRAGの流れ
```
質問「角度測定の方法は？」
↓
1. 質問をベクトル化（数値に変換）
2. 似たベクトルを文書から検索
3. 関連部分を抽出（例：「AngleMeasurement関数で...」）
4. GPT-4に「この資料を参考に回答して」と指示
5. 自然な日本語で回答生成
```

### なぜRAGが必要？
- **最新情報**: GPTの学習データにない最新マニュアルも対応
- **専門知識**: 会社固有の測定器仕様書も理解
- **正確性**: 根拠となる文書を明示して回答

## 🚀 性能最適化ガイド

### ベクトル化処理の高速化

文書をRAGシステムに読み込む際のベクトル化処理時間を最適化できます。

#### 性能モード一覧

| モード | バッチサイズ | 遅延時間 | 100チャンク処理時間* |
|--------|-------------|----------|-------------------|
| **turbo** | 100 | 0.1秒 | ~0.1秒 |
| **extreme** | 200 | 0.1秒 | ~0.05秒 |
| **ultra** | 300 | 0.1秒 | ~0.03秒 |
| **maximum** | 400 | 0.1秒 | ~0.025秒 |
| **insane** | 500 | 0.1秒 | ~0.02秒 |

*実際の処理時間はAzure OpenAIの負荷状況により変動します

#### 性能モードの選び方

```bash
# システム起動時に選択
python integrated_agent.py
```

**推奨用途:**
- **turbo**: 1000チャンク未満の文書
- **extreme**: 1000-5000チャンクの文書
- **ultra**: 5000-10000チャンクの文書
- **maximum**: 10000-20000チャンクの文書
- **insane**: 20000チャンク以上の大容量文書（推奨デフォルト）

#### 技術詳細

**大容量バッチ適応調整機能:**
- 429エラー検出時：**バッチサイズ維持**、遅延時間のみ前回成功値に固定
- 全モード0.1秒固定：大容量バッチに最適化された間隔
- 最大500バッチ/0.1秒の高速処理が可能

**使用例:**
```python
# 大容量バッチモード
rag_system = PDFRAGSystem(performance_mode="insane")

# または統合システムで
agent = create_integrated_agent(..., performance_mode="maximum")
```

#### 注意事項
- **insane**: 500バッチ同時処理、最大並列度
- **maximum**: 400バッチ、大容量文書対応
- **エラー後固定化**: 一度最適設定を発見したら、その設定で永続安定動作
- **処理能力**: 20000チャンク文書を約40秒で処理可能