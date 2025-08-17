from typing import Any, Optional
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from pdf_rag_core import PDFRAGSystem
import logging

logger = logging.getLogger(__name__)


class DocumentationSearchTool(BaseTool):
    """測定器の機能や使い方についてドキュメントを検索するツール"""
    
    name: str = "documentation_search"
    description: str = """Use this tool to search for information about measurement functions, features, or usage instructions. 
    Perfect for questions like: 'What functions are available for distance measurement?', 'How to use angle measurement?', 
    'What features does this device have?', 'どの機能を使えばよい？', '角度測定の方法は？'"""
    
    rag_system: Optional[PDFRAGSystem] = Field(default=None, description="RAG system instance")
    
    def __init__(self, rag_system: PDFRAGSystem = None, **kwargs):
        super().__init__(**kwargs)
        self.rag_system = rag_system
    
    def _run(self, query: str) -> str:
        """ドキュメント検索を実行"""
        try:
            if not self.rag_system:
                return "RAGシステムが初期化されていません。先にドキュメントを読み込んでください。"
            
            if not self.rag_system.qa_chain:
                return "ドキュメントが読み込まれていません。先にPDFファイルを読み込んでください。"
            
            logger.info(f"RAGツールでドキュメント検索実行: {query}")
            
            result = self.rag_system.ask(query)
            
            # 回答と参照元を整形
            answer = result["answer"]
            sources = result.get("source_documents", [])
            
            response = f"📚 ドキュメント検索結果:\n{answer}"
            
            if sources:
                response += f"\n\n📖 参照元: {len(sources)}件"
                for i, doc in enumerate(sources[:2], 1):  # 最大2件表示
                    page = doc.metadata.get('page', '不明')
                    response += f"\n  [{i}] ページ{page}"
            
            return response
            
        except Exception as e:
            logger.error(f"RAGツールでエラー発生: {e}")
            return f"ドキュメント検索中にエラーが発生しました: {str(e)}"
    
    async def _arun(self, query: str) -> str:
        """非同期実行（非同期がサポートされていない場合は同期実行）"""
        return self._run(query)


def create_rag_tool(
    azure_endpoint: str,
    azure_deployment: str, 
    embedding_deployment: str,
    api_key: str,
    documentation_path: str,
    api_version: str = "2024-12-01-preview"
) -> DocumentationSearchTool:
    """
    RAGツールを作成し、指定されたドキュメントを事前読み込み
    
    Args:
        azure_endpoint: Azure OpenAI エンドポイント
        azure_deployment: チャット用デプロイメント名
        embedding_deployment: 埋め込み用デプロイメント名
        api_key: Azure OpenAI APIキー
        documentation_path: 事前読み込みするドキュメントのパス
        api_version: Azure OpenAI APIバージョン
    
    Returns:
        初期化済みRAGツール
    """
    logger.info("RAGツールを初期化中...")
    
    # RAGシステムの初期化
    rag_system = PDFRAGSystem(
        azure_endpoint=azure_endpoint,
        azure_deployment=azure_deployment,
        embedding_deployment=embedding_deployment,
        api_key=api_key,
        api_version=api_version,
        batch_size=3,      # より保守的な設定
        batch_delay=10.0   # 10秒間隔
    )
    
    # ドキュメントの事前読み込み
    logger.info(f"ドキュメントを事前読み込み中: {documentation_path}")
    rag_system.load_document(documentation_path)
    logger.info("RAGツールの準備完了")
    
    # RAGツールを作成
    return DocumentationSearchTool(rag_system=rag_system)