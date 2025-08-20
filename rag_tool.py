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
    api_version: str = "2024-12-01-preview",
    performance_mode: str = "insane"
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
        performance_mode: 性能モード ("safe", "balanced", "fast", "turbo")
    
    Returns:
        初期化済みRAGツール
    """
    logger.info(f"RAGツールを初期化中（{performance_mode}モード）...")
    
    # RAGシステムの初期化（性能モード適用）
    rag_system = PDFRAGSystem(
        azure_endpoint=azure_endpoint,
        azure_deployment=azure_deployment,
        embedding_deployment=embedding_deployment,
        api_key=api_key,
        api_version=api_version,
        performance_mode=performance_mode
    )
    
    # ドキュメントの事前読み込み
    logger.info(f"ドキュメントを事前読み込み中: {documentation_path}")
    rag_system.load_document(documentation_path)
    logger.info("RAGツールの準備完了")
    
    # RAGツールを作成
    return DocumentationSearchTool(rag_system=rag_system)


class DocumentAddTool(BaseTool):
    """RAGシステムに新しい文書を動的に追加するツール"""
    
    name: str = "add_document" 
    description: str = """Use this tool when user wants to add new documents to the RAG system. 
    Perfect for requests like: 'Add another document', '新しい文書を追加したい', 'Read another manual', 
    'Load more documentation'. Ask for the document path (file path or URL) and add it to the knowledge base."""
    
    rag_system: Optional[PDFRAGSystem] = Field(default=None, description="RAG system instance")
    
    def __init__(self, rag_system: PDFRAGSystem = None, **kwargs):
        super().__init__(**kwargs)
        self.rag_system = rag_system
    
    def _run(self, document_path: str) -> str:
        """新しい文書をRAGシステムに追加"""
        try:
            if not self.rag_system:
                return "RAGシステムが初期化されていません。"
            
            logger.info(f"DocumentAddTool で新しい文書を追加: {document_path}")
            
            # 文書を追加（初回の場合は load_document を使用）
            if not self.rag_system.vectorstore:
                # 初回文書読み込み
                self.rag_system.load_document(document_path)
                doc_info = self.rag_system.get_document_info()
                
                response = f"""📄 初回文書読み込み完了!
                
読み込んだ文書:
- ページ/セクション数: {doc_info['pages']}
- 総文字数: {doc_info['total_characters']:,}
- チャンク数: {doc_info['chunks']}

文書検索機能が利用可能になりました！"""
                
                return response
            else:
                # 追加文書読み込み
                result = self.rag_system.add_document(document_path)
                
                # 結果を整形
                response = f"""📄 文書追加完了!
            
追加した文書:
- ページ/セクション数: {result['added_pages']}
- チャンク数: {result['added_chunks']}

現在の知識ベース:
- 総ページ数: {result['total_pages']}
- 総チャンク数: {result['total_chunks']}
- 総文字数: {result['total_characters']:,}

新しい文書についても質問できるようになりました！"""
            
            return response
            
        except Exception as e:
            logger.error(f"DocumentAddToolでエラー発生: {e}")
            return f"文書追加中にエラーが発生しました: {str(e)}"
    
    async def _arun(self, document_path: str) -> str:
        """非同期実行（非同期がサポートされていない場合は同期実行）"""
        return self._run(document_path)


def create_document_add_tool(rag_system: PDFRAGSystem) -> DocumentAddTool:
    """
    文書追加ツールを作成
    
    Args:
        rag_system: 既存のRAGシステムインスタンス
        
    Returns:
        文書追加ツール
    """
    return DocumentAddTool(rag_system=rag_system)


def create_empty_rag_system(
    azure_endpoint: str,
    azure_deployment: str,
    embedding_deployment: str,
    api_key: str,
    api_version: str = "2024-12-01-preview",
    performance_mode: str = "insane"
) -> PDFRAGSystem:
    """
    空のRAGシステムを作成（後で文書追加用）
    
    Args:
        azure_endpoint: Azure OpenAI エンドポイント
        azure_deployment: チャット用デプロイメント名
        embedding_deployment: 埋め込み用デプロイメント名
        api_key: Azure OpenAI APIキー
        api_version: Azure OpenAI APIバージョン
        performance_mode: 性能モード ("safe", "balanced", "fast", "turbo")
        
    Returns:
        初期化済みの空のRAGシステム
    """
    logger.info(f"空のRAGシステムを初期化中（{performance_mode}モード）...")
    
    return PDFRAGSystem(
        azure_endpoint=azure_endpoint,
        azure_deployment=azure_deployment,
        embedding_deployment=embedding_deployment,
        api_key=api_key,
        api_version=api_version,
        performance_mode=performance_mode
    )


def get_performance_info(rag_system: PDFRAGSystem) -> str:
    """
    RAGシステムの性能設定情報を取得
    
    Args:
        rag_system: RAGシステムインスタンス
        
    Returns:
        性能情報の文字列
    """
    if not rag_system:
        return "RAGシステムが初期化されていません"
    
    info = rag_system.get_performance_info()
    
    estimated_time = info.get('estimated_time_per_100_chunks', 0)
    
    return f"""📊 性能設定情報:
- モード: {info['performance_mode']}
- バッチサイズ: {info['batch_size']}
- 遅延時間: {info['batch_delay']}秒
- 適応モード: {'有効' if info['adaptive_mode'] else '無効'}
- 100チャンクあたりの推定時間: {estimated_time:.1f}分"""