"""
簡単なPDF RAGシステム
PDFファイルを読み込んで質問に答えるシステム
"""
import os
import logging
import time
from typing import List, Optional
from pathlib import Path
from urllib.parse import urlparse

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings
from langchain_openai import AzureChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredPowerPointLoader,
    Docx2txtLoader,
    WebBaseLoader,
    TextLoader
)
from langchain.schema import Document

import tiktoken
from tqdm import tqdm
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PDFRAGSystem:
    """PDF文書を使った検索拡張生成(RAG)システム"""
    
    def __init__(
        self, 
        azure_endpoint: Optional[str] = None,
        azure_deployment: Optional[str] = None,
        embedding_deployment: Optional[str] = None,
        api_key: Optional[str] = None,
        api_version: str = "2024-12-01-preview",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        batch_size: int = 5,
        batch_delay: float = 15.0
    ):
        """
        RAGシステムの初期化
        
        Args:
            azure_endpoint: Azure OpenAI エンドポイント
            azure_deployment: Azure OpenAI デプロイメント名（チャット用）
            embedding_deployment: Azure OpenAI 埋め込み用デプロイメント名
            api_key: Azure OpenAI APIキー
            api_version: Azure OpenAI APIバージョン
            chunk_size: テキストチャンクサイズ
            chunk_overlap: チャンク間の重複サイズ
            batch_size: 埋め込み生成バッチサイズ
            batch_delay: バッチ間の待機時間（秒）
        """
        load_dotenv()
        
        self.azure_endpoint = azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        self.azure_deployment = azure_deployment or os.getenv("AZURE_OPENAI_DEPLOYMENT")
        self.embedding_deployment = embedding_deployment or os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
        self.api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
        self.api_version = api_version
        
        if not self.azure_endpoint:
            raise ValueError("Azure OpenAI エンドポイントが設定されていません")
        if not self.azure_deployment:
            raise ValueError("Azure OpenAI デプロイメント名が設定されていません")
        if not self.embedding_deployment:
            raise ValueError("Azure OpenAI 埋め込み用デプロイメント名が設定されていません")
        if not self.api_key:
            raise ValueError("Azure OpenAI APIキーが設定されていません")
            
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.batch_size = batch_size
        self.batch_delay = batch_delay
        
        # 埋め込みモデルの初期化
        self.embeddings = AzureOpenAIEmbeddings(
            azure_endpoint=self.azure_endpoint,
            azure_deployment=self.embedding_deployment,  # 埋め込み用デプロイメントを使用
            api_key=self.api_key,
            api_version=self.api_version
        )
        
        # LLMの初期化
        self.llm = AzureChatOpenAI(
            azure_endpoint=self.azure_endpoint,
            azure_deployment=self.azure_deployment,
            api_key=self.api_key,
            api_version=self.api_version,
            temperature=0.1
        )
        
        # テキスト分割器の初期化
        self.text_splitter = CharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separator="\n"
        )
        
        self.vectorstore = None
        self.qa_chain = None
        self.documents = []
        
        logger.info("PDFRAGSystemが初期化されました")
    
    def load_document(self, document_path: str, doc_type: str = "auto") -> None:
        """
        文書ファイルまたはURLを読み込み、ベクトルストアを構築
        
        Args:
            document_path: 文書ファイルのパスまたはURL
            doc_type: 文書タイプ ("auto", "pdf", "pptx", "docx", "web", "txt")
        """
        # 文書タイプの自動判定
        if doc_type == "auto":
            doc_type = self._detect_document_type(document_path)
        
        logger.info(f"文書を読み込み中: {document_path} (タイプ: {doc_type})")
        
        # 文書タイプに応じたローダーを選択
        loader = self._get_loader(document_path, doc_type)
        self.documents = loader.load()
        
        logger.info(f"{len(self.documents)}ページ/セクションの文書を読み込みました")
        
        # テキストを分割
        texts = self.text_splitter.split_documents(self.documents)
        logger.info(f"テキストを{len(texts)}チャンクに分割しました")
        
        # ベクトルストアをバッチ処理で構築
        logger.info(f"ベクトル埋め込みを生成中... (バッチサイズ: {self.batch_size}, 遅延: {self.batch_delay}秒)")
        self.vectorstore = self._build_vectorstore_with_batches(texts)
        
        # QAチェーンを初期化
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_kwargs={"k": 3}
            ),
            return_source_documents=True
        )
        
        logger.info("RAGシステムの準備が完了しました")
    
    def _build_vectorstore_with_batches(self, texts: List[Document]) -> FAISS:
        """
        バッチ処理でベクトルストアを構築（レート制限対応）
        
        Args:
            texts: 分割されたテキストのリスト
            
        Returns:
            構築されたFAISSベクトルストア
        """
        vectorstore = None
        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size
        
        logger.info(f"合計{len(texts)}チャンクを{total_batches}バッチで処理します")
        
        for i in tqdm(range(0, len(texts), self.batch_size), desc="埋め込み生成"):
            batch = texts[i:i + self.batch_size]
            batch_num = i // self.batch_size + 1
            
            logger.info(f"バッチ {batch_num}/{total_batches} を処理中... ({len(batch)}チャンク)")
            
            try:
                if vectorstore is None:
                    # 最初のバッチでベクトルストアを初期化
                    vectorstore = FAISS.from_documents(
                        documents=batch,
                        embedding=self.embeddings
                    )
                else:
                    # 既存のベクトルストアに追加
                    batch_vectorstore = FAISS.from_documents(
                        documents=batch,
                        embedding=self.embeddings
                    )
                    vectorstore.merge_from(batch_vectorstore)
                
                logger.info(f"バッチ {batch_num} 完了")
                
                # 最後のバッチでない場合は待機
                if i + self.batch_size < len(texts):
                    logger.info(f"レート制限回避のため{self.batch_delay}秒待機中...")
                    time.sleep(self.batch_delay)
                    
            except Exception as e:
                if "429" in str(e) or "Too Many Requests" in str(e):
                    logger.warning(f"レート制限に達しました。{self.batch_delay * 2}秒待機後にリトライします...")
                    time.sleep(self.batch_delay * 2)
                    # 同じバッチを再試行
                    i -= self.batch_size
                    continue
                else:
                    raise e
        
        return vectorstore
    
    def _detect_document_type(self, document_path: str) -> str:
        """
        ファイルパスまたはURLから文書タイプを自動判定
        
        Args:
            document_path: ファイルパスまたはURL
            
        Returns:
            判定された文書タイプ
        """
        # URLの場合
        parsed = urlparse(document_path)
        if parsed.scheme in ['http', 'https']:
            return "web"
        
        # ファイル拡張子による判定
        path = Path(document_path)
        extension = path.suffix.lower()
        
        type_mapping = {
            '.pdf': 'pdf',
            '.pptx': 'pptx',
            '.ppt': 'pptx',
            '.docx': 'docx',
            '.doc': 'docx',
            '.txt': 'txt',
            '.md': 'txt',
        }
        
        return type_mapping.get(extension, 'txt')
    
    def _get_loader(self, document_path: str, doc_type: str):
        """
        文書タイプに応じたローダーを取得
        
        Args:
            document_path: ファイルパスまたはURL
            doc_type: 文書タイプ
            
        Returns:
            適切なローダーインスタンス
        """
        if doc_type == "pdf":
            if not Path(document_path).exists():
                raise FileNotFoundError(f"PDFファイルが見つかりません: {document_path}")
            return PyPDFLoader(document_path)
        
        elif doc_type == "pptx":
            if not Path(document_path).exists():
                raise FileNotFoundError(f"PowerPointファイルが見つかりません: {document_path}")
            return UnstructuredPowerPointLoader(document_path)
        
        elif doc_type == "docx":
            if not Path(document_path).exists():
                raise FileNotFoundError(f"Wordファイルが見つかりません: {document_path}")
            return Docx2txtLoader(document_path)
        
        elif doc_type == "web":
            return WebBaseLoader(document_path)
        
        elif doc_type == "txt":
            if not Path(document_path).exists():
                raise FileNotFoundError(f"テキストファイルが見つかりません: {document_path}")
            return TextLoader(document_path, encoding='utf-8')
        
        else:
            raise ValueError(f"サポートされていない文書タイプです: {doc_type}")
    
    def load_pdf(self, pdf_path: str) -> None:
        """
        PDFファイルを読み込み（後方互換性のため）
        
        Args:
            pdf_path: PDFファイルのパス
        """
        self.load_document(pdf_path, "pdf")
    
    def add_document(self, document_path: str, doc_type: str = "auto") -> None:
        """
        既存のRAGシステムに新しい文書を追加
        
        Args:
            document_path: 追加する文書ファイルのパスまたはURL
            doc_type: 文書タイプ ("auto", "pdf", "pptx", "docx", "web", "txt")
        """
        if not self.vectorstore:
            logger.error("ベースとなるベクトルストアが存在しません。まず初期文書を読み込んでください。")
            raise ValueError("まず初期文書を読み込んでから追加してください（load_document()を先に実行）")
        
        logger.info(f"新しい文書を追加中: {document_path}")
        
        # 文書タイプの自動判定
        if doc_type == "auto":
            doc_type = self._detect_document_type(document_path)
        
        logger.info(f"文書を読み込み中: {document_path} (タイプ: {doc_type})")
        
        # 新しい文書を読み込み
        loader = self._get_loader(document_path, doc_type)
        new_documents = loader.load()
        
        logger.info(f"{len(new_documents)}ページ/セクションの新しい文書を読み込みました")
        
        # テキストを分割
        new_texts = self.text_splitter.split_documents(new_documents)
        logger.info(f"新しいテキストを{len(new_texts)}チャンクに分割しました")
        
        # 新しいベクトルストアを構築
        logger.info("新しい文書のベクトル埋め込みを生成中...")
        new_vectorstore = self._build_vectorstore_with_batches(new_texts)
        
        # 既存のベクトルストアにマージ
        logger.info("既存のベクトルストアに新しい文書を統合中...")
        self.vectorstore.merge_from(new_vectorstore)
        
        # 文書リストを更新
        self.documents.extend(new_documents)
        
        # QAチェーンを再初期化（必要に応じて）
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_kwargs={"k": 5}  # 文書が増えたのでk=5に変更
            ),
            return_source_documents=True
        )
        
        logger.info(f"文書の追加が完了しました。総文書数: {len(self.documents)}")
        
        # 更新後の情報を返す
        total_chars = sum(len(doc.page_content) for doc in self.documents)
        total_chunks = len(self.vectorstore.index_to_docstore_id) if self.vectorstore else 0
        
        return {
            "added_pages": len(new_documents),
            "added_chunks": len(new_texts),
            "total_pages": len(self.documents),
            "total_chunks": total_chunks,
            "total_characters": total_chars
        }
    
    def ask(self, question: str) -> dict:
        """
        質問に対して回答を生成
        
        Args:
            question: 質問文
            
        Returns:
            回答と参照元を含む辞書
        """
        if not self.qa_chain:
            raise ValueError("まず文書ファイルを読み込んでください（load_document()またはload_pdf()を呼び出してください）")
        
        logger.info(f"質問を処理中: {question}")
        
        result = self.qa_chain({"query": question})
        
        response = {
            "answer": result["result"],
            "source_documents": result["source_documents"]
        }
        
        return response
    
    def get_document_info(self) -> dict:
        """
        読み込まれた文書の情報を取得
        
        Returns:
            文書情報の辞書
        """
        if not self.documents:
            return {"status": "文書が読み込まれていません"}
        
        total_chars = sum(len(doc.page_content) for doc in self.documents)
        
        return {
            "pages": len(self.documents),
            "total_characters": total_chars,
            "chunks": len(self.vectorstore.index_to_docstore_id) if self.vectorstore else 0
        }
    
    def save_vectorstore(self, save_path: str) -> None:
        """
        ベクトルストアを保存
        
        Args:
            save_path: 保存先パス
        """
        if not self.vectorstore:
            raise ValueError("ベクトルストアが初期化されていません")
        
        self.vectorstore.save_local(save_path)
        logger.info(f"ベクトルストアを保存しました: {save_path}")
    
    def load_vectorstore(self, load_path: str) -> None:
        """
        保存されたベクトルストアを読み込み
        
        Args:
            load_path: 読み込み元パス
        """
        self.vectorstore = FAISS.load_local(
            load_path, 
            embeddings=self.embeddings
        )
        
        # QAチェーンを再初期化
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_kwargs={"k": 3}
            ),
            return_source_documents=True
        )
        
        logger.info(f"ベクトルストアを読み込みました: {load_path}")


def main():
    """シンプルなCLIインターフェース"""
    print("=== マルチフォーマット RAGシステム ===")
    print("PDF、PowerPoint、Word、Webページから質問に答えるシステムです")
    
    # 文書ファイルパスの入力
    document_path = input("\n文書ファイルのパス、またはURLを入力してください: ").strip()
    
    try:
        # RAGシステムの初期化
        rag = PDFRAGSystem()
        
        # 文書の読み込み（自動判定）
        rag.load_document(document_path)
        
        # 文書情報の表示
        info = rag.get_document_info()
        print(f"\n文書情報:")
        print(f"  ページ数: {info['pages']}")
        print(f"  文字数: {info['total_characters']:,}")
        print(f"  チャンク数: {info['chunks']}")
        
        # 質問応答ループ
        print("\n質問を入力してください（'quit'で終了）:")
        while True:
            question = input("\n質問: ").strip()
            
            if question.lower() in ['quit', 'exit', '終了']:
                break
            
            if not question:
                continue
            
            try:
                result = rag.ask(question)
                print(f"\n回答: {result['answer']}")
                
                # 参照元の表示
                if result['source_documents']:
                    print(f"\n参照元: {len(result['source_documents'])}件")
                    for i, doc in enumerate(result['source_documents'], 1):
                        print(f"  [{i}] ページ{doc.metadata.get('page', '不明')}")
                        
            except Exception as e:
                print(f"エラーが発生しました: {e}")
    
    except Exception as e:
        print(f"エラー: {e}")
        return 1
    
    print("\nありがとうございました！")
    return 0


if __name__ == "__main__":
    exit(main())