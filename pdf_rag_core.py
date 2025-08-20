"""
簡単なPDF RAGシステム
PDFファイルを読み込んで質問に答えるシステム
"""
import os
import logging
import time
import hashlib
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
        batch_delay: float = 15.0,
        performance_mode: str = "insane"
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
            performance_mode: 性能モード ("safe", "balanced", "fast", "turbo")
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
        
        # 性能モードに基づいたバッチ設定の適用
        optimized_settings = self._get_performance_settings(performance_mode)
        # 性能モード設定を常に優先（パラメータで上書きされた場合のみそれを使用）
        if batch_size == 5 and batch_delay == 15.0:
            # デフォルト値の場合は性能モード設定を使用
            self.batch_size = optimized_settings["batch_size"]
            self.batch_delay = optimized_settings["batch_delay"]
        else:
            # 明示的に指定された場合はそれを使用
            self.batch_size = batch_size
            self.batch_delay = batch_delay
        
        self.performance_mode = performance_mode
        self.adaptive_mode = optimized_settings["adaptive"]
        
        # 大容量バッチモードの初期設定
        if performance_mode == "insane":
            self.batch_size = 500
            self.batch_delay = 0.1
        elif performance_mode == "maximum":
            self.batch_size = 400
            self.batch_delay = 0.1
        elif performance_mode == "ultra":
            self.batch_size = 300
            self.batch_delay = 0.1
        elif performance_mode == "extreme":
            self.batch_size = 200
            self.batch_delay = 0.1
        elif performance_mode == "turbo":
            self.batch_size = 100
            self.batch_delay = 0.1
        
        logger.info(f"🚀 RAGシステム初期化: {performance_mode}モード")
        logger.info(f"   📊 設定: batch_size={self.batch_size}, delay={self.batch_delay}秒, 適応モード={'有効' if self.adaptive_mode else '無効'}")
        
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
        
        # 最適化管理用の変数
        self.last_successful_delay = None    # 前回成功した遅延時間
        self.error_occurred = False          # 429エラー発生フラグ
        self.optimal_delay_found = False     # 最適遅延確定フラグ
        
        # キャッシュ管理用の変数
        self.cache_dir = Path("./cache")
        self.cache_dir.mkdir(exist_ok=True)  # キャッシュディレクトリを作成
        self.current_document_info = None    # 現在の文書情報
        
        logger.info("PDFRAGSystemが初期化されました")
    
    def _get_performance_settings(self, mode: str) -> dict:
        """
        性能モードに応じた設定を取得
        
        Args:
            mode: 性能モード
            
        Returns:
            最適化設定の辞書
        """
        settings = {
            "turbo": {        # 100バッチサイズ
                "batch_size": 100,
                "batch_delay": 0.1,
                "adaptive": True
            },
            "extreme": {      # 200バッチサイズ
                "batch_size": 200,
                "batch_delay": 0.1,
                "adaptive": True
            },
            "ultra": {        # 300バッチサイズ
                "batch_size": 300,
                "batch_delay": 0.1,
                "adaptive": True
            },
            "maximum": {      # 400バッチサイズ
                "batch_size": 400,
                "batch_delay": 0.1,
                "adaptive": True
            },
            "insane": {       # 500バッチサイズ
                "batch_size": 500,
                "batch_delay": 0.1,
                "adaptive": True
            }
        }
        
        if mode not in settings:
            logger.warning(f"未知の性能モード: {mode}. デフォルトモードを使用します。")
            mode = "insane"
        
        return settings[mode]
    
    def _get_file_cache_key(self, document_path: str) -> str:
        """
        ファイルのキャッシュキーを生成
        
        Args:
            document_path: ファイルパスまたはURL
            
        Returns:
            キャッシュキー（ハッシュ値）
        """
        # URLの場合はそのままハッシュ化
        if document_path.startswith(('http://', 'https://')):
            content = document_path
        else:
            # ローカルファイルの場合はパス+サイズ+更新時刻でハッシュ化
            file_path = Path(document_path)
            if not file_path.exists():
                raise FileNotFoundError(f"ファイルが見つかりません: {document_path}")
            
            stat = file_path.stat()
            content = f"{document_path}_{stat.st_size}_{stat.st_mtime}"
        
        return hashlib.md5(content.encode()).hexdigest()
    
    def _get_document_name(self, document_path: str) -> str:
        """
        文書名を取得（表示用）
        
        Args:
            document_path: ファイルパスまたはURL
            
        Returns:
            文書名
        """
        if document_path.startswith(('http://', 'https://')):
            return document_path
        else:
            return Path(document_path).name
    
    def _save_document_metadata(self, cache_path: Path, document_path: str):
        """
        文書のメタデータを保存
        
        Args:
            cache_path: キャッシュディレクトリパス
            document_path: 文書パス
        """
        metadata = {
            "document_path": document_path,
            "document_name": self._get_document_name(document_path),
            "pages": len(self.documents),
            "chunks": len(self.vectorstore.index_to_docstore_id) if self.vectorstore else 0,
            "total_characters": sum(len(doc.page_content) for doc in self.documents)
        }
        
        metadata_file = cache_path / "metadata.txt"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            for key, value in metadata.items():
                f.write(f"{key}: {value}\n")
        
        self.current_document_info = metadata
    
    def _load_document_metadata(self, cache_path: Path) -> dict:
        """
        文書のメタデータを読み込み
        
        Args:
            cache_path: キャッシュディレクトリパス
            
        Returns:
            文書メタデータ
        """
        metadata_file = cache_path / "metadata.txt"
        if not metadata_file.exists():
            return {}
        
        metadata = {}
        with open(metadata_file, 'r', encoding='utf-8') as f:
            for line in f:
                if ': ' in line:
                    key, value = line.strip().split(': ', 1)
                    # 数値の場合は変換
                    if key in ['pages', 'chunks', 'total_characters']:
                        metadata[key] = int(value)
                    else:
                        metadata[key] = value
        
        return metadata
    
    def load_document(self, document_path: str, doc_type: str = "auto") -> None:
        """
        文書ファイルまたはURLを読み込み、ベクトルストアを構築（自動キャッシュ対応）
        
        Args:
            document_path: 文書ファイルのパスまたはURL
            doc_type: 文書タイプ ("auto", "pdf", "pptx", "docx", "web", "txt")
        """
        # 文書名を取得
        document_name = self._get_document_name(document_path)
        
        # キャッシュキーを生成
        try:
            cache_key = self._get_file_cache_key(document_path)
            cache_path = self.cache_dir / cache_key
            
            # 既存キャッシュをチェック
            if cache_path.exists() and (cache_path / "metadata.txt").exists():
                # キャッシュから復元
                logger.info(f"📂 キャッシュから復元中: {document_name}")
                self.load_vectorstore(str(cache_path))
                
                # メタデータを読み込み
                metadata = self._load_document_metadata(cache_path)
                self.current_document_info = metadata
                
                # QAチェーンを初期化
                self.qa_chain = RetrievalQA.from_chain_type(
                    llm=self.llm,
                    chain_type="stuff",
                    retriever=self.vectorstore.as_retriever(
                        search_kwargs={"k": 3}
                    ),
                    return_source_documents=True
                )
                
                logger.info(f"✅ キャッシュ復元完了: {document_name}")
                logger.info(f"   📊 ページ数: {metadata.get('pages', 0)}, チャンク数: {metadata.get('chunks', 0)}")
                return
                
        except Exception as e:
            logger.warning(f"キャッシュキー生成失敗: {e}")
            # キャッシュが使えない場合は新規処理を続行
        
        # 新規ベクトル化処理
        logger.info(f"📄 新規ベクトル化開始: {document_name}")
        
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
        
        # 自動キャッシュ保存
        try:
            self.save_vectorstore(str(cache_path))
            self._save_document_metadata(cache_path, document_path)
            logger.info(f"💾 キャッシュ保存完了: {document_name}")
        except Exception as e:
            logger.warning(f"キャッシュ保存失敗: {e}")
        
        logger.info(f"✅ 新規ベクトル化完了: {document_name}")
        logger.info(f"   📊 ページ数: {len(self.documents)}, チャンク数: {len(texts)}")
    
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
        
        logger.info(f"🔄 合計{len(texts)}チャンクを{total_batches}バッチで処理します")
        logger.info(f"⚡ 最高速設定: {self.batch_size}チャンク/バッチ, {self.batch_delay}秒間隔")
        
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
                    current_delay = self.batch_delay
                    
                    # 最適設定確定後は固定値を使用
                    if self.optimal_delay_found:
                        current_delay = self.batch_delay
                        logger.info(f"🎯 最適設定で継続: {current_delay:.1f}秒待機中")
                    elif self.adaptive_mode and batch_num > 2 and not self.error_occurred:
                        # 前回成功設定を記録してから最適化
                        self.last_successful_delay = current_delay
                        
                        # エラー未発生時のみ最適化継続（大容量バッチモード）
                        # 0.1秒で固定（大容量バッチでは遅延時間の最適化は不要）
                        current_delay = 0.1  # 0.1秒固定
                        
                        logger.info(f"⚡ 最適化中: {current_delay:.1f}秒待機 (前回成功: {self.last_successful_delay:.1f}秒)")
                    else:
                        logger.info(f"⏰ 基本待機: {current_delay:.1f}秒")
                    
                    time.sleep(current_delay)
                    
            except Exception as e:
                if "429" in str(e) or "Too Many Requests" in str(e):
                    self.error_occurred = True
                    
                    # 前回成功設定があれば、バッチサイズを維持して遅延時間のみ戻す
                    if self.last_successful_delay and not self.optimal_delay_found:
                        self.batch_delay = self.last_successful_delay
                        self.optimal_delay_found = True
                        logger.warning(f"🎯 最適設定確定！バッチサイズ{self.batch_size}維持、遅延{self.batch_delay:.1f}秒で固定")
                    else:
                        # 前回成功設定がない場合の従来処理
                        if self.adaptive_mode and self.batch_size > 2:
                            old_batch_size = self.batch_size
                            self.batch_size = max(self.batch_size - 2, 2)
                            logger.warning(f"🚨 バッチサイズ縮小: {old_batch_size}→{self.batch_size}")
                    
                    wait_time = max(self.batch_delay * 3, 5.0)  # 最低5秒は待機
                    logger.warning(f"⏸️  レート制限回復待機: {wait_time:.1f}秒...")
                    time.sleep(wait_time)
                    
                    # 同じバッチを再試行
                    i -= self.batch_size
                    continue
                else:
                    raise e
        
        return vectorstore
    
    def get_performance_info(self) -> dict:
        """
        現在の性能設定情報を取得
        
        Returns:
            性能設定の詳細情報
        """
        return {
            "performance_mode": self.performance_mode,
            "batch_size": self.batch_size,
            "batch_delay": self.batch_delay,
            "adaptive_mode": self.adaptive_mode,
            "estimated_time_per_100_chunks": (100 / self.batch_size) * self.batch_delay / 60  # 分
        }
    
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
    
    def add_document(self, document_path: str, doc_type: str = "auto") -> dict:
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
        # キャッシュされた情報がある場合はそれを使用
        if self.current_document_info:
            return self.current_document_info
            
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
            embeddings=self.embeddings,
            allow_dangerous_deserialization=True  # 自分で作成したキャッシュファイルなので安全
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
    
    def get_cache_info(self) -> dict:
        """
        キャッシュディレクトリの情報を取得
        
        Returns:
            キャッシュ情報の辞書
        """
        if not self.cache_dir.exists():
            return {"cache_count": 0, "cache_size": 0}
        
        cache_count = 0
        cache_size = 0
        
        for cache_folder in self.cache_dir.iterdir():
            if cache_folder.is_dir():
                cache_count += 1
                # フォルダサイズを計算
                for file_path in cache_folder.rglob('*'):
                    if file_path.is_file():
                        cache_size += file_path.stat().st_size
        
        return {
            "cache_count": cache_count,
            "cache_size": cache_size,
            "cache_size_mb": round(cache_size / (1024 * 1024), 2)
        }
    
    def clear_cache(self) -> bool:
        """
        キャッシュディレクトリを削除
        
        Returns:
            削除成功の可否
        """
        try:
            import shutil
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
                self.cache_dir.mkdir(exist_ok=True)
            logger.info("キャッシュが削除されました")
            return True
        except Exception as e:
            logger.error(f"キャッシュ削除失敗: {e}")
            return False


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