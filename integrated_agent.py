import os
import sys
from typing import List
from langchain_openai import AzureChatOpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from csharp_tools import create_tools_from_csharp_server, test_csharp_server_connection
from rag_tool import create_rag_tool
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_integrated_agent(
    azure_endpoint: str,
    azure_deployment: str,
    embedding_deployment: str,
    api_key: str,
    documentation_path: str,
    api_version: str = "2024-12-01-preview",
    csharp_server_url: str = "http://localhost:8080"
) -> AgentExecutor:
    """
    RAG機能 + Function Calling を統合したLangChainエージェントを作成
    
    Args:
        azure_endpoint: Azure OpenAI エンドポイントURL
        azure_deployment: Azure OpenAI チャット用デプロイメント名
        embedding_deployment: Azure OpenAI 埋め込み用デプロイメント名
        api_key: Azure OpenAI APIキー
        documentation_path: 事前読み込みするドキュメントのパス
        api_version: Azure OpenAI API バージョン
        csharp_server_url: C#関数サーバーのURL
        
    Returns:
        設定済みAgentExecutorインスタンス
    """
    
    print("=== 統合測定システム初期化 ===")
    
    # C#サーバー接続をテスト
    print(f"Testing connection to C# server at {csharp_server_url}...")
    if not test_csharp_server_connection(csharp_server_url):
        raise Exception(f"Cannot connect to C# server at {csharp_server_url}. Make sure the server is running.")
    print("✓ C# server connection successful")
    
    # Azure OpenAI client を作成
    llm = AzureChatOpenAI(
        azure_endpoint=azure_endpoint,
        azure_deployment=azure_deployment,
        api_key=api_key,
        api_version=api_version,
        temperature=0.7
    )
    print("✓ Azure OpenAI client created")
    
    # C#関数ツールを作成
    print("Fetching C# function tools...")
    csharp_tools = create_tools_from_csharp_server(csharp_server_url)
    print(f"✓ Loaded {len(csharp_tools)} C# function tools:")
    for tool in csharp_tools:
        print(f"  - {tool.name}: {tool.description}")
    
    # RAGツールを作成（ドキュメント事前読み込み）
    print("Initializing RAG documentation system...")
    rag_tool = create_rag_tool(
        azure_endpoint=azure_endpoint,
        azure_deployment=azure_deployment,
        embedding_deployment=embedding_deployment,
        api_key=api_key,
        documentation_path=documentation_path,
        api_version=api_version
    )
    print("✓ RAG documentation system ready")
    
    # 全ツールを統合
    all_tools = csharp_tools + [rag_tool]
    print(f"✓ Total tools available: {len(all_tools)}")
    
    # メモリを作成（会話履歴管理）
    memory = ConversationBufferMemory(
        return_messages=True,
        memory_key="chat_history"
    )
    
    # プロンプトテンプレートを作成
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an intelligent measurement system assistant. You have access to two types of capabilities:

1. **Documentation Search**: Use the 'documentation_search' tool to find information about measurement functions, features, and usage instructions.
2. **Function Execution**: Use the available measurement tools to perform actual calculations and measurements.

Guidelines:
- When users ask about "what functions are available", "how to use", or "which feature to use" → Use documentation_search
- When users ask to "measure", "calculate", or "execute" something → Use the appropriate measurement function
- You can use both in sequence: first search for information, then execute the function
- Always provide clear, helpful responses in the user's language (Japanese or English)
- If unsure about which tool to use, try documentation_search first

Available measurement functions will be dynamically loaded from the C# server."""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    # Agentを作成
    agent = create_openai_functions_agent(
        llm=llm,
        tools=all_tools,
        prompt=prompt
    )
    
    # Agent Executorを作成
    agent_executor = AgentExecutor(
        agent=agent,
        tools=all_tools,
        memory=memory,
        verbose=True,
        max_iterations=15,
        return_intermediate_steps=True
    )
    
    print("✓ Integrated agent created successfully")
    return agent_executor


def main():
    """統合システムのメイン関数"""
    
    # 環境変数を読み込み
    load_dotenv()
    
    # 設定値
    AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT") 
    EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
    API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
    CSHARP_SERVER_URL = "http://localhost:8080"
    
    # 必須設定の確認
    if not all([AZURE_ENDPOINT, AZURE_DEPLOYMENT, EMBEDDING_DEPLOYMENT, API_KEY]):
        print("Error: Azure OpenAI設定が不完全です")
        print("必要な環境変数:")
        print("  AZURE_OPENAI_ENDPOINT")
        print("  AZURE_OPENAI_DEPLOYMENT") 
        print("  AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
        print("  AZURE_OPENAI_API_KEY")
        sys.exit(1)
    
    # ドキュメントパスの入力
    documentation_path = input("事前読み込みするドキュメントのパス（PDF/PowerPoint/Word/URL）を入力してください: ").strip()
    
    if not documentation_path:
        print("ドキュメントパスが指定されていません。RAG機能なしで開始します...")
        # RAG機能なしでの実装も可能だが、ここでは簡略化
        sys.exit(1)
    
    try:
        # 統合エージェントを作成
        print("\n統合測定システムを初期化中...")
        agent_executor = create_integrated_agent(
            azure_endpoint=AZURE_ENDPOINT,
            azure_deployment=AZURE_DEPLOYMENT,
            embedding_deployment=EMBEDDING_DEPLOYMENT,
            api_key=API_KEY,
            documentation_path=documentation_path,
            csharp_server_url=CSHARP_SERVER_URL
        )
        
        print("\n" + "="*80)
        print("🚀 統合測定システム Ready!")
        print("="*80)
        print("利用可能な機能:")
        print("📚 ドキュメント検索: 機能の説明や使い方を調べる")
        print("⚙️  測定機能: 実際の計算や測定を実行")
        print("🔄 複合処理: 機能を調べた後、実際に実行")
        print("\n例:")
        print("- '距離測定機能について教えて'")
        print("- '点(1,1)と点(5,4)の距離を測って'") 
        print("- '角度測定の機能はある？あれば実際に使って'")
        print("\nType 'exit', 'quit', or '終了' to quit.")
        print("="*80)
        
        # インタラクティブチャットループ
        while True:
            try:
                user_input = input("\n質問・指示: ").strip()
                
                if user_input.lower() in ['exit', 'quit', '終了', '']:
                    print("システムを終了します。")
                    break
                
                print("\n処理中...")
                response = agent_executor.invoke({"input": user_input})
                print(f"\n💬 回答:\n{response['output']}")
                
            except KeyboardInterrupt:
                print("\n\nシステムを終了します。")
                break
            except Exception as e:
                print(f"\nエラーが発生しました: {str(e)}")
                
    except Exception as e:
        print(f"初期化エラー: {str(e)}")
        print("\nトラブルシューティング:")
        print("1. C#サーバーが起動していることを確認")
        print("2. Azure OpenAI設定（.env）が正しいことを確認")
        print("3. ドキュメントファイルが存在することを確認")
        print("4. 埋め込み用デプロイメントが作成されていることを確認")
        sys.exit(1)


if __name__ == "__main__":
    main()