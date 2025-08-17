import requests
import uuid
from typing import Any, Dict, List, Optional
from langchain.tools import BaseTool
from pydantic import BaseModel, Field


class CSharpFunctionTool(BaseTool):
    """C# HTTPサーバー上で関数を実行するカスタムツール。"""
    
    name: str = Field(description="Name of the function")
    description: str = Field(description="Description of what the function does")
    base_url: str = Field(default="http://localhost:8080", description="Base URL of the C# server")
    parameters_schema: Dict[str, Any] = Field(description="JSON schema for function parameters")
    
    def _run(self, **kwargs: Any) -> str:
        """C#サーバー上で関数を実行する。"""
        try:
            # 一意のリクエストIDを生成
            request_id = str(uuid.uuid4())
            
            # Prepare the request payload
            payload = {
                "function_name": self.name,
                "arguments": kwargs,
                "request_id": request_id
            }
            
            # Make the HTTP request to the C# server
            response = requests.post(
                f"{self.base_url}/execute",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            # Check if the request was successful
            response.raise_for_status()
            
            # Parse the response
            result_data = response.json()
            
            if result_data.get("success", False):
                return str(result_data["result"])
            else:
                error_msg = result_data.get("error", "Unknown error occurred")
                raise Exception(f"Function execution failed: {error_msg}")
                
        except requests.exceptions.RequestException as e:
            raise Exception(f"HTTP request failed: {str(e)}")
        except Exception as e:
            raise Exception(f"Tool execution error: {str(e)}")

    async def _arun(self, **kwargs: Any) -> str:
        """_runの非同期版（未実装、同期版にフォールバック）。"""
        return self._run(**kwargs)


def create_tools_from_csharp_server(base_url: str = "http://localhost:8080") -> List[CSharpFunctionTool]:
    """
    C#サーバーからツール定義を取得してLangChainツールを作成。
    
    Args:
        base_url: C# HTTPサーバーのベースURL
        
    Returns:
        CSharpFunctionToolインスタンスのリスト
    """
    try:
        # Get tool definitions from the C# server
        response = requests.get(f"{base_url}/tools", timeout=30)
        response.raise_for_status()
        
        tools_data = response.json()
        tools = []
        
        for tool_def in tools_data.get("tools", []):
            tool = CSharpFunctionTool(
                name=tool_def["name"],
                description=tool_def["description"],
                base_url=base_url,
                parameters_schema=tool_def.get("parameters", {})
            )
            tools.append(tool)
        
        return tools
        
    except requests.exceptions.RequestException as e:
        raise Exception(f"Failed to connect to C# server at {base_url}: {str(e)}")
    except Exception as e:
        raise Exception(f"Error creating tools from C# server: {str(e)}")


def test_csharp_server_connection(base_url: str = "http://localhost:8080") -> bool:
    """
    C#サーバーが実行中でアクセス可能かテストする。
    
    Args:
        base_url: C# HTTPサーバーのベースURL
        
    Returns:
        サーバーがアクセス可能な場合True、そうでなければFalse
    """
    try:
        response = requests.get(f"{base_url}/tools", timeout=5)
        return response.status_code == 200
    except:
        return False