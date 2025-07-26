import asyncio
import os
import json
from typing import Any, Dict, List, Optional, Tuple
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

# mcp client COmponents
from typing import Optional
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


from google import genai
from google.genai import types
from google.genai.types import Tool, FunctionDeclaration
from google.genai.types import GenerateContentConfig

from dotenv import load_dotenv

load_dotenv()

def clean_schema(schema):
    """removes the field Title from config; Since it is forbidden in Gemini"""
    if isinstance(schema, dict):
        schema.pop("title",None)
        
    if "properties" in schema and isinstance(schema["properties"], dict):
        for key in schema["properties"]:
            schema["properties"][key] = clean_schema(schema["properties"][key])
            
    return schema


def convert_mcp_tools_to_gemini(mcp_tools):
    
    gemini_tools = []
    
    for tool in mcp_tools:
        
        parameters = clean_schema(tool.inputSchema)
        
        function_declarations = FunctionDeclaration(
            name= tool.name,
            description=tool.description,
            parameters=parameters
        )
        
        gemini_tool = Tool(function_declarations=[function_declarations])
        gemini_tools.append(gemini_tool)
        
    return gemini_tools

# Pydantic models for request/response
class ConnectRequest(BaseModel):
    server_url: str


class ToolCallRequest(BaseModel):
    tool_name: str
    arguments: Optional[Dict[str, Any]] = None


class ResourceReadRequest(BaseModel):
    resource_uri: str


class QueryRequest(BaseModel):
    query: str


class ToolInfo(BaseModel):
    name: str
    description: str


class ResourceInfo(BaseModel):
    uri: str
    description: str


class ToolCallResponse(BaseModel):
    success: bool
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class ResourceReadResponse(BaseModel):
    success: bool
    content: Optional[str] = None
    mime_type: Optional[str] = None
    error: Optional[str] = None


class MCPSessionManager:
    """Manages MCP client session across FastAPI requests"""
    
    def __init__(self):
        self.server_url: Optional[str] = None
        self.session: Optional[ClientSession] = None
        self.client_context = None
        self.is_connected = False
        
        gemini_api_key = os.getenv("GOOGLE_API_KEY")
        if not gemini_api_key:
            raise ValueError("Enter Gemini Key. Add it to Your .env")
        
        self.genai_client = genai.Client(api_key=gemini_api_key)

    async def connect(self, server_url: str):
        """Establish connection to MCP server"""
        if self.is_connected:
            await self.disconnect()
        
        self.server_url = server_url
        
        try:
            # Create and enter the streamable HTTP client context
            self.client_context = streamablehttp_client(server_url)
            read_stream, write_stream, _ = await self.client_context.__aenter__()

            # Create and enter the client session context
            self.session = ClientSession(read_stream, write_stream)
            await self.session.__aenter__()

            # Initialize the connection
            await self.session.initialize()
            self.is_connected = True
            
            response = await self.session.list_tools()
            
            self.function_declarations = convert_mcp_tools_to_gemini(response.tools)
            
        except Exception as e:
            await self._cleanup()
            raise e
        
    async def process_query(self, query:str) -> str:
        user_prompt_content = types.Content(
            role='user',
            parts=[types.Part.from_text(text=query)]
        )
        
        response = self.genai_client.models.generate_content(
            model='gemini-2.0-flash-001',
            contents=[user_prompt_content],
            config=types.GenerateContentConfig(
                tools=self.function_declarations,
            ),
        )

        final_text = []
        assistant_message_content = []
        
        for candidate in response.candidates:
            if candidate.content.parts:
                for part in candidate.content.parts:
                    if isinstance(part, types.Part):
                        if part.function_call:
                            function_call_part = part
                            tool_name = function_call_part.function_call.name
                            tool_args = function_call_part.function_call.args
                            
                            print(f"\n[Gemini requested tool call: {tool_name} with args {tool_args}]")

                            try:
                                result = await self.session.call_tool(tool_name, tool_args)
                                function_response = {"result": result.content}
                            except Exception as e:
                                function_response = {"error": str(e)}
                                
                            function_response_part = types.Part.from_function_response(
                                name=tool_name,
                                response=function_response
                            )
                            
                            function_response_content = types.Content(
                                role='tool',
                                parts=[function_response_part]
                            )
                            
                            response = self.genai_client.models.generate_content(
                                model='gemini-2.0-flash-001',
                                contents=[
                                    user_prompt_content,
                                    function_call_part,
                                    function_response_content,
                                ],
                                config=types.GenerateContentConfig(
                                    tools=self.function_declarations,
                                ),
                            )
                            
                            final_text.append(response.candidates[0].content.parts[0].text)
                        else:
                            final_text.append(part.text)
                            
        return "\n".join(final_text)

    async def disconnect(self):
        """Close connection to MCP server"""
        await self._cleanup()

    async def _cleanup(self):
        """Clean up connections properly"""
        self.is_connected = False
        
        try:
            if self.session:
                await self.session.__aexit__(None, None, None)
                self.session = None
        except Exception:
            pass

        try:
            if self.client_context:
                await self.client_context.__aexit__(None, None, None)
                self.client_context = None
        except Exception:
            pass

    def ensure_connected(self):
        """Ensure session is connected"""
        if not self.is_connected or not self.session:
            raise HTTPException(status_code=400, detail="Not connected to MCP server. Please connect first.")

    async def list_tools(self) -> List[ToolInfo]:
        """List available tools"""
        self.ensure_connected()
        
        try:
            tools_result = await self.session.list_tools()
            return [ToolInfo(name=tool.name, description=tool.description) for tool in tools_result.tools]
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error listing tools: {str(e)}")

    async def list_resources(self) -> List[ResourceInfo]:
        """List available resources"""
        self.ensure_connected()
        
        try:
            resources_result = await self.session.list_resources()
            return [ResourceInfo(uri=res.uri, description=res.description) for res in resources_result.resources]
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error listing resources: {str(e)}")

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any] = None) -> Dict[str, Any]:
        """Call a tool on the server"""
        self.ensure_connected()
        
        arguments = arguments or {}
        
        try:
            result = await self.session.call_tool(tool_name, arguments)

            if result.content and len(result.content) > 0:
                content = result.content[0]
                if hasattr(content, "text"):
                    try:
                        parsed_result = json.loads(content.text)
                        return parsed_result
                    except json.JSONDecodeError:
                        return {"result": content.text}
                else:
                    return {"result": str(content)}
            else:
                return {"status": "success"}

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error calling tool: {str(e)}")

    async def read_resource(self, resource_uri: str) -> Tuple[str, str]:
        """Read a resource from the server"""
        self.ensure_connected()
        
        try:
            result = await self.session.read_resource(resource_uri)

            # Handle the response format properly
            if hasattr(result, "contents") and result.contents:
                content = result.contents[0]
                if hasattr(content, "text"):
                    content_text = content.text
                    mime_type = getattr(content, "mimeType", "text/plain")
                elif hasattr(content, "blob"):
                    content_text = content.blob
                    mime_type = getattr(content, "mimeType", "application/octet-stream")
                else:
                    content_text = str(content)
                    mime_type = "text/plain"
            else:
                content_text = str(result)
                mime_type = "text/plain"

            return content_text, mime_type

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error reading resource: {str(e)}")


# Global session manager
session_manager = MCPSessionManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan"""
    # Startup
    yield
    # Shutdown
    await session_manager.disconnect()


# Create FastAPI app
app = FastAPI(
    title="MCP Client API",
    description="FastAPI server for MCP (Model Context Protocol) client operations",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "MCP Client API Server",
        "connected": session_manager.is_connected,
        "server_url": session_manager.server_url
    }


@app.post("/connect")
async def connect(request: ConnectRequest):
    """Connect to MCP server"""
    try:
        await session_manager.connect(request.server_url)
        return {
            "success": True,
            "message": f"Connected to {request.server_url}",
            "server_url": request.server_url
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Connection failed: {str(e)}")


@app.post("/disconnect")
async def disconnect():
    """Disconnect from MCP server"""
    try:
        await session_manager.disconnect()
        return {
            "success": True,
            "message": "Disconnected from MCP server"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Disconnect failed: {str(e)}")

@app.get("/status")
async def get_status():
    """Get connection status"""
    return {
        "connected": session_manager.is_connected,
        "server_url": session_manager.server_url
    }


@app.get("/tools", response_model=List[ToolInfo])
async def list_tools():
    """List available tools from MCP server"""
    return await session_manager.list_tools()


@app.get("/resources", response_model=List[ResourceInfo])
async def list_resources():
    """List available resources from MCP server"""
    return await session_manager.list_resources()


@app.post("/tools/call", response_model=ToolCallResponse)
async def call_tool(request: ToolCallRequest):
    """Call a tool on the MCP server"""
    try:
        result = await session_manager.call_tool(request.tool_name, request.arguments)
        return ToolCallResponse(success=True, result=result)
    except HTTPException:
        raise
    except Exception as e:
        return ToolCallResponse(success=False, error=str(e))


@app.post("/resources/read", response_model=ResourceReadResponse)
async def read_resource(request: ResourceReadRequest):
    """Read a resource from the MCP server"""
    try:
        content, mime_type = await session_manager.read_resource(request.resource_uri)
        return ResourceReadResponse(success=True, content=content, mime_type=mime_type)
    except HTTPException:
        raise
    except Exception as e:
        return ResourceReadResponse(success=False, error=str(e))


@app.post("/query")
async def process_query(request: QueryRequest):
    """Process a query using Gemini with MCP tools"""
    try:
        result = await session_manager.process_query(request.query)
        return {
            "success": True,
            "response": result
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


# Additional convenience endpoints
@app.get("/tools/{tool_name}")
async def get_tool_info(tool_name: str):
    """Get information about a specific tool"""
    tools = await session_manager.list_tools()
    for tool in tools:
        if tool.name == tool_name:
            return tool
    raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found")


@app.get("/resources/{resource_uri:path}")
async def get_resource(resource_uri: str):
    """Get a specific resource (convenience endpoint)"""
    try:
        content, mime_type = await session_manager.read_resource(resource_uri)
        return {
            "uri": resource_uri,
            "content": content,
            "mime_type": mime_type
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)