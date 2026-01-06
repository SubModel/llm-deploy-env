# llm_proxy.py - LLM 请求代理服务（关键词过滤和替换）
import json
import os
import re
import asyncio
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn
import argparse

# 配置
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:30001")  # 后端LLM服务地址

app = FastAPI(title="LLM Proxy")
client = httpx.AsyncClient(timeout=120.0)


def process_request_body(body: dict) -> dict:
    """处理请求体，过滤和替换关键词"""
    body = body.copy()
    model = body.get("model", "")
    
    # 针对 qwen3 模型的特殊处理
    if model in ["qwen3-max", "qwen3-max-thinking"]:
        # 处理 response_format
        if body.get("response_format", {}).get("type"):
            body.pop("response_format", None)
        
        # 替换消息内容中的标签（仅针对 qwen3-max）
        if model == "qwen3-max" and isinstance(body.get("messages"), list):
            for message in body["messages"]:
                content = message.get("content")
                if isinstance(content, str):
                    # 替换 <think> 为 <thinking>
                    content = re.sub(r"<think>", "<thinking>", content, flags=re.IGNORECASE)
                    content = re.sub(r"</think>", "</thinking>", content, flags=re.IGNORECASE)
                    message["content"] = content
    
    return body


async def forward_to_backend(path: str, request: Request, body: dict = None):
    """转发请求到后端服务（非流式）"""
    url = f"{BACKEND_URL}{path}"
    headers = {k: v for k, v in request.headers.items()
               if k.lower() not in ("host", "content-length")}
    
    if body is None:
        body = await request.json()
    
    try:
        response = await client.post(url, json=body, headers=headers)
        return response
    except httpx.TimeoutException:
        print(f"⚠️ 请求超时: {url}")
        raise
    except httpx.RequestError as e:
        print(f"⚠️ 请求失败: {e}")
        raise


async def forward_to_backend_stream(path: str, request: Request, body: dict = None):
    """转发流式请求到后端服务"""
    url = f"{BACKEND_URL}{path}"
    headers = {k: v for k, v in request.headers.items()
               if k.lower() not in ("host", "content-length")}
    
    if body is None:
        body = await request.json()
    
    async def generate():
        try:
            async with client.stream("POST", url, json=body, headers=headers, timeout=120.0) as response:
                if response.status_code >= 400:
                    error_text = await response.aread()
                    try:
                        error_info = json.loads(error_text.decode())
                        yield json.dumps(error_info, ensure_ascii=False).encode()
                    except:
                        error_msg = error_text[:200] if len(error_text) > 200 else error_text
                        yield json.dumps({"error": {"message": error_msg.decode(errors="ignore")}}, 
                                       ensure_ascii=False).encode()
                    return
                
                async for chunk in response.aiter_bytes():
                    yield chunk
        except httpx.TimeoutException:
            print(f"⚠️ 流式请求超时: {url}")
            yield json.dumps({"error": {"message": "请求超时"}}, ensure_ascii=False).encode()
        except httpx.RequestError as e:
            print(f"⚠️ 流式请求失败: {e}")
            yield json.dumps({"error": {"message": f"请求失败: {str(e)}"}}, ensure_ascii=False).encode()
    
    return generate


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """聊天补全端点 - 主要处理逻辑"""
    body = await request.json()
    stream = body.get("stream", False)
    
    # 处理请求体（过滤和替换关键词）
    processed_body = process_request_body(body)
    
    # 流式处理
    if stream:
        try:
            stream_generator_func = await forward_to_backend_stream("/v1/chat/completions", request, processed_body)
            return StreamingResponse(
                stream_generator_func(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no"
                }
            )
        except Exception as e:
            print(f"⚠️ 流式请求处理失败: {e}")
            return JSONResponse(
                content={"error": {"message": f"流式请求处理失败: {str(e)}"}},
                status_code=500
            )
    
    # 非流式处理
    try:
        response = await forward_to_backend("/v1/chat/completions", request, processed_body)
        if response.status_code >= 400:
            error_result = response.json() if response.headers.get("content-type", "").startswith("application/json") \
                          else {"error": {"message": response.text[:200]}}
            return JSONResponse(content=error_result, status_code=response.status_code)
        
        return JSONResponse(response.json())
    except Exception as e:
        print(f"⚠️ 非流式请求处理失败: {e}")
        return JSONResponse(
            content={"error": {"message": f"请求处理失败: {str(e)}"}},
            status_code=500
        )


@app.get("/health")
async def health():
    """健康检查端点"""
    return {"status": "ok", "backend_url": BACKEND_URL}


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD", "PATCH"])
async def proxy_other_endpoints(path: str, request: Request):
    """其他请求直接透传"""
    url = f"{BACKEND_URL}/{path}"
    headers = {k: v for k, v in request.headers.items()
               if k.lower() not in ("host", "content-length")}
    
    try:
        if request.method == "GET":
            response = await client.get(url, headers=headers)
        else:
            body = await request.body()
            response = await client.request(request.method, url, content=body, headers=headers)
        
        # 尝试返回 JSON，如果失败则返回原始内容
        try:
            return JSONResponse(response.json(), status_code=response.status_code)
        except:
            return JSONResponse({"content": response.text}, status_code=response.status_code)
    except httpx.ConnectError as e:
        return JSONResponse(
            content={"error": {"message": f"无法连接到后端服务 {BACKEND_URL}: {str(e)}"}},
            status_code=503
        )
    except Exception as e:
        return JSONResponse(
            content={"error": {"message": f"请求失败: {str(e)}"}},
            status_code=500
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Proxy Server")
    parser.add_argument("-port", type=int, default=30000, help="代理服务端口 (默认: 30000)")
    parser.add_argument("-backend", type=str, default=BACKEND_URL, help="后端LLM服务地址")
    args = parser.parse_args()
    
    # 更新后端URL
    BACKEND_URL = args.backend
    
    print("🚀 启动 LLM 代理服务")
    print(f"   代理地址: http://0.0.0.0:{args.port}")
    print(f"   后端服务: {BACKEND_URL}")
    print(f"   功能: 关键词过滤和替换")
    
    uvicorn.run(app, host="0.0.0.0", port=args.port)
