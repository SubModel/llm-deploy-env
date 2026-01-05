# bocha_server.py - 透明代理 + MCP 工具调用支持（连接池 + 并行化）
import json
import os
import sys
import asyncio
import httpx
import logging
import uuid
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn
import argparse

# MCP 客户端
from mcp import ClientSession
from mcp.client.sse import sse_client

# 过滤 MCP heartbeat 警告
logging.getLogger("root").setLevel(logging.ERROR)

# 配置
SGLANG_BASE_URL_TOOL = os.getenv("MODEL_SERVER_TOOL", os.getenv("MODEL_SERVER", "http://localhost:30001"))
SGLANG_BASE_URL_CHAT = os.getenv("MODEL_SERVER_CHAT", os.getenv("MODEL_SERVER", "http://localhost:30001"))
MODEL_NAME_TOOL = os.getenv("MODEL_NAME_TOOL", "")  # 工具判断模型名
BOCHA_API_KEY = os.getenv("BOCHA_API_KEY", "sk-fb4d4d4481924d4bbe48499fd52d55bb")
BOCHA_MCP_URL = "https://mcp.bocha.cn/sse"
MCP_POOL_SIZE = int(os.getenv("MCP_POOL_SIZE", "2"))  # 连接池大小

# 全局 MCP 连接池管理
mcp_tools: list = []
_mcp_pool: list[ClientSession] = []
_mcp_pool_lock = asyncio.Lock()
_pool_semaphore: asyncio.Semaphore | None = None
_connection_tasks: list[asyncio.Task] = []
_connection_states: dict[int, dict] = {}  # 连接状态跟踪 {connection_id: {"last_heartbeat": time, "reconnecting": bool}}
HEARTBEAT_INTERVAL = 60  # 心跳间隔（秒），1分钟
HEARTBEAT_TIMEOUT = 10  # 心跳超时时间（秒）
RECONNECT_DELAY = 5  # 重连延迟（秒）


async def _check_connection_health(session: ClientSession) -> bool:
    """检查连接健康状态（心跳检测）"""
    try:
        # 使用轻量级的 list_tools 作为心跳检测
        await asyncio.wait_for(session.list_tools(), timeout=HEARTBEAT_TIMEOUT)
        return True
    except (asyncio.TimeoutError, Exception) as e:
        print(f"💔 连接健康检查失败: {e}")
        return False


async def _maintain_single_connection(connection_id: int):
    """维护单个 MCP 连接的后台任务（带自动重连和心跳检测）"""
    global mcp_tools, _connection_states
    headers = {"Authorization": f"Bearer {BOCHA_API_KEY}"}
    
    # 初始化连接状态
    _connection_states[connection_id] = {
        "last_heartbeat": time.time(),
        "reconnecting": False
    }
    
    # 错峰建立连接,避免并发连接超时
    await asyncio.sleep(connection_id * 2)
    
    retry_count = 0
    max_retries = 10  # 最大重试次数（0表示无限重试，但记录次数）
    
    while True:
        session = None
        try:
            # 检查是否应该重连
            if _connection_states[connection_id]["reconnecting"]:
                await asyncio.sleep(RECONNECT_DELAY)
            
            print(f"🔗 [{connection_id}] 正在建立 MCP 连接...")
            _connection_states[connection_id]["reconnecting"] = False
            
            async with sse_client(BOCHA_MCP_URL, headers=headers) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()

                    # 第一个连接获取工具列表
                    if connection_id == 0:
                        tools_result = await session.list_tools()
                        mcp_tools = tools_result.tools
                        print(f"✅ [{connection_id}] MCP 连接已建立，工具: {[t.name for t in mcp_tools]}")
                    else:
                        print(f"✅ [{connection_id}] MCP 连接已建立")

                    # 添加到连接池
                    async with _mcp_pool_lock:
                        if session not in _mcp_pool:
                            _mcp_pool.append(session)

                    # 保持连接并定期心跳检测
                    last_heartbeat_time = time.time()
                    while True:
                        await asyncio.sleep(min(HEARTBEAT_INTERVAL, 60))  # 最多等待60秒检查一次
                        
                        # 检查是否需要心跳检测
                        elapsed = time.time() - last_heartbeat_time
                        if elapsed >= HEARTBEAT_INTERVAL:
                            is_healthy = await _check_connection_health(session)
                            last_heartbeat_time = time.time()
                            _connection_states[connection_id]["last_heartbeat"] = last_heartbeat_time
                            
                            if not is_healthy:
                                print(f"⚠️ [{connection_id}] 连接健康检查失败，准备重连...")
                                # 从连接池移除失效连接
                                async with _mcp_pool_lock:
                                    if session in _mcp_pool:
                                        _mcp_pool.remove(session)
                                break  # 跳出内层循环，触发重连
                            else:
                                print(f"💚 [{connection_id}] 连接健康检查通过")
                        
                        # 检查是否被取消
                        if asyncio.current_task().cancelled():
                            raise asyncio.CancelledError()

        except asyncio.CancelledError:
            print(f"🔌 [{connection_id}] MCP 连接任务被取消")
            # 确保从连接池移除
            async with _mcp_pool_lock:
                if session and session in _mcp_pool:
                    _mcp_pool.remove(session)
            raise
        except Exception as e:
            retry_count += 1
            print(f"❌ [{connection_id}] MCP 连接错误 (重试 {retry_count}): {e}")
            
            # 从连接池移除失效连接
            async with _mcp_pool_lock:
                if session and session in _mcp_pool:
                    _mcp_pool.remove(session)
            
            # 标记为正在重连
            _connection_states[connection_id]["reconnecting"] = True
            
            # 如果连接池为空且这是第一个连接，等待更长时间
            async with _mcp_pool_lock:
                if len(_mcp_pool) == 0 and connection_id == 0:
                    print(f"⏳ [{connection_id}] 连接池为空，等待 {RECONNECT_DELAY * 2} 秒后重连...")
                    await asyncio.sleep(RECONNECT_DELAY * 2)
                else:
                    await asyncio.sleep(RECONNECT_DELAY)


async def init_mcp_pool():
    """初始化 MCP 连接池"""
    global _connection_tasks, _pool_semaphore

    print(f"🚀 正在初始化 MCP 连接池 (大小: {MCP_POOL_SIZE})...")
    _pool_semaphore = asyncio.Semaphore(MCP_POOL_SIZE)

    _connection_tasks = [
        asyncio.create_task(_maintain_single_connection(i))
        for i in range(MCP_POOL_SIZE)
    ]

    # 等待连接建立（最多 30 秒）
    max_wait = 30
    start = asyncio.get_event_loop().time()
    while len(_mcp_pool) < MCP_POOL_SIZE:
        if asyncio.get_event_loop().time() - start > max_wait:
            break
        await asyncio.sleep(0.1)
        
        # 检查任务是否失败，如果失败则重启
        for i, task in enumerate(_connection_tasks):
            if task.done():
                try:
                    task.result()
                except (asyncio.CancelledError, Exception) as e:
                    print(f"⚠️ 连接任务 [{i}] 初始化时失败: {e}，正在重启...")
                    _connection_tasks[i] = asyncio.create_task(_maintain_single_connection(i))

    if len(_mcp_pool) == 0:
        print(f"⚠️ 警告: 初始化时无法建立任何 MCP 连接，但连接任务将继续在后台重试")
    else:
        print(f"✅ MCP 连接池初始化完成 ({len(_mcp_pool)}/{MCP_POOL_SIZE} 个连接)")


async def close_mcp_pool():
    """关闭 MCP 连接池"""
    global _mcp_pool, _connection_tasks

    print("🔌 正在关闭 MCP 连接池...")
    for task in _connection_tasks:
        if not task.done():
            task.cancel()

    await asyncio.gather(*_connection_tasks, return_exceptions=True)
    _mcp_pool = []
    _connection_tasks = []
    print("✅ MCP 连接池已关闭")


async def _ensure_connection_tasks_running():
    """确保所有连接任务正在运行，如果任务失败则重启"""
    global _connection_tasks
    
    if _pool_semaphore is None:
        return False  # 连接池未初始化
    
    # 检查并重启失败的任务
    for i, task in enumerate(_connection_tasks):
        if task.done():
            # 任务已完成（可能是失败或取消）
            try:
                task.result()  # 获取结果，如果有异常会抛出
            except asyncio.CancelledError:
                # 任务被取消，这是正常的关闭流程，不重启
                continue
            except Exception as e:
                print(f"⚠️ 连接任务 [{i}] 已停止: {e}，正在重启...")
                # 重启任务
                _connection_tasks[i] = asyncio.create_task(_maintain_single_connection(i))
    
    return True


async def get_mcp_session() -> ClientSession:
    """从连接池获取一个可用的 MCP 连接（带健康检查和等待重连）"""
    # 确保连接任务正在运行
    if not await _ensure_connection_tasks_running():
        raise RuntimeError("MCP 连接池未初始化或已关闭")
    
    # 等待连接建立（最多等待30秒）
    max_wait_time = 30
    start_time = time.time()
    
    while len(_mcp_pool) == 0:
        if time.time() - start_time > max_wait_time:
            raise RuntimeError("MCP 连接池为空，等待连接建立超时")
        print(f"⏳ 等待 MCP 连接建立... (已等待 {int(time.time() - start_time)} 秒)")
        await asyncio.sleep(1)
        # 再次检查任务状态
        await _ensure_connection_tasks_running()
    
    task_id = id(asyncio.current_task())
    max_attempts = max(len(_mcp_pool) * 2, 10)  # 至少尝试10次
    
    for attempt in range(max_attempts):
        # 如果连接池为空，等待一下
        if len(_mcp_pool) == 0:
            await asyncio.sleep(0.5)
            await _ensure_connection_tasks_running()
            continue
        
        idx = (task_id + attempt) % len(_mcp_pool)
        session = _mcp_pool[idx]
        
        # 快速健康检查（使用较短的超时）
        try:
            await asyncio.wait_for(session.list_tools(), timeout=2)
            return session
        except (asyncio.TimeoutError, Exception) as e:
            # 连接失效，从池中移除
            print(f"⚠️ 检测到失效连接 [{idx}]，正在移除: {e}")
            async with _mcp_pool_lock:
                if session in _mcp_pool:
                    _mcp_pool.remove(session)
            # 继续尝试下一个连接
            if len(_mcp_pool) == 0:
                # 连接池又空了，等待一下
                await asyncio.sleep(0.5)
                await _ensure_connection_tasks_running()
            continue
    
    raise RuntimeError("无法获取可用的 MCP 连接")


def get_openai_tools_from_mcp():
    """将 MCP 工具转换为 OpenAI tools 格式（仅 bocha_web_search）"""
    openai_tools = []
    for tool in mcp_tools:
        # 只使用 bocha_web_search
        if tool.name == "bocha_web_search":
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": getattr(tool, "description", "") or "",
                    "parameters": getattr(tool, "inputSchema", {"type": "object", "properties": {}})
                }
            })
    return openai_tools


async def call_mcp_tool(tool_name: str, arguments: dict) -> str:
    """通过连接池调用 MCP 工具（带信号量限流和连接健康检查）"""
    if _pool_semaphore is None:
        raise RuntimeError("MCP 连接池未初始化")

    max_retries = 5  # 增加重试次数，给重连更多时间
    for attempt in range(max_retries):
        try:
            async with _pool_semaphore:
                session = await get_mcp_session()  # 这里已经包含健康检查和等待重连
                result = await session.call_tool(tool_name, arguments)
                if result.content:
                    contents = []
                    for item in result.content:
                        if hasattr(item, "text"):
                            contents.append(item.text)
                    return "\n".join(contents) if contents else str(result)
                return str(result)
        except RuntimeError as e:
            # 连接池问题，等待后重试
            error_msg = str(e)
            print(f"⚠️ MCP 连接池问题 (尝试 {attempt + 1}/{max_retries}): {error_msg}")
            
            # 确保连接任务正在运行
            await _ensure_connection_tasks_running()
            
            if attempt < max_retries - 1:
                wait_time = min(2.0 * (attempt + 1), 10.0)  # 指数退避，最多10秒
                print(f"⏳ 等待 {wait_time:.1f} 秒后重试...")
                await asyncio.sleep(wait_time)
            else:
                return f"MCP 调用错误: 连接池不可用 - {error_msg}"
        except Exception as e:
            print(f"⚠️ MCP 调用失败 (尝试 {attempt + 1}/{max_retries}): {e}")
            
            # 确保连接任务正在运行
            await _ensure_connection_tasks_running()
            
            if attempt < max_retries - 1:
                await asyncio.sleep(0.5)
            else:
                return f"MCP 调用错误: {str(e)}"


async def call_mcp_tool_batch(tool_calls: list) -> dict:
    """并行调用多个 MCP 工具"""
    tasks = []
    tool_call_ids = []
    for tool_call in tool_calls:
        tool_name = tool_call["function"]["name"]
        arguments = json.loads(tool_call["function"]["arguments"])
        tool_call_ids.append(tool_call["id"])
        tasks.append(call_mcp_tool(tool_name, arguments))

    print(f"🔧 并行调用 {len(tasks)} 个 MCP 工具...")
    results = await asyncio.gather(*tasks, return_exceptions=True)

    tool_results = {}
    for tool_call_id, result in zip(tool_call_ids, results):
        if isinstance(result, Exception):
            tool_results[tool_call_id] = f"错误: {str(result)}"
        else:
            tool_results[tool_call_id] = result
    return tool_results


# FastAPI 生命周期
@asynccontextmanager
async def lifespan(app: FastAPI):
    if BOCHA_API_KEY:
        try:
            await init_mcp_pool()
        except Exception as e:
            print(f"⚠️ MCP 初始化失败: {e}")
    yield
    await close_mcp_pool()


app = FastAPI(title="Bocha MCP Proxy", lifespan=lifespan)
client = httpx.AsyncClient(timeout=120.0)


async def forward_to_sglang(path: str, request: Request, body: dict | None = None, use_tool_model: bool = False):
    """透传请求到 SGLang（非流式）
    
    Args:
        path: API路径
        request: FastAPI请求对象
        body: 请求体（可选）
        use_tool_model: 如果为True，使用工具判断模型；如果为False，使用回答用户的模型
    """
    base_url = SGLANG_BASE_URL_TOOL if use_tool_model else SGLANG_BASE_URL_CHAT
    url = f"{base_url}{path}"
    headers = {k: v for k, v in request.headers.items()
               if k.lower() not in ("host", "content-length")}
    if body is None:
        body = await request.json()
    try:
        response = await client.post(url, json=body, headers=headers)
        # 检查响应状态码
        if response.status_code >= 400:
            print(f"⚠️  SGLang返回错误状态码: {response.status_code}")
            # 尝试读取错误信息
            try:
                error_info = response.json()
                print(f"   错误详情: {error_info}")
            except:
                print(f"   错误内容: {response.text[:200]}")
        return response
    except httpx.TimeoutException:
        print(f"⚠️  请求SGLang超时: {url}")
        raise
    except httpx.RequestError as e:
        print(f"⚠️  请求SGLang失败: {e}")
        raise


async def forward_to_sglang_stream(path: str, request: Request, body: dict | None = None, use_tool_model: bool = False):
    """透传流式请求到 SGLang（真正的流式传输）
    
    Args:
        path: API路径
        request: FastAPI请求对象
        body: 请求体（可选）
        use_tool_model: 如果为True，使用工具判断模型；如果为False，使用回答用户的模型
    """
    base_url = SGLANG_BASE_URL_TOOL if use_tool_model else SGLANG_BASE_URL_CHAT
    url = f"{base_url}{path}"
    headers = {k: v for k, v in request.headers.items()
               if k.lower() not in ("host", "content-length")}
    if body is None:
        body = await request.json()
    
    async def generate():
        try:
            async with client.stream("POST", url, json=body, headers=headers, timeout=120.0) as response:
                # 检查响应状态码
                if response.status_code >= 400:
                    # 对于错误响应，读取完整错误信息并返回
                    error_text = await response.aread()
                    try:
                        error_info = json.loads(error_text.decode())
                        error_json = json.dumps(error_info, ensure_ascii=False).encode()
                        yield error_json
                    except:
                        error_msg = error_text[:200] if len(error_text) > 200 else error_text
                        error_json = json.dumps({"error": {"message": error_msg.decode(errors="ignore")}}, ensure_ascii=False).encode()
                        yield error_json
                    return
                
                # 流式传输响应数据
                async for chunk in response.aiter_bytes():
                    yield chunk
        except httpx.TimeoutException:
            print(f"⚠️  流式请求SGLang超时: {url}")
            error_json = json.dumps({"error": {"message": "请求超时"}}, ensure_ascii=False).encode()
            yield error_json
        except httpx.RequestError as e:
            print(f"⚠️  流式请求SGLang失败: {e}")
            error_json = json.dumps({"error": {"message": f"请求失败: {str(e)}"}}, ensure_ascii=False).encode()
            yield error_json
    
    return generate


async def quick_detect_web_search_query(request: Request, body: dict) -> str | None:
    """
    快速检测web_search工具的query参数
    
    使用轻量级检测请求，只生成工具调用，不生成完整答案
    
    Args:
        request: FastAPI请求对象
        body: 请求体，可能包含search_options.forced_search参数
    
    Returns:
        str | None: 如果检测到web_search工具调用，返回query参数；否则返回None
    """
    # 检查enable_search开关
    if not body.get("enable_search", False):
        return None
    
    # 检查是否有MCP工具可用
    if not mcp_tools:
        return None
    
    # 检查forced_search参数（从search_options中读取）
    search_options = body.get("search_options", {})
    forced_search = search_options.get("forced_search", False)
    
    try:
        # 创建检测请求
        detect_body = body.copy()
        detect_messages = body.get("messages", []).copy()
        
        # 根据forced_search参数决定system message的内容
        if forced_search:
            # 强制搜索模式：明确要求必须调用搜索工具
            system_content = "你必须调用bocha_web_search工具进行联网搜索，生成搜索词条并调用工具，只生成工具调用，不要生成其他内容。"
        else:
            # 默认模式：让模型判断是否需要搜索
            system_content = "你需要判断是否需要搜索信息。如果需要搜索，请调用bocha_web_search工具，只生成工具调用，不要生成其他内容。"
        
        # 添加特殊的system message，引导模型只生成工具调用
        # 检查是否已有system message
        has_system = any(msg.get("role") == "system" for msg in detect_messages)
        if not has_system:
            detect_messages.insert(0, {
                "role": "system",
                "content": system_content
            })
        else:
            # 如果有system message，修改第一个system message
            for i, msg in enumerate(detect_messages):
                if msg.get("role") == "system":
                    detect_messages[i] = {
                        "role": "system",
                        "content": msg.get("content", "") + f"\n\n重要：{system_content}"
                    }
                    break
        
        detect_body["messages"] = detect_messages
        detect_body["stream"] = False
        # 如果设置了MODEL_NAME_TOOL，使用指定的模型
        if MODEL_NAME_TOOL:
            detect_body["model"] = MODEL_NAME_TOOL
        # 限制最大token数，快速检测（使用较小的值，但不超过原始max_tokens）
        original_max_tokens = body.get("max_tokens")
        if original_max_tokens:
            detect_body["max_tokens"] = min(original_max_tokens, 100)
        else:
            detect_body["max_tokens"] = 100  # 如果没有设置，使用100作为快速检测的限制
        
        # 检查tool_choice，如果明确设置为"none"且不是强制搜索模式，则不进行检测
        if detect_body.get("tool_choice") == "none" and not forced_search:
            print("ℹ️  tool_choice设置为none，跳过快速检测")
            return None
        
        # 确保工具定义已注入
        if detect_body.get("tools") is None:
            detect_body["tools"] = get_openai_tools_from_mcp()
        elif detect_body.get("tools"):
            # 确保bocha_web_search在工具列表中
            existing_tool_names = [t.get("function", {}).get("name") for t in detect_body.get("tools", [])]
            if "bocha_web_search" not in existing_tool_names:
                detect_body["tools"].extend(get_openai_tools_from_mcp())
        
        # 根据forced_search参数设置tool_choice
        if forced_search:
            # 强制搜索模式：强制要求调用工具
            detect_body["tool_choice"] = {"type": "function", "function": {"name": "bocha_web_search"}}
            print("🔍 强制联网搜索模式：强制调用web_search工具...")
        else:
            # 默认模式：让模型自动判断
            detect_body["tool_choice"] = "auto"
            print("🔍 快速检测web_search query（模型判断是否需要搜索）...")
        detect_start_time = time.time()
        response = await forward_to_sglang("/v1/chat/completions", request, detect_body, use_tool_model=True)
        detect_duration = time.time() - detect_start_time
        result = response.json()
        print(f"⏱️  快速检测耗时: {detect_duration:.2f}秒")
        
        # 解析响应，提取工具调用
        if result.get("choices") and result["choices"][0].get("message", {}).get("tool_calls"):
            tool_calls = result["choices"][0]["message"]["tool_calls"]
            
            # 查找bocha_web_search工具调用
            for tool_call in tool_calls:
                if tool_call.get("function", {}).get("name") == "bocha_web_search":
                    arguments = json.loads(tool_call.get("function", {}).get("arguments", "{}"))
                    query = arguments.get("query", "")
                    if query:
                        print(f"✅ 检测到web_search query: {query[:100]}...")
                        return query
        
        print("ℹ️  未检测到web_search工具调用")
        return None
        
    except Exception as e:
        print(f"⚠️  快速检测失败: {e}")
        return None


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """
    智能路由:
    1. 检查enable_search开关，决定是否进行快速检测
    2. 如果enable_search=true，根据search_options.forced_search参数决定搜索行为：
       - search_options.forced_search=true: 强制联网搜索，跳过模型判断
       - search_options.forced_search=false (默认): 模型判断是否需要联网搜索
    3. 执行web_search工具后，继续对话
    4. 支持真正的流式和非流式模式
    
    请求结构示例:
    {
        "enable_search": True,
        "search_options": {
            "forced_search": True  # 强制联网搜索
        }
    }
    """
    body = await request.json()
    messages = body.get("messages", [])
    stream = body.get("stream", False)
    enable_search = body.get("enable_search", False)
    # 从search_options中读取forced_search参数
    search_options = body.get("search_options", {})
    forced_search = search_options.get("forced_search", False)

    # 如果enable_search=true，进行快速检测并执行搜索
    if enable_search:
        search_start_time = time.time()
        query = await quick_detect_web_search_query(request, body)
        
        if query:
            try:
                # 执行web_search工具
                search_mode = "强制联网搜索" if forced_search else "模型判断搜索"
                print(f"🔍 [{search_mode}] 执行web_search: {query[:100]}...")
                mcp_start_time = time.time()
                search_result = await call_mcp_tool("bocha_web_search", {"query": query})
                mcp_duration = time.time() - mcp_start_time
                print(f"⏱️  MCP web_search执行耗时: {mcp_duration:.2f}秒")
                
                # 检查搜索结果是否包含错误信息
                if search_result.startswith("MCP 调用错误"):
                    print(f"⚠️  web_search执行失败: {search_result}")
                    # 即使失败，也添加到messages中，让模型知道搜索失败了
                    search_result = f"搜索失败: {search_result}"
                
                # 构造工具调用消息（模拟工具调用的格式）
                tool_call_id = f"call_{uuid.uuid4().hex[:8]}"
                
                # 添加assistant消息（包含工具调用）
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": tool_call_id,
                        "type": "function",
                        "function": {
                            "name": "bocha_web_search",
                            "arguments": json.dumps({"query": query})
                        }
                    }]
                })
                
                # 添加工具结果消息
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": search_result
                })
                
                total_duration = time.time() - search_start_time
                detect_duration = total_duration - mcp_duration
                print(f"📄 搜索结果已添加到messages，长度: {len(search_result)}")
                print(f"⏱️  搜索流程总耗时: {total_duration:.2f}秒 (快速检测: {detect_duration:.2f}秒, MCP执行: {mcp_duration:.2f}秒)")
                
                # 更新body中的messages
                body["messages"] = messages
                # 移除工具定义，因为已经执行完毕
                body.pop("tools", None)
                body.pop("tool_choice", None)
                body.pop("enable_search", None)  # 移除enable_search，避免重复检测
                body.pop("search_options", None)  # 移除search_options，避免重复检测
            except Exception as e:
                print(f"⚠️  web_search执行异常: {e}")
                # 发生异常时，移除enable_search和search_options，直接透传
                body.pop("enable_search", None)
                body.pop("search_options", None)
        else:
            # 检测失败或未检测到query
            if forced_search:
                print("⚠️  强制搜索模式下未找到query，可能存在问题")
            else:
                print("ℹ️  模型判断不需要搜索，直接透传请求")
            body.pop("enable_search", None)
            body.pop("search_options", None)

    # 流式处理
    if stream:
        # 直接发送流式请求（如果已经执行了搜索，messages中已包含搜索结果）
        try:
            stream_generator_func = await forward_to_sglang_stream("/v1/chat/completions", request, body, use_tool_model=False)
            return StreamingResponse(
                stream_generator_func(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no"  # 禁用nginx缓冲
                }
            )
        except Exception as e:
            print(f"⚠️  流式请求处理失败: {e}")
            return JSONResponse(
                content={"error": {"message": f"流式请求处理失败: {str(e)}"}},
                status_code=500
            )

    # 非流式处理
    try:
        response = await forward_to_sglang("/v1/chat/completions", request, body, use_tool_model=False)
        # 检查响应状态码
        if response.status_code >= 400:
            error_result = response.json() if response.headers.get("content-type", "").startswith("application/json") else {"error": {"message": response.text[:200]}}
            return JSONResponse(content=error_result, status_code=response.status_code)
        
        result = response.json()
    except Exception as e:
        print(f"⚠️  非流式请求处理失败: {e}")
        return JSONResponse(
            content={"error": {"message": f"请求处理失败: {str(e)}"}},
            status_code=500
        )
    
    # 检查是否有工具调用（这种情况应该很少，因为enable_search时我们已经提前处理了）
    if result.get("choices") and result["choices"][0].get("message", {}).get("tool_calls"):
        tool_calls = result["choices"][0]["message"]["tool_calls"]
        messages.append(result["choices"][0]["message"])

        print(f"🔧 MCP 调用: 并行执行 {len(tool_calls)} 个工具")
        tool_results = await call_mcp_tool_batch(tool_calls)

        for tool_call in tool_calls:
            print(f"📄 MCP 结果: {tool_results[tool_call['id']][:200]}...")
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": tool_results[tool_call["id"]]
            })

        body["messages"] = messages
        body.pop("tools", None)
        body.pop("tool_choice", None)

        final_response = await forward_to_sglang("/v1/chat/completions", request, body, use_tool_model=False)
        return JSONResponse(final_response.json())

    return JSONResponse(result)


@app.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy_other_endpoints(path: str, request: Request):
    """其他请求直接透传（默认使用回答用户的模型）"""
    url = f"{SGLANG_BASE_URL_CHAT}/v1/{path}"
    headers = {k: v for k, v in request.headers.items()
               if k.lower() not in ("host", "content-length")}

    if request.method == "GET":
        response = await client.get(url, headers=headers)
    else:
        body = await request.body()
        response = await client.request(request.method, url, content=body, headers=headers)

    return JSONResponse(response.json())


@app.get("/health")
async def health():
    """健康检查端点，包含连接池状态信息"""
    connection_states = []
    for conn_id, state in _connection_states.items():
        connection_states.append({
            "connection_id": conn_id,
            "last_heartbeat": state["last_heartbeat"],
            "reconnecting": state["reconnecting"],
            "time_since_heartbeat": time.time() - state["last_heartbeat"]
        })
    
    return {
        "status": "ok",
        "mcp_pool_size": len(_mcp_pool),
        "mcp_pool_target": MCP_POOL_SIZE,
        "mcp_tools": [t.name for t in mcp_tools] if mcp_tools else [],
        "connection_states": connection_states,
        "heartbeat_interval": HEARTBEAT_INTERVAL
    }


if __name__ == "__main__":
    if not BOCHA_API_KEY:
        print("⚠️ 警告: BOCHA_API_KEY 未设置，MCP 工具不可用")

    # Command line argument parsing
    parser = argparse.ArgumentParser(description="Bocha MCP Proxy Server")
    parser.add_argument("-port", type=int, default=30000, help="Port to run the proxy server on (default: 30000)")
    args = parser.parse_args()

    print("🚀 Starting Bocha MCP Proxy Service (Connection Pool + Parallelization Mode)")
    print(f"   Proxy Address: http://0.0.0.0:{args.port}")
    print(f"   Backend SGLang (Tool): {SGLANG_BASE_URL_TOOL}")
    print(f"   Backend SGLang (Chat): {SGLANG_BASE_URL_CHAT}")
    if MODEL_NAME_TOOL:
        print(f"   Model Name (Tool): {MODEL_NAME_TOOL}")
    print(f"   Bocha MCP: {BOCHA_MCP_URL}")
    print(f"   Connection Pool Size: {MCP_POOL_SIZE}")

    uvicorn.run(app, host="0.0.0.0", port=args.port)