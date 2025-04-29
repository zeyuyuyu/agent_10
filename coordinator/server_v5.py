from fastapi import FastAPI, File, Form, UploadFile
from typing import Optional, List
from pydantic import BaseModel
from starlette.responses import StreamingResponse
from scheduler.llm_scheduler import LLMScheduler  # 引用调度器

app = FastAPI(title="Multi-Agent Scheduler API")
scheduler = LLMScheduler()

# ------------------ Pydantic 模型 ------------------
class TraceItem(BaseModel):
    agent: str
    subtask: str
    output: str

class DispatchResp(BaseModel):
    markdown: str
    trace: List[TraceItem]

# ------------------ 路由 ------------------
@app.post("/submit_task", response_model=DispatchResp)
async def submit_task(
    text: str = Form(...),
    file: Optional[UploadFile] = File(None)
):
    """
    处理上传的 PDF 或任务文本，返回任务状态（包括子任务分配和处理状态）
    """
    pdf_bytes = await file.read() if file else None

    # 创建流式响应
    async def generate_responses():
        # 1. 获取子任务列表并返回给前端
        result = scheduler.dispatch(
            context_id="ctx001",
            task=text,
            pdf_bytes=pdf_bytes,
            plain_text=None if pdf_bytes else text,
            progress_cb=push_progress  # 回调函数实时推送状态给前端
        )
        yield f"data: {result}\n\n"  # 将子任务列表或初始数据返回给前端
        
        # 模拟多个中间过程，每次都推送状态
        for trace in scheduler.trace:
            # 模拟每个代理的处理进度更新
            yield f"data: {trace}\n\n"
    
    return StreamingResponse(generate_responses(), media_type="text/event-stream")


def push_progress(payload: dict):
    """
    通过此回调将调度器当前进度推送到前端
    - `status`: 可以是 'subtasks'（子任务列表）、'assign'（任务分配）或 'done'（任务完成）
    - `subtasks`: 子任务列表
    - `agent`: 当前任务分配的代理
    - `subtask`: 当前处理的子任务
    - `output`: 当前状态，'Processing' 或最终结果
    """
    # 在这里你可以实时更新前端状态
    print(f"Progress Update: {payload}")
    # 例如，你可以将该信息存储在数据库、缓存或者通过 WebSocket 推送给前端

