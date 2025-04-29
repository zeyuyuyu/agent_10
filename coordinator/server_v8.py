import asyncio
from uuid import uuid4
from fastapi import FastAPI, Form, File, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import json
from typing import Optional, Dict, Any
from scheduler.llm_scheduler import LLMScheduler

app = FastAPI(title="Multi-Agent Scheduler API")
scheduler = LLMScheduler()

# ---- CORS（跨端口访问时需要） ----
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # 生产请写具体域名
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- 端点 1：一次性 JSON ----------
@app.post("/submit_task")
async def submit_task(text: str = Form(...), file: Optional[UploadFile] = File(None)):
    pdf = await file.read() if file else None
    result = scheduler.dispatch(
        context_id="ctx_once",       # 保留一次性用
        task=text,
        pdf_bytes=pdf,
        plain_text=None if pdf else text,
    )
    return JSONResponse(result)


# ---------- 端点 2：Server-Sent Events（流式） ----------
@app.post("/submit_task_stream")
async def submit_task_stream(text: str = Form(...), file: Optional[UploadFile] = File(None)):
    pdf = await file.read() if file else None
    queue: "asyncio.Queue[Dict[str, Any]]" = asyncio.Queue()

    # 每次生成唯一的 context_id
    ctx_id = f"ctx_{uuid4().hex}"

    # 获取当前事件循环
    loop = asyncio.get_running_loop()

    # 回调：调度器每有进度就丢进队列
    def push_progress(payload: Dict[str, Any]):
        # 把 payload 丢回主循环中的 queue
        loop.call_soon_threadsafe(queue.put_nowait, payload)

    # 在线程池跑调度器
    asyncio.create_task(asyncio.to_thread(
        scheduler.dispatch,
        context_id=ctx_id,            # 动态生成的 context_id
        task=text,
        pdf_bytes=pdf,
        plain_text=None if pdf else text,
        progress_cb=push_progress
    ))

    async def event_gen():
        while True:
            payload = await queue.get()
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
            if payload.get("status") == "done":
                break

    return StreamingResponse(event_gen(), media_type="text/event-stream")


# ---------- 静态前端 ----------
frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
app.mount("/static", StaticFiles(directory=frontend_dir, html=True), name="frontend")

@app.get("/")
async def index():
    return FileResponse(frontend_dir / "index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("coordinator.server:app", host="0.0.0.0", port=8080, reload=True)

