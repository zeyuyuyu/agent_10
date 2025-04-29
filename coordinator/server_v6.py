from fastapi import FastAPI, Form, File, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import asyncio, json
from typing import Optional, Dict, Any

from scheduler.llm_scheduler import LLMScheduler   # 调度器

app = FastAPI(title="Multi-Agent Scheduler API")
scheduler = LLMScheduler()

# ---- 跨域：若前端与 API 不同域/端口需放开 ----
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # 生产请改具体域名
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- 静态托管：访问 http://<IP>:8080/ 即打开 index.html ----
frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")

# ---------- 端点 1：一次性 JSON ----------
@app.post("/submit_task")
async def submit_task(text: str = Form(...), file: Optional[UploadFile] = File(None)):
    pdf = await file.read() if file else None
    result = scheduler.dispatch(
        context_id="ctx_once",
        task=text,
        pdf_bytes=pdf,
        plain_text=None if pdf else text,
    )
    return JSONResponse(result)


# ---------- 端点 2：Server-Sent Events 流式 ----------
@app.post("/submit_task_stream")
async def submit_task_stream(text: str = Form(...), file: Optional[UploadFile] = File(None)):
    pdf = await file.read() if file else None
    queue: "asyncio.Queue[Dict[str, Any]]" = asyncio.Queue()

    # 回调：调度器每有进度就丢进队列
    def push_progress(payload: Dict[str, Any]):
        asyncio.create_task(queue.put(payload))

    # 在线程里跑调度器
    asyncio.create_task(asyncio.to_thread(
        scheduler.dispatch,
        "ctx_stream",
        text,
        pdf,
        None if pdf else text,
        push_progress
    ))

    async def event_gen():
        while True:
            payload = await queue.get()
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
            if payload.get("status") == "done":
                break

    return StreamingResponse(event_gen(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("coordinator.server:app", host="0.0.0.0", port=8080, reload=True)

