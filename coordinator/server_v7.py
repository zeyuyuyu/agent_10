from fastapi import FastAPI, Form, File, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import asyncio, json
from typing import Optional, Dict, Any

from scheduler.llm_scheduler import LLMScheduler

app = FastAPI(title="Multi-Agent Scheduler API")
scheduler = LLMScheduler()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/submit_task")
async def submit_task(text: str = Form(...), file: Optional[UploadFile] = File(None)):
    pdf = await file.read() if file else None
    result = scheduler.dispatch("ctx_once", text, pdf, None, None if pdf else text)
    return JSONResponse(result)


@app.post("/submit_task_stream")
async def submit_task_stream(text: str = Form(...), file: Optional[UploadFile] = File(None)):
    pdf   = await file.read() if file else None
    queue: "asyncio.Queue[Dict[str, Any]]" = asyncio.Queue()
    loop  = asyncio.get_running_loop()               # ★ 拿到主事件循环

    # ---------- 修正的回调 ----------
    def push_progress(payload: Dict[str, Any]):
        # 把 payload 丢回主循环中的 queue
        loop.call_soon_threadsafe(queue.put_nowait, payload)

    # 在线程池跑调度器
    asyncio.create_task(asyncio.to_thread(
        scheduler.dispatch,
        "ctx_stream",          # context_id
        text,                  # task
        pdf,                   # pdf_bytes
        None,                  # file_bytes
        None if pdf else text, # plain_text
        push_progress          # progress_cb
    ))

    async def event_gen():
        while True:
            payload = await queue.get()
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
            if payload.get("status") == "done":
                break

    return StreamingResponse(event_gen(), media_type="text/event-stream")


# ---------- 静态文件 ----------
frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
app.mount("/static", StaticFiles(directory=frontend_dir, html=True), name="frontend")

@app.get("/")
async def index():
    return FileResponse(frontend_dir / "index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("coordinator.server:app", host="0.0.0.0", port=8080, reload=True)

