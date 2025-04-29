from pathlib import Path
import asyncio, json
from uuid import uuid4

from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from scheduler.llm_scheduler import LLMScheduler

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"])
sch = LLMScheduler()

# ---------- 静态前端 ----------
BASE_DIR      = Path(__file__).resolve().parent.parent
FRONTEND_DIR  = BASE_DIR / "frontend"
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="frontend")

@app.get("/")
async def index():                    # 关键：根路径直接返回 index.html
    return FileResponse(FRONTEND_DIR / "index.html")

# ---------- SSE ----------
@app.post("/submit_task_stream")
async def submit_task_stream(text: str = Form(""), file: UploadFile | None = None):

    pdf = await file.read() if file else None
    q   = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def push(msg): loop.call_soon_threadsafe(q.put_nowait, msg)

    asyncio.create_task(asyncio.to_thread(
        sch.dispatch, f"ctx_{uuid4().hex}", text,
        pdf_bytes=pdf, progress_cb=push
    ))

    async def event_gen():
        while True:
            msg = await q.get()
            yield f"data: {json.dumps(msg, ensure_ascii=False)}\n\n"
            if msg.get("type") == "chat_file":
                break

    return StreamingResponse(event_gen(), media_type="text/event-stream")


if __name__ == "__main__":            # 直接 python coordinator/server.py 也能跑
    import uvicorn
    uvicorn.run("coordinator.server:app",
                host="0.0.0.0", port=8080, reload=True)

