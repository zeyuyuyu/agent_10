from fastapi import FastAPI, File, Form, UploadFile
from typing import Optional
from scheduler.llm_scheduler import LLMScheduler

app = FastAPI()
scheduler = LLMScheduler()

@app.post("/submit_task")
async def submit_task(
    text: str = Form(...),              # 用户任务全文
    file: Optional[UploadFile] = File(None)  # 可选 PDF
):
    pdf_bytes = await file.read() if file else None

    # 调度器签名：dispatch(context_id, task, file_bytes=None)
    result, trace = scheduler.dispatch(
        context_id="ctx001",
        task=text,          # ← 改这里
        file_bytes=pdf_bytes
    )

    return {"memory": result, "trace": trace}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)

