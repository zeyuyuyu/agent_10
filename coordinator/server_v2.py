from fastapi import FastAPI, File, Form, UploadFile
from typing import Optional, List
from pydantic import BaseModel

# 引用调度器，路径从 coordinator 导入 scheduler 下的 llm_scheduler
from scheduler.llm_scheduler import LLMScheduler  

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
    如果上传了文件 => 调度器自动拆页
    否则由调度器判断任务复杂度，决定要不要拆分子任务
    """
    pdf_bytes = await file.read() if file else None

    result = scheduler.dispatch(
        context_id="ctx001",
        task=text,
        pdf_bytes=pdf_bytes,
        plain_text=None if pdf_bytes else text,
        progress_cb=None
    )
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080, reload=True)

