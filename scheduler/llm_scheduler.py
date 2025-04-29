

import json, textwrap, requests, uuid, fitz, os, time
from typing import List, Dict, Tuple, Any, Callable
from openai import OpenAI
from mcp.memory import MemoryStore

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

REGISTRY: Dict[str, Dict[str, str]] = {
    "llama2_agent": {
        "url": "http://136.59.129.136:34517/infer",
        "desc": "长篇文本解析、逻辑推理",
    },
    "llama2_agent_2": {
        "url": "http://142.214.185.187:30934/infer",
        "desc": "OCR、表格/图像抽取",
    },
    "web_build_agent": {
        "url": "http://54.179.24.46:5000/infer",
        "desc": "构建网页并返回 URL",
    },
    "web_search_agent": {
        "url": "http://54.179.24.46:5001/infer",
        "desc": "联网搜索并返回摘要",
    },
}


# ---------- Utils -----------------------------------------------------
def _page_ranges(ids: List[str]) -> List[Tuple[str, str]]:
    nums = sorted(int(i.split("_")[1]) for i in ids)
    out, start, prev = [], nums[0], nums[0]
    for n in nums[1:]:
        if n == prev + 1:
            prev = n
        else:
            out.append((start, prev))
            start = prev = n
    out.append((start, prev))
    return [(f"page_{a}", f"page_{b}") for a, b in out]


# =====================================================================
class LLMScheduler:
    # -----------------------------------------------------------------
    def __init__(self):
        self.mem = MemoryStore()

    @staticmethod
    def _push(cb, payload):
        cb and cb(payload)

    @staticmethod
    def _pdf_pages(data):
        return [
            (f"page_{i+1}", p.get_text("text"))
            for i, p in enumerate(fitz.open(stream=data, filetype="pdf"))
        ]

    # ---------- 公共轮询 ----------
    def _poll_status(self, base_url: str, task_id: str):
        status_url = f"{base_url}/status?task_id={task_id}"
        for _ in range(40):  # ≈2 min
            time.sleep(3)
            r = requests.get(status_url, timeout=10)
            if r.status_code != 200:
                continue
            js = r.json()
            st = js.get("status")
            if st == "success":
                return "succeed", js.get("result", "")
            if st in ("error", "interrupted"):
                return "failed", js.get("error", "unknown error")
        return "failed", "Timeout: task not ready after 120 s"

    # ---------- 调用 Agent ----------
    def _call_agent(self, ag: str, payload: Dict[str, Any]):
        if ag in ("web_build_agent", "web_search_agent"):
            try:
                task_id = payload.get("subtask_id") or uuid.uuid4().hex
                base = REGISTRY[ag]["url"].rsplit("/", 1)[0]
                requests.post(
                    REGISTRY[ag]["url"],
                    json={"task_id": task_id, "prompt": payload["prompt"]},
                    timeout=15,
                )
                return self._poll_status(base, task_id)
            except Exception as e:
                return "failed", f"[❌]{e}"

        try:
            r = requests.post(REGISTRY[ag]["url"], json=payload, timeout=180)
            res = r.json() if r.status_code == 200 else {}
            return ("succeed", res.get("result", "")) if "result" in res else (
                "failed",
                str(res)[:120],
            )
        except Exception as e:
            return "failed", f"[❌]{e}"

    # ---------- GPT 辅助 ----------
    def _plan_pdf(self, task, pages):
        summary_lines = [
            f"{pid}: {textwrap.shorten(txt.replace(chr(10), ' '), 120)}"
            for pid, txt in pages
        ]
        system = "根据 agent 能力分配页面，返回 JSON {agent:[page_id,…]}。\n" + "\n".join(
            f"{k}: {v['desc']}" for k, v in REGISTRY.items()
        )
        rsp = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": "任务:" + task + "\n页面:\n" + "\n".join(summary_lines),
                },
            ],
            response_format={"type": "json_object"},
        )
        try:
            data = json.loads(rsp.choices[0].message.content)
        except Exception:
            data = {}
        return {k: v for k, v in data.items() if k in REGISTRY and isinstance(v, list)}

    def _need_split(self, task):
        rsp = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "仅回答 yes/no"},
                {"role": "user", "content": task},
            ],
            max_tokens=1,
        )
        return rsp.choices[0].message.content.strip().lower().startswith("y")

    def _plan_text(self, task):
        sys = (
            "拆分任务并分配 agent，返回 JSON {agent:[子任务,…]}。\n"
            "如果你认为无需调用任何 agent 而可以直接回答，请返回空对象 {}。\n"
            + "\n".join(f"{k}: {v['desc']}" for k, v in REGISTRY.items())
        )
        rsp = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": sys},
                {"role": "user", "content": task},
            ],
            response_format={"type": "json_object"},
        )
        try:
            data = json.loads(rsp.choices[0].message.content)
        except Exception:
            data = {}
        return {k: v for k, v in data.items() if k in REGISTRY and isinstance(v, list)}

    def _reply_by_gpt(self, task: str, cb):
        rsp = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "你是一名高效助手，直接回答用户问题。"},
                {"role": "user", "content": task},
            ],
        )
        self._push(
            cb,
            {
                "type": "chat_file",
                "data": {"format": "markdown", "file_data": rsp.choices[0].message.content},
            },
        )

    # ---------- dispatch ---------------------------------------------
    def dispatch(
        self,
        ctx,
        task,
        *,
        pdf_bytes=None,
        progress_cb: Callable[[Dict[str, Any]], None] | None = None,
    ):
        self._push(
            progress_cb,
            {"type": "chat_text", "data": {"message": "已接收任务，开始规划…"}},
        )

        text_plan = {}
        if pdf_bytes is None:
            # 1) 先让 GPT 做文本规划
            text_plan = self._plan_text(task)
            # 2) 若 GPT 认为无需任何 agent（返回 {}）→ 直接回复
            if not text_plan:
                self._reply_by_gpt(task, progress_cb)
                return

        subtasks: list[dict] = []
        page_dict: dict[str, str] = {}

        # ---- PDF ----
        if pdf_bytes:
            pages = self._pdf_pages(pdf_bytes)
            page_dict = dict(pages)
            plan = self._plan_pdf(task, pages) or {}
            if not plan:
                half = len(pages) // 2 or 1
                plan = {
                    "llama2_agent": [p for p, _ in pages[:half]],
                    "llama2_agent_2": [p for p, _ in pages[half:]],
                }
            idx = 1
            for ag, ids in plan.items():
                for p1, p2 in _page_ranges(ids):
                    subtasks.append(
                        {
                            "index": idx,
                            "subtask_id": uuid.uuid4().hex,
                            "description": "Process "
                            + (p1 if p1 == p2 else f"{p1}~{p2}"),
                            "agent": ag,
                            "pages": (p1, p2),
                        }
                    )
                    idx += 1

        # ---- 纯文本 ----
        else:
            idx = 1
            for ag, lst in text_plan.items():
                for s in lst:
                    subtasks.append(
                        {
                            "index": idx,
                            "subtask_id": uuid.uuid4().hex,
                            "description": s,
                            "agent": ag,
                        }
                    )
                    idx += 1

        # ---- 推送子任务列表 ----
        self._push(
            progress_cb,
            {
                "type": "subtask_list",
                "data": {
                    "list": [
                        {
                            "index": s["index"],
                            "description": s["description"],
                            "agent_name": s["agent"],
                        }
                        for s in subtasks
                    ]
                },
            },
        )

        # ---- 执行子任务 ----
        results = []
        for st in subtasks:
            self._push(progress_cb, {"type": "subtask_start", "data": st})
            self._push(
                progress_cb,
                {
                    "type": "action_start",
                    "data": {
                        "subtask_index": st["index"],
                        "index": 1,
                        "agent_name": st["agent"],
                        "description": st["description"],
                    },
                },
            )

            if pdf_bytes:
                p1, p2 = st["pages"]
                s, e = int(p1.split("_")[1]), int(p2.split("_")[1])
                prompt = (
                    "【任务】" + task + "\n\n"
                    + "\n\n".join(page_dict[f"page_{i}"] for i in range(s, e + 1))
                )
            else:
                prompt = "【任务】" + task + "\n\n【子任务】" + st["description"]

            status, result = self._call_agent(
                st["agent"], {"subtask_id": st["subtask_id"], "prompt": prompt}
            )

            self._push(
                progress_cb,
                {
                    "type": "action_end",
                    "data": {
                        "subtask_index": st["index"],
                        "index": 1,
                        "agent_name": st["agent"],
                        "status": status,
                        "result_format": "markdown",
                        "result": result,
                    },
                },
            )
            self._push(progress_cb, {"type": "subtask_end", "data": {"index": st["index"]}})
            results.append(
                {
                    "index": st["index"],
                    "agent": st["agent"],
                    "description": st["description"],
                    "status": status,
                    "result": result,
                }
            )

        # ---- GPT 汇总 ----
        summary_prompt = (
            "下面给出若干子任务的输出，请你做一个整体总结，指出主要产出，如有失败说明原因，输出 Markdown。"
        )
        concat_md = "\n\n".join(
            f"### 子任务 {r['index']} ({r['agent']}, {r['status']})\n{r['result']}"
            for r in results
        )
        rsp = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "你是可靠的项目助理，擅长总结信息。"},
                {"role": "user", "content": summary_prompt + "\n\n" + concat_md},
            ],
        )
        self._push(
            progress_cb,
            {
                "type": "chat_file",
                "data": {"format": "markdown", "file_data": rsp.choices[0].message.content},
            },
        )

