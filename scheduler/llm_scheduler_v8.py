import os, requests
from typing import List, Dict, Tuple, Optional, Callable, Any

import fitz  # PyMuPDF
from openai import OpenAI
from mcp.memory import MemoryStore

try:
    from pdf2image import convert_from_bytes
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# ---------- OpenAI ----------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------- Agent Registry ----------
REGISTRY: Dict[str, str] = {
    "llama2_agent":   "http://136.59.129.136:34517/infer",
    "llama2_agent_2": "http://142.214.185.187:30934/infer",
}

# ------------------------------------------------------------


class LLMScheduler:
    """
    调度流程：
    1. 判断任务是否需要拆分（PDF 必拆；纯文本由 GPT-4o 判断）。
    2. 无需拆分 → GPT-4o 直接生成 Markdown。
    3. 需要拆分 → 推送子任务列表；分配子任务给各 agent；逐步推送进度。
    4. Agent 全部完成后，再汇总为最终 Markdown。
    """

    def __init__(self):
        self.memory = MemoryStore()
        self.trace: List[Dict[str, Any]] = []

    # ---------- PDF → pages ----------
    @staticmethod
    def _pdf_pages(data: bytes) -> List[Tuple[str, str]]:
        pages: List[Tuple[str, str]] = []
        doc = fitz.open(stream=data, filetype="pdf")
        for idx, page in enumerate(doc, 1):
            txt = page.get_text("text") or ""
            if not txt.strip() and OCR_AVAILABLE:
                img = convert_from_bytes(
                    data, dpi=300, first_page=idx, last_page=idx
                )[0]
                txt = pytesseract.image_to_string(img, lang="eng+chi_sim")
            pages.append((f"page_{idx}", txt.strip()))
        return pages

    # ---------- 判断是否需要拆分 ----------
    def _need_split(self, task: str) -> bool:
        rsp = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是任务复杂度评估器，仅回答 'yes' 或 'no'。"
                        "当且仅当【任务】需要被拆分为子任务才能高效完成时回答 'yes'，"
                        "否则回答 'no'。"
                    ),
                },
                {"role": "user", "content": f"【任务】{task}"},
            ],
            max_tokens=1,
        )
        answer = rsp.choices[0].message.content.strip().lower()
        return answer.startswith("y")

    # ---------- GPT-4o 页面分配 ----------
    def _plan_pages(self, task: str, page_ids: List[str]) -> Dict[str, List[str]]:
        ids_str = ", ".join(page_ids)
        sys_msg = (
            "你是任务调度专家，根据【任务】将页面分配给下列 agent："
            + ", ".join(REGISTRY.keys())
            + "。格式：agent: id1,id2 或 agent: m-n。仅返回分配结果。"
        )
        rsp = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": f"【任务】{task}\n【页面列表】{ids_str}"},
            ],
        )
        return self._parse_page_plan(rsp.choices[0].message.content)

    # ---------- GPT-4o 纯文本任务拆解 ----------
    def _plan_subtasks(self, task: str) -> Dict[str, List[str]]:
        sys_msg = (
            "你是任务调度专家，请把【任务】拆分为多个可并行子任务，并分配给 agent："
            + ", ".join(REGISTRY.keys())
            + "。输出每行：agent: 子任务描述。仅返回结果。"
        )
        rsp = client.chat.completions.create(
            model="gpt-4o",               # ← ★ 修正字符串闭合
            messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": task},
            ],
        )
        mapping: Dict[str, List[str]] = {k: [] for k in REGISTRY}
        for line in rsp.choices[0].message.content.splitlines():
            if ":" not in line:
                continue
            ag, desc = line.split(":", 1)
            ag = ag.strip()
            if ag in REGISTRY:
                mapping[ag].append(desc.strip())
        return {k: v for k, v in mapping.items() if v}

    # ---------- 解析页面分配 ----------
    @staticmethod
    def _parse_page_plan(text: str) -> Dict[str, List[str]]:
        plan = {k: [] for k in REGISTRY}
        for ln in text.splitlines():
            if ":" not in ln:
                continue
            ag, ids = ln.split(":", 1)
            ag = ag.strip()
            if ag not in REGISTRY:
                continue
            ids = ids.replace(" ", "")
            for part in ids.split(","):
                if not part:
                    continue
                if "-" in part:
                    s, e = part.split("-")
                    s_idx = int("".join(filter(str.isdigit, s)))
                    e_idx = int("".join(filter(str.isdigit, e)))
                    prefix = "".join(filter(str.isalpha, s)) or "page_"
                    plan[ag] += [f"{prefix}{i}" for i in range(s_idx, e_idx + 1)]
                else:
                    plan[ag].append(part)
        return {k: v for k, v in plan.items() if v}

    # ---------- push helper ----------
    @staticmethod
    def _push(cb: Optional[Callable[[Dict[str, Any]], None]], payload: Dict[str, Any]):
        if cb:
            cb(payload)

    # ---------- dispatch ----------
    def dispatch(
        self,
        context_id: str,
        task: str,
        pdf_bytes: Optional[bytes] = None,
        file_bytes: Optional[bytes] = None,
        plain_text: Optional[str] = None,
        progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:

        pdf_data = pdf_bytes or file_bytes

        # --------  PDF 任务  --------
        if pdf_data:
            pages = self._pdf_pages(pdf_data)
            page_ids = [pid for pid, _ in pages]
            show_ids = [f"Process {pid}" for pid in page_ids]
            self._push(progress_cb, {"status": "subtasks", "subtasks": show_ids})

            plan = self._plan_pages(task, page_ids)
            if not plan:
                plan = {ag: [] for ag in REGISTRY}
                for idx, pid in enumerate(page_ids):
                    ag = list(REGISTRY.keys())[idx % len(REGISTRY)]
                    plan[ag].append(pid)

            page_map = dict(pages)
            make_prompt = lambda pid: f"【任务指令】{task}\n\n【{pid} 原文】\n{page_map.get(pid, '')}"
            sub_ids_source = plan

        # -------- 纯文本 / 指令 ----------- 
        else:
            text = plain_text if plain_text is not None else task
            if not self._need_split(task):
                rsp = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": f"请以 Markdown 格式完整回答：\n\n{text.strip()}"}],
                )
                md = rsp.choices[0].message.content.strip()
                self._push(progress_cb, {"status": "done", "markdown": md})
                return {"markdown": md, "trace": self.trace}

            # 需要拆分
            plan = self._plan_subtasks(task)
            all_subs = [sub for lst in plan.values() for sub in lst]
            show_subs = [f"Process {s}" for s in all_subs]
            self._push(progress_cb, {"status": "subtasks", "subtasks": show_subs})
            make_prompt = lambda sub: f"【总任务】{task}\n\n【子任务描述】{sub}"
            sub_ids_source = plan

        # ---------- 调用各 agent ----------
        for ag, keys in sub_ids_source.items():
            for key in keys:
                sub_name = f"Process {key}"
                prompt_text = make_prompt(key)
                self._push(progress_cb, {"status": "assign", "agent": ag, "subtask": sub_name, "output": "Processing"})
                payload = {
                    "prompt": prompt_text,
                    "context_id": context_id,
                    "sub_id": key,
                    "agent_name": ag,
                    "shared_memory": self.memory.get(context_id),
                }
                try:
                    resp = requests.post(REGISTRY[ag], json=payload, timeout=120)
                    res_text = resp.json().get("result", "")
                except Exception as e:
                    res_text = f"[❌ 调用失败] {e}"

                ag_mem = self.memory.get(context_id).get(ag, {}) or {}
                ag_mem[sub_name] = res_text
                self.memory.update(context_id, {ag: ag_mem})
                self.trace.append({"agent": ag, "subtask": sub_name, "output": res_text})

        # ---------- 汇总 ----------
        collected = "\n\n".join(
            f"【{ag}】\n" + "\n".join(mem.values())
            for ag, mem in self.memory.get(context_id).items()
            if ag in REGISTRY
        )
        rsp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": "请综合以下各智能体输出，总结共识、差异并用 Markdown 返回最终建议：\n" + collected
            }]
        )
        summary_md = rsp.choices[0].message.content.strip()
        self.memory.update(context_id, {"summary": summary_md})
        self.trace.append({"agent": "scheduler", "subtask": "自动总结", "output": summary_md})

        self._push(progress_cb, {"status": "done", "markdown": summary_md})
        return {"markdown": summary_md, "trace": self.trace}

