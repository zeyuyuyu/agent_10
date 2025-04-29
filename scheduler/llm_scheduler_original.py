import os
import io
import requests
from typing import List, Dict, Tuple, Optional

import fitz  # PyMuPDF
from openai import OpenAI
from mcp.memory import MemoryStore

try:
    from pdf2image import convert_from_bytes
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# ---------- 配置 ----------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

REGISTRY: Dict[str, str] = {
    "llama2_agent":   "http://136.59.129.136:34517/infer",
    "llama2_agent_2": "http://142.214.185.187:30934/infer"
}


class LLMScheduler:
    """
    三种场景：
      1. task + PDF（二进制/文件流）→ 按页拆分 → GPT-4o 分页分配
      2. task + plain_text        → 视为单页 text_1 → 分给首个 agent
      3. 仅 task 指令            → GPT-4o 拆子任务 → 分配给各 agent
    """

    def __init__(self):
        self.memory = MemoryStore()
        self.trace: List[Dict] = []

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

    # ---------- GPT-4o 页面分配 ----------
    def _plan_pages(
        self, task: str, page_ids: List[str]
    ) -> Dict[str, List[str]]:
        ids_str = ", ".join(page_ids)
        sys_msg = (
            "你是任务调度专家，根据【任务】将页面分配给下列 agent："
            + ", ".join(REGISTRY.keys())
            + "。格式：agent: id1,id2 或 agent: m-n。仅返回分配结果。"
        )
        user_msg = f"【任务】{task}\n【页面列表】{ids_str}"
        rsp = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": user_msg},
            ],
        )
        return self._parse_page_plan(rsp.choices[0].message.content)

    # ---------- GPT-4o 子任务分配 ----------
    def _plan_subtasks(self, task: str) -> Dict[str, List[str]]:
        sys_msg = (
            "你是任务调度专家，请把【任务】拆分为多个可并行子任务，并分配给 agent："
            + ", ".join(REGISTRY.keys())
            + "。输出每行：agent: 子任务描述。仅返回结果。"
        )
        rsp = client.chat.completions.create(
            model="gpt-4o",
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

    # ---------- 解析页面 plan ----------
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
                    plan[ag] += [
                        f"{prefix}{i}" for i in range(s_idx, e_idx + 1)
                    ]
                else:
                    plan[ag].append(part)
        return {k: v for k, v in plan.items() if v}

    # ---------- dispatch ----------
    def dispatch(
        self,
        context_id: str,
        task: str,
        pdf_bytes: Optional[bytes] = None,
        file_bytes: Optional[bytes] = None,
        plain_text: Optional[str] = None,
    ):
        # ① 判断输入类型
        pdf_data = pdf_bytes or file_bytes

        if pdf_data:
            # —— PDF 路径 ——
            pages = self._pdf_pages(pdf_data)
            page_map = dict(pages)
            plan = self._plan_pages(task, [pid for pid, _ in pages])

            # GPT 分配失败时，平均轮转给 agent
            if not plan:
                plan = {ag: [] for ag in REGISTRY}
                for idx, (pid, _) in enumerate(pages):
                    ag = list(REGISTRY.keys())[idx % len(REGISTRY)]
                    plan[ag].append(pid)

            make_prompt = lambda pid: (
                f"【任务指令】{task}\n\n"
                f"【{pid} 原文】\n{page_map.get(pid, '')}"
            )
            sub_ids_source = plan

        elif plain_text is not None:
            # —— 单段文本 ——
            pages = [("text_1", plain_text.strip())]
            plan = {list(REGISTRY.keys())[0]: ["text_1"]}
            make_prompt = (
                lambda _:
                f"【任务指令】{task}\n\n【文本内容】\n{plain_text.strip()}"
            )
            sub_ids_source = plan

        else:
            # —— 仅指令，拆子任务 ——
            pages = []
            plan = self._plan_subtasks(task)
            make_prompt = (
                lambda sub:
                f"【总任务】{task}\n\n【子任务描述】{sub}"
            )
            sub_ids_source = plan

        # ② 分发
        for ag, keys in sub_ids_source.items():
            for key in keys:
                prompt_text = make_prompt(key)
                payload = {
                    "prompt": prompt_text,
                    "context_id": context_id,
                    "sub_id": key,
                    "agent_name": ag,
                    "shared_memory": self.memory.get(context_id),
                }
                try:
                    resp = requests.post(
                        REGISTRY[ag], json=payload, timeout=120
                    )
                    res_text = resp.json().get("result", "")
                except Exception as e:
                    self.trace.append(
                        {
                            "agent": ag,
                            "subtask": key,
                            "output": f"[❌ 调用失败] {e}",
                        }
                    )
                    continue

                # —— 累积写入 memory —— #
                ag_mem = self.memory.get(context_id).get(ag, {})
                if not isinstance(ag_mem, dict):
                    ag_mem = {}
                ag_mem[key] = res_text
                self.memory.update(context_id, {ag: ag_mem})
                # —— End —— #

                self.trace.append(
                    {"agent": ag, "subtask": key, "output": res_text}
                )

        # ③ 汇总
        collected = "\n\n".join(
            f"【{ag}】\n" + "\n".join(mem.values())
            for ag, mem in self.memory.get(context_id).items()
            if ag in REGISTRY
        )

        rsp = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": (
                        "请综合以下各智能体输出，总结共识、差异并提出建议：\n"
                        + collected
                    ),
                }
            ],
        )
        summary = rsp.choices[0].message.content.strip()
        self.memory.update(context_id, {"summary": summary})
        self.trace.append(
            {
                "agent": "scheduler",
                "subtask": "自动总结",
                "output": summary,
            }
        )
        return self.memory.get(context_id), self.trace

