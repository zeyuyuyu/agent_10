import json, textwrap, requests, fitz, uuid                    # ← uuid 用来生成 subtask_id
from typing import List, Dict, Tuple, Any, Callable, Optional
from openai import OpenAI
from mcp.memory import MemoryStore
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

REGISTRY: Dict[str, Dict[str, str]] = {
    "llama2_agent":   {"url": "http://136.59.129.136:34517/infer",
                       "desc": "长篇文本解析、逻辑推理"},
    "llama2_agent_2": {"url": "http://142.214.185.187:30934/infer",
                       "desc": "OCR、表格/图像抽取"},
}

def _page_ranges(ids: List[str]) -> List[Tuple[str, str]]:
    nums = sorted(int(i.split("_")[1]) for i in ids)
    out, start = [], nums[0]; prev = start
    for n in nums[1:]:
        if n == prev + 1: prev = n
        else: out.append((start, prev)); start = prev = n
    out.append((start, prev))
    return [(f"page_{a}", f"page_{b}") for a, b in out]

class LLMScheduler:
    def __init__(self): self.mem = MemoryStore()
    @staticmethod
    def _push(cb,p): cb and cb(p)
    @staticmethod
    def _pdf_pages(data): return [(f"page_{i+1}",p.get_text("text"))
                                  for i,p in enumerate(fitz.open(stream=data,filetype="pdf"))]

    def _call_agent(self, ag: str, payload: Dict[str,Any]):
        """MCP：始终发送 JSON，其中至少包含 subtask_id 与 prompt"""
        try:
            r = requests.post(REGISTRY[ag]["url"], json=payload, timeout=180)
            res = r.json() if r.status_code == 200 else {}
            return ("succeed", res.get("result", "")) if "result" in res \
                   else ("failed", str(res)[:120])
        except Exception as e:
            return "failed", f"[❌]{e}"

    # ---------- GPT 规划 ----------（保持原样） ---------------------------
    def _plan_pdf(self,task,pages):
        summary_lines=[]
        for pid,txt in pages:
            clean=txt.replace("\n"," ")
            summary_lines.append(f"{pid}: {textwrap.shorten(clean,120)}")
        system="根据 agent 能力分配页面，返回 JSON {agent:[page_id,…]}。\n"+ \
               "\n".join(f"{k}: {v['desc']}" for k,v in REGISTRY.items())
        rsp=client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role":"system","content":system},
                      {"role":"user","content":"任务:"+task+"\n页面:\n"+"\n".join(summary_lines)}],
            response_format={"type":"json_object"})
        try: data=json.loads(rsp.choices[0].message.content)
        except: data={}
        return {k:v for k,v in data.items() if k in REGISTRY and isinstance(v,list)}

    def _need_split(self,task):
        rsp=client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role":"system","content":"仅回答 yes/no"},
                      {"role":"user","content":task}],max_tokens=1)
        return rsp.choices[0].message.content.strip().lower().startswith("y")

    def _plan_text(self,task):
        sys="拆分任务并分配 agent，返回 JSON {agent:[子任务,…]}。\n"+ \
            "\n".join(f"{k}: {v['desc']}" for k,v in REGISTRY.items())
        rsp=client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role":"system","content":sys},
                      {"role":"user","content":task}],
            response_format={"type":"json_object"})
        try: data=json.loads(rsp.choices[0].message.content)
        except: data={}
        return {k:v for k,v in data.items() if k in REGISTRY and isinstance(v,list)}

    # ---------- dispatch ---------- ---------------------------------------
    def dispatch(self,ctx,task,*,pdf_bytes=None,
                 progress_cb:Callable[[Dict[str,Any]],None]|None=None):

        self._push(progress_cb,{"type":"chat_text",
                                "data":{"message":"已接收任务，开始规划…"}})

        subtasks=[]; page_dict={}

        # ---- PDF ----
        if pdf_bytes:
            pages=self._pdf_pages(pdf_bytes); page_dict=dict(pages)
            plan=self._plan_pdf(task,pages) or {}
            if not plan:
                half=len(pages)//2 or 1
                plan={"llama2_agent":[p for p,_ in pages[:half]],
                      "llama2_agent_2":[p for p,_ in pages[half:]]}
            idx=1
            for ag,ids in plan.items():
                for p1,p2 in _page_ranges(ids):
                    subtasks.append({
                        "index": idx,
                        "subtask_id": uuid.uuid4().hex,          # ← 唯一 subtask_id
                        "description": "Process "+(p1 if p1==p2 else f"{p1}~{p2}"),
                        "agent": ag,
                        "pages": (p1,p2)
                    })
                    idx+=1
        # ---- TEXT ----
        else:
            if not self._need_split(task):
                subtasks=[{
                    "index": 1,
                    "subtask_id": uuid.uuid4().hex,
                    "description": task,
                    "agent": next(iter(REGISTRY))
                }]
            else:
                plan=self._plan_text(task); idx=1
                for ag,lst in plan.items():
                    for s in lst:
                        subtasks.append({
                            "index": idx,
                            "subtask_id": uuid.uuid4().hex,
                            "description": s,
                            "agent": ag
                        }); idx+=1

        self._push(progress_cb,{"type":"subtask_list",
                                "data":{"list":[{"index":s["index"],
                                                 "description":s["description"]}
                                                 for s in subtasks]}})

        # ---- execute subtasks ----
        for st in subtasks:
            self._push(progress_cb,{"type":"subtask_start","data":st})
            self._push(progress_cb,{"type":"action_start",
                                    "data":{"subtask_index":st["index"],
                                            "index":1,
                                            "agent_name":st["agent"],
                                            "description":st["description"]}})

            if pdf_bytes:
                p1,p2=st["pages"]
                s=int(p1.split("_")[1]); e=int(p2.split("_")[1])
                prompt="【任务】"+task+"\n\n" + "\n\n".join(
                    page_dict[f"page_{i}"] for i in range(s,e+1))
            else:
                prompt="【任务】"+task+"\n\n【子任务】"+st["description"]

            # 发送符合 MCP 的载荷：带 subtask_id
            payload = {"subtask_id": st["subtask_id"], "prompt": prompt}

            status,result=self._call_agent(st["agent"], payload)

            self._push(progress_cb,{"type":"action_end",
                                    "data":{"subtask_index":st["index"],
                                            "index":1,
                                            "agent_name":st["agent"],
                                            "status":status,
                                            "result_format":"markdown",
                                            "result":result}})
            self._push(progress_cb,{"type":"subtask_end",
                                    "data":{"index":st["index"]}})

        self._push(progress_cb,{"type":"chat_file",
                                "data":{"format":"markdown",
                                        "file_data":"# 任务全部完成 ✅"}})

