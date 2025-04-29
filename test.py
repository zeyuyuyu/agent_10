# test_scheduler.py
from llm_scheduler import LLMScheduler

s = LLMScheduler()
result = s.dispatch(
    context_id="demo",
    task="用一句话介绍 Python，并给三个关键特性",
    plain_text="用一句话介绍 Python，并给三个关键特性"
)
print(result)

