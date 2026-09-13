"""第 1.3 节智能旅行助手的程序入口。"""

import os
import re
from typing import Callable

from openai import OpenAI
from dotenv import load_dotenv

from get_attraction import get_attraction
from get_weather import get_weather
from system_prompt import AGENT_SYSTEM_PROMPT


class OpenAICompatibleClient:
    """调用任何兼容 OpenAI Chat Completions 接口的模型服务。"""

    def __init__(
        self, model: str, api_key: str, base_url: str, timeout: float = 60
    ) -> None:
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

    def generate(self, prompt: str, system_prompt: str) -> str:
        print("正在调用大语言模型...")
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            stream=False,
        )
        answer = response.choices[0].message.content
        if not answer:
            raise RuntimeError("模型返回了空响应。")
        return answer


AVAILABLE_TOOLS: dict[str, Callable[..., str]] = {
    "get_weather": get_weather,
    "get_attraction": get_attraction,
}


def load_llm() -> OpenAICompatibleClient:
    """从环境变量读取模型配置，并在缺失时给出可操作的提示。"""
    load_dotenv()
    required = ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL_ID")
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        raise RuntimeError(
            "缺少配置：" + ", ".join(missing) + "。请复制 .env.example 为 .env 并填写真实值，"
            "再运行程序。"
        )
    return OpenAICompatibleClient(
        model=os.environ["LLM_MODEL_ID"],
        api_key=os.environ["LLM_API_KEY"],
        base_url=os.environ["LLM_BASE_URL"],
        timeout=float(os.getenv("LLM_TIMEOUT", "60")),
    )


def parse_action(output: str) -> str:
    """从模型的一对 Thought-Action 输出中提取 Action。"""
    match = re.search(r"^Action:\s*(.+)$", output, re.MULTILINE)
    if not match:
        raise ValueError("未能解析到 Action 字段。")
    return match.group(1).strip()


def run_agent(user_prompt: str, max_steps: int = 5) -> str:
    """运行 Thought-Action-Observation 循环，返回最终回答。"""
    llm = load_llm()
    prompt_history = [f"用户请求：{user_prompt}"]

    for step in range(1, max_steps + 1):
        print(f"\n--- 循环 {step} ---")
        llm_output = llm.generate("\n".join(prompt_history), AGENT_SYSTEM_PROMPT).strip()
        print(f"模型输出：\n{llm_output}")
        prompt_history.append(llm_output)

        try:
            action = parse_action(llm_output)
            finish = re.fullmatch(r"Finish\[(.*)\]", action, re.DOTALL)
            if finish:
                return finish.group(1).strip()

            tool_call = re.fullmatch(r"(\w+)\((.*)\)", action, re.DOTALL)
            if not tool_call:
                raise ValueError("Action 不是合法的工具调用或 Finish[最终答案]。")
            tool_name, args_text = tool_call.groups()
            if tool_name not in AVAILABLE_TOOLS:
                raise ValueError(f"未定义的工具：{tool_name}")

            kwargs = dict(re.findall(r'(\w+)="([^"]*)"', args_text))
            observation = AVAILABLE_TOOLS[tool_name](**kwargs)
        except (TypeError, ValueError) as exc:
            observation = f"错误：{exc} 请严格按规定格式重试。"

        observation_message = f"Observation: {observation}"
        print(observation_message)
        prompt_history.append(observation_message)

    raise RuntimeError(f"智能体在 {max_steps} 次循环内没有完成任务。")


if __name__ == "__main__":
    print("智能旅行助手已启动。输入 exit 或 quit 可退出。")
    while True:
        question = input("\n请输入你的问题：").strip()
        if question.lower() in {"exit", "quit"}:
            print("已退出智能旅行助手。")
            break
        if not question:
            print("问题不能为空，请重新输入。")
            continue
        try:
            result = run_agent(question)
            print(f"\n任务完成，最终答案：{result}")
        except Exception as exc:
            print(f"\n本次任务失败：{exc}")
