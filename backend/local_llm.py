"""
DeepSeek API LLM 调用模块
通过 OpenAI 兼容接口调用 DeepSeek API
"""

import os
from openai import OpenAI


class LocalLLM:
    """DeepSeek API LLM 封装"""

    def __init__(
        self,
        model: str = "deepseek-chat",
        api_key: str | None = None,
        base_url: str = "https://api.deepseek.com",
        temperature: float = 0.0,
        max_tokens: int = 8192,
        reasoning_effort: str | None = None,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort

        key = api_key or os.getenv("DEEPSEEK_API_KEY", "")
        if not key:
            raise ValueError(
                "DeepSeek API Key 未设置。请通过参数传入或设置环境变量 DEEPSEEK_API_KEY"
            )

        self.client = OpenAI(api_key=key, base_url=base_url)

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        """
        调用 DeepSeek API 生成回答

        当开启 Thinking 模式时，若 content 为空（推理消耗全部 token），
        自动重试：先尝试不带 thinking 的纯回答模式，确保至少有一次有效输出。

        Args:
            prompt: 用户提示词
            system_prompt: 系统提示词（可选）

        Returns:
            模型生成的文本回答
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # ----- 主调用（可能带 thinking） -----
        try:
            kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "stream": False,
            }
            if self.reasoning_effort:
                kwargs["reasoning_effort"] = self.reasoning_effort
                kwargs["extra_body"] = {"thinking": {"type": "enabled"}}

            response = self.client.chat.completions.create(**kwargs)
            msg = response.choices[0].message
            content = msg.content

            # content 有效，直接返回
            if content and content.strip():
                return content

            # ----- content 为空：可能是 Thinking 消耗了全部 token -----
            # 尝试从 reasoning_content 读取（某些 SDK 版本有此字段）
            reasoning = getattr(msg, "reasoning_content", None)
            if reasoning and reasoning.strip():
                return reasoning

            # 如果开启了 thinking 但 content 为空，重试不带 thinking
            if self.reasoning_effort:
                return self._generate_simple(messages)

        except Exception as e:
            # API 调用异常，重试不带 thinking
            if self.reasoning_effort:
                try:
                    return self._generate_simple(messages)
                except Exception:
                    return f"[LLM 调用失败] {e}"
            return f"[LLM 调用失败] {e}"

        # 理论上不会到这里（所有路径都有 return），放一个兜底
        return ""

    def _generate_simple(self, messages):
        """不带 Thinking 的纯回答模式（fallback）"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=False,
        )
        content = response.choices[0].message.content
        return content if content and content.strip() else ""


    def check_health(self) -> bool:
        """检查 DeepSeek API 连通性"""
        try:
            self.generate("ping", system_prompt="回复 pong")
            return True
        except Exception:
            return False
