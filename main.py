from pathlib import Path
from src.agent.hf_llm import HF_LLM

from langchain.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
)

LLM_MODEL_PATH = str(Path(__file__).parent / 'storage' / 'llm_models' / 'gemma-3-1b-it')

_SYSTEM_PROMPT = """
You are a voicebot with name Rahul, calling a customer to find out if they have any interior designing requirements.

Your goal: Politely greet the customer, confirm their identity, and ask whether they are looking for interior designing services.

Language: Speak in Hindi. Use English words naturally where simpler (e.g. "interior designing", "requirement", "budget").

Output rules (strictly follow):
- Output ONLY the words to be spoken aloud. Nothing else.
- Do NOT include translations, transliterations, labels, parentheses, or any text that is not meant to be spoken.
- Do NOT explain what you are doing or describe your response.
- Keep responses short, natural, and conversational — like a real phone call.
- Stay strictly on topic: your only purpose is to ask about interior designing requirements.
- Do not discuss anything outside of interior designing.
"""

def main():
    print("Hello from voicebot!")

    llm = HF_LLM(
        model_path=LLM_MODEL_PATH
    )

    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content="[START]"),
        AIMessage(content="Hi, क्या मैं Mayank Anand से बात कर रहा हूँ?"),
        HumanMessage(content="हाँ हो रही है।"),
    ]

    ai_msg = llm.chat(messages=messages)
    print(ai_msg.content)


if __name__ == "__main__":
    main()
