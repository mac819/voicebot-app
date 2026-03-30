#%%
from pathlib import Path
from huggingface_hub import login
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

from langchain.messages import (
    HumanMessage,
    SystemMessage,
)
env_path = "/home/mayank/Documents/projects/voicebot/.env"


hf_hub_token = None
login(token=hf_hub_token)
# %%
# model_id = "google/gemma-3-1b-it"
model_path = str(Path(__file__).parent.parent.parent / 'storage' / 'llm_models' / 'gemma-3-1b-it')
llm = HuggingFacePipeline.from_model_id(
    # model_id=model_id,
    model_id=model_path,
    task="text-generation",
    pipeline_kwargs=dict(
        max_new_tokens=512,
        do_sample=False,
        repetition_penalty=1.03,
    )
)
# %%
chat_model = ChatHuggingFace(llm=llm)
# %%
messages = [
    SystemMessage(content="You're a helpful assistant"),
    HumanMessage(
        content="What happens when an unstoppable force meets an immovable object?"
    ),
]
ai_msg = chat_model.invoke(messages)
# %%
print(ai_msg.content)
# %%
