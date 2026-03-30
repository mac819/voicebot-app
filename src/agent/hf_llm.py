from typing import Iterator
from src.agent.base.base_llm import BaseLLM
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    BaseMessage,
    BaseMessageChunk
)
from transformers import BitsAndBytesConfig


class HF_LLM(BaseLLM[list[HumanMessage | SystemMessage], BaseMessage, BaseMessageChunk]):

    def __init__(self, model_path: str):
        super().__init__()
        self.model_path = model_path

        self.load_model()


    def load_model(self, ):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        self.llm = HuggingFacePipeline.from_model_id(
            model_id=self.model_path,
            task="text-generation",
            model_kwargs=dict(
                quantization_config=quantization_config,
            ),
            pipeline_kwargs=dict(
                max_new_tokens=512,
                do_sample=False,
                repetition_penalty=1.03,
                return_full_text=False,
            )
        )

        self.chat_model = ChatHuggingFace(llm=self.llm)


    def chat(self, messages: list[HumanMessage | SystemMessage], **kwargs) -> BaseMessage: 
        return self.chat_model.invoke(messages, **kwargs)


    def stream(self, messages: list[HumanMessage | SystemMessage], **kwargs) -> Iterator[BaseMessageChunk]: 
        return self.chat_model.stream(messages, **kwargs)