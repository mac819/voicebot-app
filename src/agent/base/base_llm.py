from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Iterator

LLMInputT = TypeVar("LLMInputT")
ChatT = TypeVar("ChatT")
ChunkT = TypeVar("ChunkT")


class BaseLLM(ABC, Generic[LLMInputT, ChatT, ChunkT]):

    @abstractmethod
    def load_model(self, *args, **kwargs) -> None: ...

    @abstractmethod
    def chat(self, message: LLMInputT, *args, **kwargs) -> ChatT: ...

    @abstractmethod
    def stream(self, message: LLMInputT, *args, **kwargs) -> Iterator[ChunkT]: ...


        