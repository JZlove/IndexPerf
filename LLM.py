import logging
import torch
from typing import List, Optional, Mapping, Any
from langchain.llms.base import LLM
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, AutoModel

logger = logging.getLogger(__name__)


max_input_size = 2048
num_output = 512
max_chunk_overlap = 20


class VicunaLLM(LLM):
    model_name = "/modules/vicuna-7b"
    pipeline: object = None

    def __init__(self):
        super().__init__()

    def load_model(self):
        logger.info("To load tokenizer.")
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        logger.info("Load tokenizer done.")

        logger.info("To load model.")
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name, low_cpu_mem_usage=True, device_map="auto", torch_dtype=torch.float16)
        logger.info("Load model done.")

        self.pipeline = pipeline("text-generation",
                                 model=model, tokenizer=tokenizer)

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        prompt_length = len(prompt)
        logger.info("To predict.")
        response = self.pipeline(prompt, max_new_tokens=num_output)[
            0]["generated_text"]
        logger.info("Predict done.")
        return response[prompt_length:]

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"name_of_model": self.model_name}

    @property
    def _llm_type(self) -> str:
        return "custom"


class ChatGLM(LLM):
    model_name = "/modules/chatglm-6b"
    tokenizer: object = None
    model: object = None

    def __init__(self):
        super().__init__()

    def load_model(self):
        logger.info("To load tokenizer.")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True)
        logger.info("Load tokenizer done.")

        logger.info("To load model.")
        self.model = AutoModel.from_pretrained(
            self.model_name, trust_remote_code=True, device_map="auto").half()
        logger.info("Load model done.")

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        logger.info(f"To predict.")
        rsp, history = self.model.chat(self.tokenizer, prompt)
        logger.info(f"Predict done.")
        return rsp

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"name_of_model": self.model_name}

    @property
    def _llm_type(self) -> str:
        return "custom"
