import logging
import torch
from typing import List, Sequence, Optional, Mapping, Any
from llama_index import GPTTreeIndex, Document, ServiceContext, PromptHelper, LLMPredictor
from llama_index.data_structs.node import Node
from llama_index.node_parser.node_utils import get_nodes_from_document
from llama_index.indices.base import BaseGPTIndex
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms.base import LLM
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

logger = logging.getLogger(__name__)

max_input_size = 4096
num_output = 1024
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
        response = self.pipeline(prompt, max_new_tokens=num_output)[
            0]["generated_text"]
        return response[prompt_length:]

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"name_of_model": self.model_name}

    @property
    def _llm_type(self) -> str:
        return "custom"


def create_nodes_from_documents(documents: Sequence[Document]):
    all_nodes: List[Node] = []
    splitter = CharacterTextSplitter(
        "\n\n", chunk_size=1800, chunk_overlap=0)

    for doc in documents:
        nodes = get_nodes_from_document(doc, text_splitter=splitter)
        all_nodes.extend(nodes)

    return all_nodes


def create_index_from_nodes(nodes: List[Node]):
    prompt_helper = PromptHelper(max_input_size=max_input_size,
                                 num_output=num_output, max_chunk_overlap=max_chunk_overlap)
    vicunallm = VicunaLLM()
    vicunallm.load_model()
    llm_predictor = LLMPredictor(llm=vicunallm)
    server_context = ServiceContext.from_defaults(
        llm_predictor=llm_predictor, prompt_helper=prompt_helper)

    logger.info("To build tree index.")
    index = GPTTreeIndex(nodes=nodes, num_children=3,
                         service_context=server_context)
    logger.info("Build tree index done.")

    return index


def query_from_index(query: str, index: Optional[BaseGPTIndex]):
    query_engine = index.as_query_engine()
    resp = query_engine.query(query)

    return resp
