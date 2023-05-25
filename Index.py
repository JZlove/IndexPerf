import logging
from typing import List, Sequence, Optional
from llama_index import GPTTreeIndex, Document, ServiceContext, PromptHelper, LLMPredictor
from llama_index.data_structs.node import Node
from llama_index.node_parser.node_utils import get_nodes_from_document
from llama_index.indices.base import BaseGPTIndex
from langchain.text_splitter import CharacterTextSplitter
from LLM import *

logger = logging.getLogger(__name__)


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
    # vicunallm = VicunaLLM()
    # vicunallm.load_model()
    # llm_predictor = LLMPredictor(llm=vicunallm)
    chatglm = ChatGLM()
    chatglm.load_model()
    llm_predictor = LLMPredictor(llm=chatglm)

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
