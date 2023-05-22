import logging
from typing import List
from llama_index import Document

logger = logging.getLogger(__name__)


def load_data(file_list: List[str]) -> List[Document]:
    ret_document = []
    for file in file_list:
        with open(file, "r", encoding="utf-8") as f:
            content = f.read()
            ret_document.append(Document(content))
    return ret_document
