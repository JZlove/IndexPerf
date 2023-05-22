import sys
import logging
from FileReader import load_data
from Index import *

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    file_list = ['data/paper_2008_Sichuan_earthquake.txt']
    documents = load_data(file_list)
    nodes = create_nodes_from_documents(documents=documents)
    logger.info("Create nodes from documents done.")
    index = create_index_from_nodes(nodes)
    logger.info("Create index from nodes done.")
    resp = query_from_index(
        query="In what year did the earthquake in Sichuan occur?", index=index)
    print(resp)
    print(resp.get_formatted_sources())
