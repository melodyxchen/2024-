"""
This script demonstrates parsing and indexing for all categories of documents using the LlamaIndex library with chunk references for recursive retrieving.

Functions:
    preprocess: Preprocesses and indexes financial and insurance text files, saving them as nodes and vector indices with chunk references.
    get_text_nodes: Converts JSON-based FAQ entries into text nodes for indexing.
"""

# Imports
from llama_index.core import VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import SimpleDirectoryReader
from llama_index.core.schema import IndexNode, TextNode
from llama_index.core.node_parser import SentenceSplitter
from typing import List
import json
import pickle
import argparse

# Set embedding model for indexing
model_args = {
    'model_name': 'maidalun1020/bce-embedding-base_v1', 
    'max_length': 512, 
    'embed_batch_size': 64, 
    'device': 'cuda'
}
embed_model = HuggingFaceEmbedding(**model_args)
Settings.embed_model = embed_model

# Set chunking parameters
Settings.chunk_size = 512
Settings.chunk_overlap = 100

# Document Preprocessing and Indexing Functions
def preprocess(docFolder: str, indexStoragePath: str, nodeDictPath: str):
    """
    Creates and saves vector indices with node dictionaries.

    This function processes each document(text file) into chunks based on predefined chunk sizes, and creates sub-chunks
    referring to bigger parent chunks for efficient recursive retrieval.

    Args:
        docFolder (str): Path to the folder containing documents to be processed.
        indexStoragePath (str): Path where the vector index will be saved.
        nodeDictPath (str): Path where the node dictionary will be saved as a pickle file.
    """
    # Load documents from specified folder
    docs = SimpleDirectoryReader(docFolder).load_data(show_progress=True)

    # Parse documents into base nodes
    node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=100)
    base_nodes = node_parser.get_nodes_from_documents(docs)

    # Set unique IDs for each base node
    for idx, node in enumerate(base_nodes):
        node.id_ = f"node-{idx}"

    # Define chunk sizes and create corresponding sub-nodes
    sub_chunk_sizes = [128, 256]
    sub_node_parsers = [SentenceSplitter(chunk_size=c, chunk_overlap=20) for c in sub_chunk_sizes]

    all_nodes = []
    for base_node in base_nodes:
        for nparser in sub_node_parsers:
            sub_nodes = nparser.get_nodes_from_documents([base_node])
            sub_index_nodes = [
                IndexNode.from_text_node(sub_node, base_node.node_id) for sub_node in sub_nodes
            ]
            all_nodes.extend(sub_index_nodes)

        # Add original base node to all_nodes
        original_node = IndexNode.from_text_node(base_node, base_node.node_id)
        all_nodes.append(original_node)

    # Create vector index from processed nodes and save to disk
    vector_index_chunk = VectorStoreIndex(all_nodes, embed_model=embed_model, show_progress=True)
    vector_index_chunk.storage_context.persist(persist_dir=indexStoragePath)

    # Save all_nodes dictionary to a pickle file
    all_nodes_dict = {node.node_id: node for node in all_nodes}
    with open(nodeDictPath, "wb") as f:
        pickle.dump(all_nodes_dict, f)



# FAQ Processing and Indexing
def get_text_nodes(json_list: List[dict]) -> List[TextNode]:
    """
    Converts a list of FAQ entries in JSON format into a list of text nodes for indexing.

    This function takes a JSON list containing FAQ entries with question and answer
    pairs, concatenates them into a single text node, and assigns a unique metadata
    identifier to each.

    Args:
        json_list (List[dict]): A list of JSON objects representing FAQ entries.

    Returns:
        List[TextNode]: A list of text nodes with concatenated FAQ questions and answers.
    """
    text_nodes = []
    for entry in json_list:
        text_node = TextNode(text=entry["text"], metadata={"file_name": entry["faq_id"]})
        text_nodes.append(text_node)
    return text_nodes


if __name__ == "__main__":
    # Set up argument parsing for paths and files
    parser = argparse.ArgumentParser(description='Process and index financial, insurance, and FAQ documents.')
    parser.add_argument('--finance_source_path', type=str, help='Path to the finance reference text files', default='processed_data/all_fin')
    parser.add_argument('--insurance_source_path', type=str, help='Path to the insurance reference text files', default='processed_data/all_in')
    parser.add_argument('--faq_source_path', type=str, help='Path to FAQ JSON file', default='競賽資料集/reference/faq/pid_map_content.json')
    parser.add_argument('--storage_path', type=str, help='Path to save indicies and node dictionaries', default='processed_data')

    args = parser.parse_args()

    # Process financial and insurance documents
    preprocess(
        docFolder = args.finance_source_path, 
        indexStoragePath = args.storage_path + "/Index_all_fin_recursiveChunk", 
        nodeDictPath = args.storage_path + "/all_nodes_dict_fin.pkl"
    )
    preprocess(
        docFolder="processed_data/all_ins", 
        indexStoragePath = args.storage_path + "/Index_all_ins_recursiveChunk", 
        nodeDictPath = args.storage_path + "/all_nodes_dict_ins.pkl"
    )

    # Load FAQ data and process into text nodes
    with open(args.faq_source_path, 'rb') as f:
        faq = json.load(f)

    faq_entries = {"files": []}
    for k in range(len(faq)):
        n = len(faq[f'{k}'])
        text=""
        for i in range(n):
            q = faq[f'{k}'][i]['question']
            a = ''
            # concate Q and A
            for j in range(len(faq[f'{k}'][i]['answers'])):
                a += faq[f'{k}'][i]['answers'][j]
            text += (q + a)
            faq_entries["files"].append({'faq_id':f'{k}', 'text': text})

    # Convert FAQ entries to text nodes and create vector index
    faq_text_nodes = get_text_nodes(faq_entries['files'])
    index_faq = VectorStoreIndex(faq_text_nodes, show_progress=True)
    index_faq.storage_context.persist(persist_dir = args.storage_path + "/Index_all_faq")