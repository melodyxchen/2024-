"""
The script loads stored indices for each category, applies recursive retrieval with chunk references and metadata filtering, 
and outputs the top-ranked retrieved document for each query.
"""

import json
import pickle
from typing import List
from llama_index.core import StorageContext, load_index_from_storage, Settings, QueryBundle
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.vector_stores.types import MetadataFilters, MetadataFilter
from llama_index.core.vector_stores import FilterOperator
from llama_index.core.retrievers import VectorIndexRetriever, RecursiveRetriever
from llama_index.core.postprocessor import SentenceTransformerRerank
import argparse

# Define embedding model settings
model_args = {
    'model_name': 'maidalun1020/bce-embedding-base_v1',
    'max_length': 512,
    'embed_batch_size': 64,
    'device': 'cuda'
}
embed_model = HuggingFaceEmbedding(**model_args)
Settings.embed_model = embed_model


def load_indices(path):
    """
    Loads document indices and corresponding node dictionaries for insurance, finance, and FAQ categories.
    
    Returns:
        tuple: Tuple containing loaded indices and node dictionaries for insurance, finance, and FAQ.
    """
    # Load insurance index and nodes
    storage_context_ins = StorageContext.from_defaults(persist_dir= path + "/Index_all_ins_recursiveChunk")
    index_ins = load_index_from_storage(storage_context_ins)
    with open(path + "/all_nodes_dict_ins.pkl", "rb") as f:
        all_nodes_dict_ins = pickle.load(f)

    # Load finance index and nodes
    storage_context_fin = StorageContext.from_defaults(persist_dir= path + "/Index_all_fin_recursiveChunk")
    index_fin = load_index_from_storage(storage_context_fin)
    with open(path + "/all_nodes_dict_fin.pkl", "rb") as f:
        all_nodes_dict_fin = pickle.load(f)

    # Load FAQ index
    storage_context_faq = StorageContext.from_defaults(persist_dir= path + "/Index_all_faq")
    index_faq = load_index_from_storage(storage_context_faq)

    return (index_ins, all_nodes_dict_ins, index_fin, all_nodes_dict_fin, index_faq)


def process_queries(questions: List[dict], indices, answer_output: str):
    """
    Processes queries by applying recursive retrieval, reranking results, and saving the answer.

    Args:
        questions (List[dict]): List of question dictionaries containing 'query' and 'source' information.
        indices (tuple): Tuple containing document indices and node dictionaries.
        answer_output (str): Path to save the retrieved answers as a JSON file.
    """
    index, all_nodes_dict_ins, index_fin, all_nodes_dict_fin, index_faq = indices
    rerank = SentenceTransformerRerank(model="maidalun1020/bce-reranker-base_v1", top_n=1)
    answer_dict = {"answers": []}

    for i in range(len(questions)):
        source = questions[i]['source']

        if questions[i]['category'] == 'insurance':
            sourceName = ['ins_'+ str(id) + '.txt' for id in source]
            filters = MetadataFilters(filters = [MetadataFilter(key = "file_name", operator = FilterOperator.IN, value = sourceName)])
            retriever = VectorIndexRetriever(index = index, similarity_top_k = 20, filters = filters, node_postprocessors = [rerank])
            retriever_chunk = RecursiveRetriever("vector", retriever_dict = {"vector": retriever}, node_dict = all_nodes_dict_ins)

        elif questions[i]['category'] == 'finance':
            sourceName = ['fin_'+ str(id) + '.txt' for id in source]
            filters = MetadataFilters(filters = [MetadataFilter(key = "file_name", operator = FilterOperator.IN, value = sourceName)])
            retriever = VectorIndexRetriever(index = index_fin, similarity_top_k = 20, filters = filters, node_postprocessors = [rerank])
            retriever_chunk = RecursiveRetriever("vector", retriever_dict = {"vector": retriever}, node_dict = all_nodes_dict_fin)

        else:  # FAQ
            sourceName = [str(id) for id in source]
            filters = MetadataFilters(filters=[MetadataFilter(key = "file_name", operator = FilterOperator.IN, value = sourceName)])
            retriever = VectorIndexRetriever(index = index_faq, similarity_top_k = 20, filters = filters, node_postprocessors = [rerank])

        query = QueryBundle(questions[i]['query'])
        if questions[i]['category'] != 'faq':
            retrieved_nodes = retriever_chunk.retrieve(query)
        else:
            retrieved_nodes = retriever.retrieve(query)

        retrieved_nodes = rerank.postprocess_nodes(retrieved_nodes, query)
        retrieved_id = retrieved_nodes[0].metadata['file_name']
        answer_dict['answers'].append({"qid": questions[i]['qid'], "retrieve": int(retrieved_id[4:-4] if questions[i]['category'] != 'faq' else retrieved_id)})

    with open(answer_output, 'w', encoding='utf8') as f:
        json.dump(answer_dict, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process and retrieve answers for finance, insurance, and FAQ questions.')
    parser.add_argument('--storage_path', type=str, help='Path storing indicies and node dictionaries', default='processed_data')
    parser.add_argument('--questions_path', type=str, help='Path to JSON file containing questions', default='競賽資料集/dataset/preliminary/questions_preliminary.json')
    parser.add_argument('--output_path', type=str, help='Output path for JSON answers file', default='pred_retrieve.json')
    args = parser.parse_args()

    # Load questions
    with open(args.questions_path, 'rb') as f:
        questions_data = json.load(f)

    # Load indices
    indices = load_indices(args.storage_path)

    # Process queries and save answers
    process_queries(questions_data['questions'], indices, args.output_path)
