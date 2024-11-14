# 2024-AI-CUP_Recursive-Retrieval
This repository solves the task of 2024 AI CUP using LlamaIndex tools. The project includes functionalities for embedding documents, applying metadata filters, and performing recursive retrievals with reranking and node references.

## Installation

### 1. Clone the repository:
```bash
git clone https://github.com/melodyxchen/2024-AI-CUP_Recursive-Retrieval.git
cd 2024-AI-CUP_Recursive-Retrieval
```
### 2. Install the required Python packages:
```bash
pip install -r requirements.txt
```
### 3. Set up API keys:
```bash
export LLAMA_CLOUD_API_KEY="your-api-key-here"
```

## Folder Descriptions

- `Preprocess/`: Scripts for processing documents before retrieval.
  - `parsing.py`: Handles pdf parsing of financial and insurance documents.
  - `index.py`: Manages document indexing and stores the indicies with node references.
- `Model/`: Scripts for running recursive retrieval.
  - `retriever.py`: Implements the retrieval and reranking of documents based on queries.
    
- `requirements.txt`: Specifies the packages required to run the project.

- `README.md`: Project documentation

## Running the Script
### 1. Parse the pdf documents:  
```bash
python Preprocess/parsing.py \
  --finance_source_path  競賽資料集/reference/finance \
  --finance_output_path processed_data/all_fin \
  --insurance_source_path  競賽資料集/reference/insurance \
  --insurance_output_path processed_data/all_fin
```
### 2. Generate and store the vector indicies
```bash
python Preprocess/index.py \
  --finance_source_path  processed_data/all_fin \
  --insurance_source_path  processed_data/all_in \
  --faq_source_path 競賽資料集/reference/faq/pid_map_content.json \
  --storage_path processed_data
```
### 3. Retrieve the answers
```bash
python retriever.py \
  --questions_path 競賽資料集/dataset/preliminary/questions_example.json \
  --output_path pred_retrieve.json
```

## Reference
https://docs.llamaindex.ai/en/stable/examples/retrievers/recursive_retriever_nodes/
https://docs.llamaindex.ai/en/stable/understanding/
https://github.com/run-llama/llama_index
