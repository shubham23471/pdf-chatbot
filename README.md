RAG pipeline to seamlessly chat with your PDF locally. 

## Architecture Overview

1. **Document Processing Pipeline** 

   `PDF Extraction → Text Chunking → Embedding Generation → Vector Storage`
- pymupdf and llama_index : To extract data from PDF and convert into markdown. 
- `all-MiniLM-L6-v2` is used as embedding model from sentence transformer. 

2. **Query Pipeline**

   `User Query → Embedding Generation → Similarity Search → Context Assembly → LLM Response`
- meta-llama/Llama-3.1-8B-Instruct : is the model of choice. 

3. **System Integration** [WIP]

   `Streamlit Frontend ↔ Flask API Backend ↔ LangChain + Llama 3 Engine`



## TODO
[x] Baseline response from Model
[]  Evalutaion Metrics