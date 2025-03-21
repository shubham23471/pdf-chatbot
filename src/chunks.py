import json
import hashlib
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

# Define markdown headers for splitting
headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
    ("####", "Header 4"),
]
markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

# Define character-level text splitter
chunk_size = 175
chunk_overlap = 20
text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def generate_chunk_id(page_id, chunk_type, chunk_index, header):
    """Generate a unique ID for each chunk."""
    id_base = f"{page_id}-{chunk_type}-{chunk_index}-{header}"
    return hashlib.md5(id_base.encode()).hexdigest()[:16]


with open("./intermediate_data/extracted_pdf_md_data.json") as f:
    markdown_data = json.load(f)

chunks = []  # Store all processed chunks

for md_doc in markdown_data:
    page_id = md_doc["page_id"]
    metadata = md_doc["metadata"]

    # Markdown-based 
    md_splits = markdown_splitter.split_text(md_doc["text"])

    # Character-based 
    text_splits = text_splitter.split_documents(md_splits)
    md_doc.pop("text")

    for i, split in enumerate(text_splits):
        split_data = split.model_dump()

        chunk_id = generate_chunk_id(page_id, "markdown_chunk", i, split_data["metadata"].get("headers", ""))
        prev_chunk_id = "None" if len(chunks) == 0 else chunks[-1]["chunk_id"]
        next_chunk_id = None  # Will be updated in the next iteration


        chunk = {
            "chunk_id": chunk_id,
            "prev_chunk_id": prev_chunk_id,
            "next_chunk_id": "None",  
            "page_id": page_id,
            "metadata": {**metadata, "headers": split_data["metadata"]},
            "page_content": split_data["page_content"],
            "chunk_index": i,
        }

        if chunks:
            chunks[-1]["next_chunk_id"] = chunk_id

        chunks.append(chunk)


with open("./intermediate_data/chunks.json", "w") as f:
    json.dump(chunks, f, indent=4)

print(f"Processed {len(chunks)} chunks and saved to chunks.json")