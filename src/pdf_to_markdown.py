# reference: https://github.com/pymupdf/RAG/blob/main/pymupdf4llm/pymupdf4llm/llama/pdf_markdown_reader.py
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
import os 
import json

import pymupdf
from pymupdf import Document as FitzDocument
from pymupdf4llm import IdentifyHeaders, to_markdown

try:
    from llama_index.core.readers.base import BaseReader
    from llama_index.core.schema import Document as LlamaIndexDocument

    print("Successfully imported LlamaIndex")
except ImportError:
    raise NotImplementedError("Please install 'llama_index'.")


class PDFMarkdownReader(BaseReader):
    """Read PDF files using PyMuPDF library."""

    meta_filter: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None

    def __init__(
        self,
        meta_filter: Optional[
            Callable[[Dict[str, Any]], Dict[str, Any]]
        ] = None,
    ):
        self.meta_filter = meta_filter

    def load_data(
        self,
        file_path: Union[Path, str],
        extra_info: Optional[Dict] = None,
        **load_kwargs: Any,
    ) -> List[LlamaIndexDocument]:
        """Loads list of documents from PDF file and also accepts extra information in dict format.

        Args:
            file_path (Union[Path, str]): The path to the PDF file.
            extra_info (Optional[Dict], optional): A dictionary containing extra information. Defaults to None.
            **load_kwargs (Any): Additional keyword arguments to be passed to the load method.

        Returns:
            List[LlamaIndexDocument]: A list of LlamaIndexDocument objects.
        """
        if not isinstance(file_path, str) and not isinstance(file_path, Path):
            raise TypeError("file_path must be a string or Path.")

        if not extra_info:
            extra_info = {}

        if extra_info and not isinstance(extra_info, dict):
            raise TypeError("extra_info must be a dictionary.")

        # extract text header information
        hdr_info = IdentifyHeaders(file_path)

        doc: FitzDocument = pymupdf.open(file_path)
        docs = []

        for page in doc:
            docs.append(
                self._process_doc_page(
                    doc, extra_info, file_path, page.number, hdr_info
                )
            )
        return docs

    # Helpers
    # ---

    def _process_doc_page(
        self,
        doc: FitzDocument,
        extra_info: Dict[str, Any],
        file_path: str,
        page_number: int,
        hdr_info: IdentifyHeaders,
    ):
        """Processes a single page of a PDF document."""
        extra_info = self._process_doc_meta(
            doc, file_path, page_number, extra_info
        )

        if self.meta_filter:
            extra_info = self.meta_filter(extra_info)

        text = to_markdown(
            doc, pages=[page_number], hdr_info=hdr_info, write_images=False
        )
        return LlamaIndexDocument(text=text, extra_info=extra_info)

    def _process_doc_meta(
        self,
        doc: FitzDocument,
        file_path: Union[Path, str],
        page_number: int,
        extra_info: Optional[Dict] = None,
    ):
        """Processes metas of a PDF document."""
        extra_info.update(doc.metadata)
        extra_info["page"] = page_number + 1
        extra_info["total_pages"] = len(doc)
        extra_info["file_path"] = str(file_path)

        return extra_info
    

pdf_markdown_reader = PDFMarkdownReader()



def convert_to_json(pdf_file_paths):
    documents = [] #TODO: conver to dict to later for faster search
    pdf_reader = PDFMarkdownReader()

    for pdf_file in pdf_file_paths:
        doc = pdf_reader.load_data(pdf_file)
  
        for page in doc:
            page_data = page.dict()
            
            # documnet metadata
            document = {
                "page_id": page_data['id_'],
                'metadata': {
                    'title': page_data['metadata']['title'],
                    'author': page_data['metadata']['author'],
                    'file_path': page_data['metadata']['file_path'],
                    'page_num': page_data['metadata']['page'],
                    'total_pages': page_data['metadata']['total_pages']
                },
                'text': page_data['text_resource']['text']
            }
            documents.append(document)
    return documents


BASE_PATH = "./data"
pdf_file_paths = ["/".join([BASE_PATH, path]) for path in os.listdir(BASE_PATH) \
                        if path.endswith(".pdf")]

# print(pdf_file_paths[2:]) # testing 
# documents = convert_to_json(pdf_file_paths[2:])
documents = convert_to_json(pdf_file_paths)

with open('./intermediate_data/extract_pdf_md_data.json', 'w') as f:
    json.dump(documents, f)