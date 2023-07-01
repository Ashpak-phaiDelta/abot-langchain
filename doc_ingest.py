
import os
from tempfile import _TemporaryFileWrapper

from doc_db import db

from langchain.document_loaders import (
    CSVLoader,
    EverNoteLoader,
    PyMuPDFLoader,
    TextLoader,
    UnstructuredEmailLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredExcelLoader
)

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from typing import List


LOADER_MAPPING = {
    ".csv": (CSVLoader, {}),
    # ".docx": (Docx2txtLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".enex": (EverNoteLoader, {}),
    ".epub": (UnstructuredEPubLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".pdf": (PyMuPDFLoader, {}),
    ".ppt": (UnstructuredPowerPointLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
    ".xlsx": (UnstructuredExcelLoader, {})
    # Add more mappings for other file extensions and loaders as needed
}

SPLIT_CHUNK_SIZE = 500
"""How many characters to split each chunk at"""
SPLIT_CHUNK_OVERLAP = 50
"""How many characters between two chunks are same"""


def dump_documents_to_db(documents: List[Document]):
    # Split document into chunks (with some overlapping content)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=SPLIT_CHUNK_SIZE,
        chunk_overlap=SPLIT_CHUNK_OVERLAP
    )
    texts: List[Document] = text_splitter.split_documents(documents)

    db.add_documents(texts)
    # db.delete_collection()

def upload_file(f: _TemporaryFileWrapper):
    ext = os.path.splitext(f.name)[-1].lower()

    if ext in LOADER_MAPPING:
        loader_class, loader_args = LOADER_MAPPING[ext]
        loader = loader_class(f.name, **loader_args)
        dump_documents_to_db(loader.load())
        return

    raise ValueError(f"Unsupported file extension '{ext}'")
