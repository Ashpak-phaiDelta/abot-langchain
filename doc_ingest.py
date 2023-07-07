
import os
from typing import IO
from tempfile import _TemporaryFileWrapper

from doc_db import db

from langchain.document_loaders import UnstructuredFileIOLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from typing import List


class UnstructuredFileIOMetadataLoader(UnstructuredFileIOLoader):
    def _get_metadata(self) -> dict:
        return {"source": self.unstructured_kwargs.get('metadata_filename')}


SPLIT_CHUNK_SIZE = 300
"""How many characters to split each chunk at"""
SPLIT_CHUNK_OVERLAP = 20
"""How many characters between two chunks are same"""


def dump_documents_to_db(documents: List[Document]) -> int:
    # Split document into chunks (with some overlapping content)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=SPLIT_CHUNK_SIZE,
        chunk_overlap=SPLIT_CHUNK_OVERLAP
    )
    texts: List[Document] = text_splitter.split_documents(documents)

    # db.delete_collection()
    db.add_documents(texts)
    db.persist()
    return len(texts)

def handle_document(file: IO) -> int:
    try:
        metadata_name = os.path.basename(file.name)
        loader = UnstructuredFileIOMetadataLoader(file, metadata_filename=metadata_name)
        return dump_documents_to_db(loader.load())
    finally:
        pass


def upload_files(*files: _TemporaryFileWrapper):
    for file in files:
        with open(file.name, 'rb') as f:
            print("Uploading file %s..." % file.name)
            num_document_chunks = handle_document(f)
            print("%s uploaded, %d chunks" % (file.name, num_document_chunks))
