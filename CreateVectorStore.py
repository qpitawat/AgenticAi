import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import re

VECTOR_STORE_PATH = "Data/vector_store"

def clean_text(text):
    text = text.replace('\uf70b', '้')
    text = text.replace('\uf70a', '่')
    text = text.replace('\uf701', '')
    text = text.replace('\uf712', '็')
    text = text.replace('\uf70e', '์')
    text = text.replace('\uf710', 'ั')
    text = text.replace('\uf702', 'ำ')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def create_vector_store():
    #โหลดเอกสาร PDF
    file_paths = [
        ("Data/ปฏิทินการศึกษา2568.pdf", "calendar"),
        ("Data/รายละเอียดของหลักสูตร.pdf", "curriculum"),
    ]
    docs = []
    for path, source_tag in file_paths:
        loader = PyMuPDFLoader(path)
        loaded_pages = loader.load()
        for i, page in enumerate(loaded_pages):
            raw_text = page.page_content
            cleaned = clean_text(raw_text)
            doc = Document(
                page_content=cleaned,
                metadata={
                    "page": i + 1,
                    "source": source_tag,
                    "file": os.path.basename(path),
                }
            )
            docs.append(doc)
    print(f"โหลดเอกสารทั้งหมด {len(docs)} หน้า")

    #แบ่งเอกสารเป็น Chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""],
        is_separator_regex=False,
    )
    splits = text_splitter.split_documents(docs)
    print(f"แบ่งเอกสารออกเป็น {len(splits)} chunks")

    #สร้าง Embeddings
    embedding = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    #สร้าง Vector Store จากเอกสาร
    vectordb = FAISS.from_documents(
        documents=splits,
        embedding=embedding
    )

    #บันทึก Vector Store
    vectordb.save_local(VECTOR_STORE_PATH)
    print("เรียบร้อยแล้ว")

if __name__ == "__main__":
    create_vector_store()