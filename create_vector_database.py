from langchain.embeddings import HuggingFaceEmbeddings
from llama_index.core import SimpleDirectoryReader
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

def create_vector_database(
    input_dir="./data",
    model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    device="cuda",
    chunk_size=700,
    chunk_overlap=70,
    persist_directory="./chroma_db/vd01",
    collection_name="crag",
    normalize_embeddings=True,
    show_progress=True,
    recursive=True,
    strip_whitespace=True
):
    """
    Creates and persists a vector database from documents.
    """
    
    model_kwargs = {"device": device}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs={"normalize_embeddings": normalize_embeddings},
        show_progress=show_progress
    )

    reader = SimpleDirectoryReader(input_dir=input_dir, recursive=recursive)
    documents = reader.load_data()

    raw_knowledge_base = [doc.to_langchain_format() for doc in documents]
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, strip_whitespace=strip_whitespace
    )
    docs_processed = text_splitter.split_documents(raw_knowledge_base)

    knowledge_vector_database = Chroma.from_documents(
        docs_processed,
        embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name
    )

    knowledge_vector_database.persist()
    return knowledge_vector_database
