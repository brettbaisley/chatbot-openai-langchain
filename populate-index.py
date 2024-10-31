from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# ---------------------- #

loader = DirectoryLoader("./fake_docs", glob="**/*.md", loader_cls=TextLoader)
docs = loader.load()
# print(docs)

# ---------------------- #

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
split_docs = text_splitter.split_documents(docs)
# print(split_docs)

# ---------------------- #

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vector_store = FAISS.from_documents(split_docs, embeddings)
vector_store.save_local("faiss_index")

print("Documents indexed and stored in FAISS vector store.")