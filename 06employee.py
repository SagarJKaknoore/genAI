from fastapi import FastAPI
from langchain_community.document_loaders import CSVLoader
from pydantic import BaseModel
loader = CSVLoader("employees.csv")
csv_pages = loader.load()
# STEP 2: Split PDF into chunks
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    separators=["\n\n", "\n", ",", " "]
)

split_docs = splitter.split_documents(csv_pages)

print(f"Total no of chunks: {len(split_docs)}")
# STEP 3: Create embeddings and FAISS vector store
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

embedding_model = OllamaEmbeddings(
    model="nomic-embed-text"
)

vector_store = FAISS.from_documents(
    split_docs,
    embedding=embedding_model
)

vector_store.save_local("faiss_leave_policy")

print("Vector Store created")

# STEP 4: Build Retrieval Chain
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA
from langchain_ollama import ChatOllama

model = ChatOllama(
    model="gpt-oss:120b-cloud"
)

print("Model is ready")


USER_PROMPT_TEMPLATE = """
Use the following pieces of the context to answer user's question.

If you don't know the answer, just say you don't know,
don't try to make up the answer.

------------------------
{context}

Question: {question}
"""


USER_PROMPT = PromptTemplate(
    template=USER_PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)


qa_chain = RetrievalQA.from_chain_type(
    model,
    chain_type="stuff",
    retriever=vector_store.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs={
        "prompt": USER_PROMPT
    }
)

print("Retrieval chain is built successfully...")


# STEP 5: FastAPI
app = FastAPI()


class QuestionRequest(BaseModel):
    question: str

@app.post("/ask")
def ask_question(request: QuestionRequest):

    result = qa_chain.invoke({
        "query": request.question
    })

    return {
        "question": request.question,
        "answer": result["result"]
    }