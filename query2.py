from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema import StrOutputParser
from langchain_core.prompts import PromptTemplate

template: str = """/
    You are a customer support specialist.
    If "Resolution Steps" section is found, you should give all resolution steps by step as in the document.
    If you don't understand the question, ask the user to rephrase the issue.
    If you still don't know the answer, just say you don't know.
    Be specific as possible. 
    
    You assist users with technical responses based on {context}. 
    
    Question: {question}. 
    """



# Load and define the embeddings and model to use
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
model = ChatOpenAI(model_name="gpt-4-turbo")

# Load the FAISS vector store from the file
vector_store = FAISS.load_local("faiss_index", OpenAIEmbeddings(), allow_dangerous_deserialization=True)

# Convert FAISS vector index into retriever
retriever = vector_store.as_retriever(
    search_type = "similarity",
    search_kwargs = { "k": 3 }
)


custom_rag_prompt = PromptTemplate.from_template(template)
rag_chain = (
    { "context": retriever, "question": RunnablePassthrough()}
    | custom_rag_prompt
    | model 
    | StrOutputParser()
    )



# Define a function to interact with the chain
def vector_search_query(user_query):
    """Generate a response to a user query."""
    return rag_chain.invoke(user_query)



# Example chat interaction
while True:
    user_input = input("\nAsk a question: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    
    print("Answer: ", vector_search_query(user_input))