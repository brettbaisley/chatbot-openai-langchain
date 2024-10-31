from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema import StrOutputParser
from langchain.prompts.chat import HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.prompts import ChatPromptTemplate

template: str = """/
    You are a customer support specialist.
    If "Resolution Steps" section is found, you should give all resolution steps by step as in the document.
    If you don't understand the question, ask the user to rephrase the issue.
    If you still don't know the answer, just say you don't know.
    Be specific as possible. 
    
    You assist users with technical responses based on {context}. 
    
    Question: {question}. 
    """

system_message_prompt_template = SystemMessagePromptTemplate.from_template(template)

human_message_prompt_template = HumanMessagePromptTemplate.from_template(
    input_variables=["question", "context"], 
    template="{question}"
    )

chat_prompt_template = ChatPromptTemplate.from_messages([
    system_message_prompt_template, 
    human_message_prompt_template
    ])





# Load and define the embeddings and model to use
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
model = ChatOpenAI(model_name="gpt-4-turbo")

# Load the FAISS vector store from the file
vector_store = FAISS.load_local("faiss_index", OpenAIEmbeddings(), allow_dangerous_deserialization=True)

# Convert FAISS vector index into retriever
retriever = vector_store.as_retriever()


# Define a function to interact with the chain
def chat_with_context(retriever, user_query):
    """Generate a response to a user query."""
    chain = (
        { "context": retriever, "question": RunnablePassthrough()}
        | chat_prompt_template
        | model 
        | StrOutputParser()
        )
    return chain.invoke(user_query)



# Example chat interaction
while True:
    user_input = input("\nAsk a question: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    
    print("Answer: ", chat_with_context(retriever, user_input))