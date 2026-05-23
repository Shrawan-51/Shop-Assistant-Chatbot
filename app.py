import os
import time
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from google import genai
from pinecone import Pinecone,ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

Pinecone_api_key=os.getenv("PINECONE_API_KEY")
pc=Pinecone(api_key=Pinecone_api_key)

index_name="shop-product-catalog"

spec=ServerlessSpec(
    cloud='aws',region='us-east-1'
)
my_index=pc.Index(index_name)
time.sleep(1)


GEMINI_API_KEY=os.getenv('GEMINI_API_KEY')
embed_model=GoogleGenerativeAIEmbeddings(api_key=GEMINI_API_KEY,model="gemini-embedding-001",output_dimensionality=768)


vectorstore=PineconeVectorStore(
    index=my_index,
    embedding=embed_model,
    text_key='Description'
)


if "chat_history" not in st.session_state:
    st.session_state.chat_history=[]

system_message=(
    "Act as a professional salesman who want to sell his store products. " 
    "generate the answers and try to recomend good products to the customer. "
    "If a query lacks a direct answer e.g. durability, generate a response based on related features. "
    "You are a helpful and respectful shop assistant who answers queries relevant only to the shop. "
    "Please answer all questions politely. Use a conversational tone, like you're chatting with someone, "
    "not like you're writing an email. If the user asks about anything outside of the shop data like if they ask "
    "something irrelevant, simply say, 'I can only provide answers related to the shop, sir."
)

def gen_answer(system_message,chat_history,prompt):
    client = genai.Client(
        api_key=os.getenv("GEMINI_API_KEY")
    )

    chat_history.append(f"User: {prompt}")

    full_prompt = (
        f"{system_message}\n\n"
        + "\n".join(chat_history)
        + "\nAssistant:"
    )

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=full_prompt
    )

    answer = response.text

    chat_history.append(f"Assistant: {answer}")

    return answer

def get_relevant_chunk(query,vectorstore):
    results=vectorstore.similarity_search(query,k=3)
    if results:
        context = "\n\n".join([
            f"{r.page_content}\nMetadata: {r.metadata}"
            for r in results
        ])
        return context
    return "No relevant search"


def make_prompt(query,context):
    return f"Query: {query}\n\nContext:\n{context}\n\nAnswer:"

st.title("Shop Catalog Chatbot")

query=st.text_input("Ask query....")

if st.button("Get Answer"):
    if query:
        relevant_text=get_relevant_chunk(query,vectorstore)
        prompt=make_prompt(query,relevant_text)

        answer=gen_answer(system_message,st.session_state.chat_history,prompt)
        st.write("Answer: ",answer)

        with st.expander("Chat History"):
            for chat in st.session_state.chat_history:
                st.write(chat)