import os
import time
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from pinecone import Pinecone,ServerlessSpec
from dotenv import load_dotenv

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

from langchain_pinecone import PineconeVectorStore

vectorstore=PineconeVectorStore(
    index=my_index,
    embedding=embed_model,
    text_key='Description'
)

query="What is the price of Superstar Bold 2 product?"

ls = vectorstore.similarity_search(
    query,
    k=1
)


chat_history=[]


system_message=(
    "Act as a professional sales man who want to sell his store products, generate the answers and try to recomend good products to the customer"
    "If a query lacks a direct answer e.g. durability, generate a response based on related features. "
    "You are a helpful and respectful shop assistant who answers queries relevant only to the shop. "
    "Please answer all questions politely. Use a conversational tone, like you're chatting with someone, "
    "not like you're writing an email. If the user asks about anything outside of the shop data like if they ask "
    "something irrelevant, simply say, 'I can only provide answers related to the shop, sir."
)

def gen_answer(system_message,chat_history,prompt):
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model=genai.GenerativeModel('gemini-2.5-flash')

    # append prompt to chat history
    chat_history.append(f"User: {prompt}")

    # combine system message to chat history
    full_prompt=f"{system_message}\n\n" + "\n".join(chat_history)+"\nAssistant:"

    # gen response
    response=model.generate_content(full_prompt).text
    chat_history.append(f"Assistant: {response}")

    return response


def get_relevant_chunk(query,vectorstore):
    results=vectorstore.similarity_search(query,k=1)
    if results:
        metadata=results[0].metadata
        context=(
            f"Product Name: {metadata.get('ProductName','Not Available')}\n"
            f"Brand: {metadata.get('ProductBrand','Not Available')}\n"
            f"Price: {metadata.get('Price','Not Available')}\n"
            f"Color: {metadata.get('PrimaryColor','Not Available')}\n"
            f"Description: {results[0].page_content}"
        )
        return context
    return "No relevant search"


def make_prompt(query,context):
    return f"Query: {query}\n\nContext:\n{context}\n\nAnswer:"


def main():
    query="What is the price of Superstar Bold 2 product? "
    relevant_text=get_relevant_chunk(query,vectorstore)
    prompt=make_prompt(query,relevant_text)

    answer=gen_answer(system_message,chat_history,prompt)
    print("Answer:",answer)

    query2="Can you tell me about the  design of this product?"
    relevant_text2=get_relevant_chunk(query2,vectorstore)
    prompt2=make_prompt(query2,relevant_text2)
    answer2=gen_answer(system_message,chat_history,prompt2)
    print(answer2)


if __name__=="__main__":
    main()