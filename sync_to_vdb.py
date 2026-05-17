import os
import time
import mysql.connector
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pinecone import Pinecone,ServerlessSpec
from dotenv import load_dotenv
from tqdm.auto import tqdm
import pandas as pd


load_dotenv()
Pinecone_api_key=os.getenv("PINECONE_API_KEY")
pc=Pinecone(api_key=Pinecone_api_key)

spec=ServerlessSpec(
    cloud='aws',region='us-east-1'
)

index_name="shop-product-catalog"
exists=[ index_info['name'] for index_info in pc.list_indexes()]

if index_name not in exists:
    pc.create_index(
        name=index_name,
        dimension=768,
        metric="dotproduct",
        spec=spec
    )
    while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)

index=pc.Index(index_name)
time.sleep(1)


db_connector=mysql.connector.connect(
    host="localhost",
    user="root",
    password=os.getenv("DB_PASSWORD"),
    database='shop_data'
)

cursor=db_connector.cursor()


#Embeding
GEMINI_API_KEY=os.getenv('GEMINI_API_KEY')
embed_model=GoogleGenerativeAIEmbeddings(api_key=GEMINI_API_KEY,model="gemini-embedding-001",output_dimensionality=768)

def fetch_data():
    query="SELECT * FROM products"
    cursor.execute(query)
    collums=[ desc[0] for desc in cursor.description]
    data=pd.DataFrame(cursor.fetchall(),columns=collums)
    return data


def sync_with_pinecone(data):
    batch_size=30
    total_batches=(len(data)+batch_size-1)//batch_size

    for i in tqdm(range(0,len(data),batch_size),desc="Processing Batches",unit='batch',total=total_batches):
        i_end=min(len(data),i+batch_size)
        batch=data.iloc[i:i_end]

        ids=[str(row['ProductID']) for _,row in batch.iterrows()]

        texts=[
            f"{row['Description']} {row['ProductName']} {row['ProductBrand']} {row['Gender']} {row['Price']} {row['PrimaryColor']}"
            for _,row in batch.iterrows()
        ]

        embeds=embed_model.embed_documents(texts)

        metadata=[
            {
                "ProductName":row['ProductName'],
                "ProductBrand":row['ProductBrand'],
                'Gender':row['Gender'],
                'Price':row['Price'],
                'PrimaryColor':row['PrimaryColor'],
                'Description':row['Description'],
            }
            for _,row in batch.iterrows()
        ]

        with tqdm(total=len(ids),desc="Upserting vectors",unit='vector') as upsert_vector:
            index.upsert(vectors=zip(ids,embeds,metadata))
            upsert_vector.update(len(ids))


def main():
    data=fetch_data()
    sync_with_pinecone(data=data)

if __name__=="__main__":
    main()

cursor.close()
db_connector.close()