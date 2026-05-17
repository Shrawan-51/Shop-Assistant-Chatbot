import mysql.connector
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

csv_file="shop-product-catalog.csv"
data=pd.read_csv(csv_file)

db_connector=mysql.connector.connect(
    host="localhost",
    user="root",
    password=os.getenv("db_password"),
    database='shop_data'
)

cursor=db_connector.cursor()

for index,row in data.iterrows():
    sql="""
    INSERT INTO products(ProductID,ProductName,ProductBrand,Gender,Price,Description,PrimaryColor)
    VALUES(%s,%s,%s,%s,%s,%s,%s)
    """
    cursor.execute(sql,tuple(row))

db_connector.commit()

cursor.close()
db_connector.close()