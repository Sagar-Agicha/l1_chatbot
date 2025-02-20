import pandas as pd
import pyodbc

import os
from dotenv import load_dotenv

load_dotenv()

server = os.getenv("SERVER")
database = os.getenv("DATABASE")
username = os.getenv("UID") 
password = os.getenv("PWD")

conn = pyodbc.connect(
    "DRIVER={ODBC Driver 17 for SQL Server};"
    f"SERVER={server};"
    f"DATABASE={database};"
    f"UID={username};"
    f"PWD={password}"
)
cursor = conn.cursor()

# cursor.execute("""
#     INSERT INTO l1_tree (user_id, user_name, phone_number, com_name, mo_name, vector_file, pdf_file)
#     VALUES (?, ?, ?, ?, ?, ?, ?)
# """, 2, "Anoop Sir", "9820011282", "Lenovo", "L14", "0", "lenevo-thinkpad-L14.pdf")


    # CREATE TABLE l1_tree (
    #     user_id INT,
    #     user_name VARCHAR(100),
    #     phone_number VARCHAR(20),
    #     com_name VARCHAR(100),
    #     mo_name VARCHAR(100),
    #     vector_file VARCHAR(255),
    #     pdf_file VARCHAR(255)
    # )

cursor.execute("""
    UPDATE l1_tree
    SET chunks_file = 'encodings/chunks_9820011282.pkl'
    WHERE user_id = 2
""")

conn.commit()
cursor.close()
conn.close()

print("CSV data successfully inserted into MS SQL Server!")
