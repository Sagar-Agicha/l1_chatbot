import pandas as pd
import pyodbc

conn = pyodbc.connect(
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=outsystems1.database.windows.net;"
    "DATABASE=OUTSYSTEM_API;"
    "UID=Galaxy;"
    "PWD=OutSystems@123"
)
cursor = conn.cursor()

cursor.execute("""
    CREATE TABLE l1_tree (
        user_id INT,
        user_name VARCHAR(100),
        phone_number VARCHAR(20),
        com_name VARCHAR(100),
        mo_name VARCHAR(100),
        vector_file VARCHAR(255),
        pdf_file VARCHAR(255)
    )
""")

cursor.execute("""
    INSERT INTO l1_tree (user_id, user_name, phone_number, com_name, mo_name, vector_file, pdf_file)
    VALUES (?, ?, ?, ?, ?, ?, ?)
""", 1, "Sagar Agicha", "7977587238", "Lenovo", "L14", "0", "lenevo-thinkpad-L14.pdf")

conn.commit()
cursor.close()
conn.close()

print("CSV data successfully inserted into MS SQL Server!")
