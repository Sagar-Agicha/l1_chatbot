import pandas as pd
import pyodbc
import os

server = os.getenv("SERVER")
database = os.getenv("DATABASE")
username = os.getenv("UID") 
password = os.getenv("PWD")

# Read CSV into DataFrame
csv_file = "C:/Users/Sagar Agicha/Downloads/data-1739790854152.csv"  # Path to your uploaded file
df = pd.read_csv(csv_file)

# MS SQL Server connection
conn = pyodbc.connect(
    "DRIVER={ODBC Driver 17 for SQL Server};"
    f"SERVER={server};"
    f"DATABASE={database};"
    f"UID={username};"
    f"PWD={password}"
)
cursor = conn.cursor()

# # Insert data row by row
# for index, row in df.iterrows():
#     cursor.execute("""
#         INSERT INTO decision_tree (dt_id, question_id, level_id, question_text, parent_id, link_id, action_id, tags_list)
#         VALUES (?, ?, ?, ?, ?, ?, ?, ?)
#     """, row.dt_id, row.question_id, row.level_id, row.question_text, row.parent_id, row.link_id, row.action_id, str(row.tags_list))


cursor.execute("SELECT * FROM decision_tree ORDER BY dt_id, question_id ASC")
datas = cursor.fetchall()
#print(datas)

for data in datas:
    print(data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8] + "\n")

conn.commit()
cursor.close()
conn.close()

print("CSV data successfully inserted into MS SQL Server!")
