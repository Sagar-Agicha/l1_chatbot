import pandas as pd
import pyodbc
import os
import datetime as dt
import pytz
import uuid

server = os.getenv("SERVER")
database = os.getenv("DATABASE")
username = os.getenv("UID") 
password = os.getenv("PWD")

def store_messages():
    # Connect to SQL Server
    conn = pyodbc.connect(
        f"DRIVER={{ODBC Driver 17 for SQL Server}};"
        f"SERVER={{outsystems1.database.windows.net}};"
        f"DATABASE={{OUTSYSTEM_API}};"
        f"UID={{Galaxy}};"
        f"PWD={{OutSystems@123}}"
    )
    cursor = conn.cursor()

    ist_timezone = pytz.timezone("Asia/Kolkata")
    current_datetime = datetime.now(ist_timezone)
    cursor.execute(
        """
        INSERT INTO WhatsAppMsgs 
        (id, uuid, session_key, message_text, media_url, media_type, media_mime_type, created_at, remote_phone_number, _2chat_link, channel_phone_number, sent_by)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """,
        (
            id,
            uuid,
            session_key,
            message_text,
            media_url,
            media_type,
            media_mime_type,
            current_datetime,
            remote_phone_number,
            _2chat_link,
            channel_phone_number,
            sent_by,
        ),
    )
    conn.commit()

conn = pyodbc.connect(
    "DRIVER={ODBC Driver 17 for SQL Server};"
    f"SERVER={server};"
    f"DATABASE={database};"
    f"UID={username};"
    f"PWD={password}"
)
cursor = conn.cursor()

ist_timezone = pytz.timezone("Asia/Kolkata")
current_datetime = dt.datetime.now(ist_timezone)
id = str(uuid.uuid4())
uuid_id = str(uuid.uuid4())
session_key = str(uuid.uuid4())
cursor.execute(
    """
    INSERT INTO l1_chat_history 
    (uuid, session_key, message_text, response, remote_phone_number, channel_phone_number, created_at, sent_by)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
""",
    (
        uuid_id,
        session_key,
        "msg",
        "res",
        "phne",
        "91+9322261280",
        str(current_datetime),
        "user",
    ),
)
conn.commit()

conn.commit()
cursor.close()
conn.close()

print("CSV data successfully inserted into MS SQL Server!")
