from fastapi import FastAPI
from pydantic import BaseModel
from guardrails import Guard
import pyodbc
from guardrails.hub import ToxicLanguage, ProfanityFree
from flask_socketio import SocketIO, send, emit
import history_samba_continuous_function as hm
import rag_samba_continuous_function as rag
import csv
import pandas as pd
from front_function import find_make_and_model
import json
import os
from datetime import datetime

app = FastAPI()

# SQL Server connection details
server = "outsystems1.database.windows.net"
database = "OUTSYSTEM_API"
username = "Galaxy"
password = "OutSystems@123"

try:
    conn = pyodbc.connect(
        f"DRIVER={{ODBC Driver 17 for SQL Server}};"
        f"SERVER={server};"
        f"DATABASE={database};"
        f"UID={username};"
        f"PWD={password}"
    )
    cursor = conn.cursor()
    
    print("Database connection successful")
except pyodbc.Error as db_err:
    print(f"Database connection error: {db_err}")

guard = Guard().use_many(
    ToxicLanguage(validation_method="sentence", threshold=0.8),
    ProfanityFree()
)

class WebhookData(BaseModel):
    message: str
    from_number: str

def get_stage(from_number: str):
    file = open("user_data.json", "r")
    data = json.load(file)
    file.close()
    return data["stage"]

def set_stage(stage: str, from_number: str):
    file = open("user_data.json", "r")
    data = json.load(file)
    data[from_number][stage] = stage
    file.close()
    file = open("user_data.json", "w")
    json.dump(data, file)
    file.close()
    return "Stage set successfully"

@app.post("/webhook")
async def webhook(request: WebhookData):
    if request.message_type == "text":
        validated_message = guard.validate(request.message)
        if validated_message.is_valid:
                if get_stage(request.from_number) is None:
                    try:
                        phone_number = "+91" + request.from_number
                        cursor.execute("""
                            SELECT user_name, mo_name 
                            FROM l1_tree 
                            WHERE phone_number = ?
                        """, (phone_number,))
                        
                        result = cursor.fetchone()
                        if result:
                            username, model_name = result
                            set_stage("data_found", request.from_number)
                            return {"message": f"Welcome {username}\nCan you please confirm your this {model_name} is your Model Name?"}
                        else:
                            set_stage("no_data", request.from_number)
                            return {"message": "No user data found"}
                            
                    except pyodbc.Error as e:
                        print(f"Database query error: {e}")
                        return {"message": "Error retrieving user data"}

                elif get_stage(request.from_number) == "data_found":
                    if request.message == "Yes":
                        set_stage("data_confirmed", request.from_number)
                        return {"message": "Thank you for confirming your model name"}
                    else:
                        set_stage("no_data", request.from_number)
                        return {"message": "Thank you for confirming your model name"}
        else:
            set_stage("msg_invalid", request.from_number)
            return {"message": "Message is invalid"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)