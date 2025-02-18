from fastapi import FastAPI
from pydantic import BaseModel
from guardrails import Guard
import pyodbc
from guardrails.hub import ToxicLanguage, ProfanityFree
import history_samba_continuous_function as hm
import rag_samba_continuous_function as rag
import csv
import pandas as pd
from front_function import find_make_and_model
import json
import os
from datetime import datetime

app = FastAPI()

server = os.getenv("SERVER")
database = os.getenv("DATABASE")
username = os.getenv("UID") 
password = os.getenv("PWD")

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
    try:
        file = open("user_data.json", "r")
        data = json.load(file)
        file.close()
        
        if from_number in data:
            return data[from_number]["stage"]
        else:
            return None
            
    except (FileNotFoundError, json.JSONDecodeError):
        return None

def check_text_content(text):
    try:
        validation_result = guard.validate(text)
        print(f"User content validation result: {validation_result.validation_passed}")
        return {
            'is_valid': validation_result.validation_passed,
            'message': "Your message contains content that violates our community guidelines. Please ensure your message is respectful and appropriate before trying again."
        }
    except Exception as e:
        print(f"User content validation error: {str(e)}")
        return {
            'is_valid': False,
            'message': "We encountered an issue processing your message. Please try again with different wording."
        }

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
    user_validation = check_text_content(request.message)
    if user_validation['is_valid']:
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
                    return {"message": "Please let me know your model name"}

            elif get_stage(request.from_number) == "no_data":
                set_stage("data_found", request.from_number)
                return {"message": "Please let me know your model name"}
    else:
        set_stage("msg_invalid", request.from_number)
        return {"message": "Message is invalid"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)