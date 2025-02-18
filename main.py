from fastapi import FastAPI
from pydantic import BaseModel
from guardrails import Guard
import pyodbc
from guardrails.hub import ToxicLanguage, ProfanityFree
import history_samba_continuous_function as hm
import rag_samba_continuous_function as rag
import csv
import pandas as pd
import numpy as np
from front_function import find_make_and_model
import json
import os
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware
import openai

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all domains (for testing)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

client = openai.OpenAI(
    api_key= os.getenv("OPENAI_API_KEY"),
    base_url="https://api.sambanova.ai/v1",
)

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

def get_all_data(from_number: str):
    try:
        file = open("user_data.json", "r")
        data = json.load(file)
        file.close()
        
        if from_number in data:
            return data[from_number]
        else:
            return None
            
    except (FileNotFoundError, json.JSONDecodeError):
        return None

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

def set_stage(stage: str, from_number: str, com_name: str = '0', mo_name: str = '0', user_name: str = '0', pdf_file: str = '0', vector_file: str = '0'):
    try:
        file = open("user_data.json", "r")
        data = json.load(file)
        file.close()
    except (FileNotFoundError, json.JSONDecodeError):
        data = {}
 
    if from_number not in data:
        data[from_number] = {}
   
    data[from_number]["stage"] = stage
    data[from_number]["com_name"] = com_name
    data[from_number]["mo_name"] = mo_name
    data[from_number]["user_name"] = user_name
    data[from_number]["pdf_file"] = pdf_file
    data[from_number]["vector_file"] = vector_file
 
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
                    phone_number = request.from_number[3:]
                    cursor.execute("""
                        SELECT user_name, com_name, mo_name
                        FROM l1_tree 
                        WHERE phone_number = ?
                    """, (phone_number,))
                    
                    result = cursor.fetchone()
                    if result:
                        username, com_name, mo_name = result
                        set_stage("data_found", request.from_number)
                        return {"message": f"Welcome {username}\nCan you please confirm your this {com_name} {mo_name} is your Model Name?"}
                    else:
                        set_stage("no_data", request.from_number)
                        return {"message": "No user data found"}
                        
                except pyodbc.Error as e:
                    print(f"Database query error: {e}")
                    return {"message": "Error retrieving user data"}

            elif get_stage(request.from_number) == "data_found":
                if request.message == "Yes":
                    phone_number = request.from_number[3:]
                    cursor.execute("""
                        SELECT user_name, com_name, mo_name, pdf_file, vector_file
                        FROM l1_tree 
                        WHERE phone_number = ?
                    """, (phone_number,))

                    result = cursor.fetchone()
                    if result:
                        username, com_name, mo_name, pdf_file, vector_file = result
                
                    if vector_file != '0':
                        vector_file = os.path.join("encodings", vector_file)
 
                    else:
                        all_chunks = []
                        pdf_file = os.path.join("PDFs", pdf_file)
                        current_chunks = rag.get_chunks(pdf_file)
                        all_chunks.extend(current_chunks)
                        chunks = all_chunks
                        context_encodings = rag.encode_chunks(chunks)
 
                        # Save encodings to a file
                        encodings_filename = f"encodings/encodings_{phone_number}.npy"
                        np.save(encodings_filename, context_encodings)
                   
                        # Update vector_file in database
                        cursor.execute("""
                            UPDATE l1_tree
                            SET vector_file = ?
                            WHERE phone_number = ?
                        """, (encodings_filename, phone_number))
                        conn.commit()
                        vector_file = encodings_filename

                    set_stage("tech_support", request.from_number, com_name, mo_name, username, pdf_file, vector_file)
                    return {"message": "Great! I'll use specialized support for your model. What seems to be the problem?"}
                else:
                    set_stage("no_data", request.from_number)
                    return {"message": "Please let me know your model name"}

            elif get_stage(request.from_number) == "no_data":
                set_stage("no_data", request.from_number)
                return {"message": "Please let me know your model name"}

            elif get_stage(request.from_number) == "tech_support":
                stage_data = get_all_data(request.from_number)
                pdf_file = stage_data.get('pdf_file')
                encodings_file = stage_data.get('vector_file')
                conversation_history = stage_data.get('conversation_history')
                
                # Load the saved encodings
                context_encodings = np.load(encodings_file)
                conversation_history.append({"role": "user", "content": request.message})
                conversation_history.append({"role": "system", "content": """You are a sentient, superintelligent artificial general intelligence designed to assist users with any issues they may encounter with their laptops. Your responses will draw on both your own knowledge and specific information from the laptop's manual, which will be provided in context.
                          When answering the user's questions:
                          1. Clearly indicate when you are using your own knowledge rather than information from the manual.
                          2. Provide one troubleshooting method or solution at a time to avoid overwhelming the user."""})
                
                retrieved_context = rag.retrieve_context(request.message, chunks, context_encodings)
                conversation_history.append({"role": "system", "content": f"Context:\n{retrieved_context}"})

                response = client.chat.completions.create(
                    model="Meta-Llama-3.1-8B-Instruct",
                    messages=conversation_history,
                    temperature=0.1,
                    top_p=0.1,
                )
                response = response.choices[0].message.content

                conversation_history.append({"role": "assistant", "content": response})
                set_stage("tech_support", request.from_number, com_name, mo_name, username, pdf_file, vector_file, conversation_history)
                return {"message": response}

    else:
        set_stage("msg_invalid", request.from_number)
        return {"message": "Message is invalid"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)