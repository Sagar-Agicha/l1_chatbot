from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from guardrails import Guard
import pyodbc
from guardrails.hub import ToxicLanguage, ProfanityFree
import history_samba_continuous_function as hm
import rag_samba_continuous_function as rag
import uuid
import csv
import pandas as pd
import numpy as np
from front_function import find_make_and_model
import json
import os
import datetime as dt
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import openai
import logging
import pickle
#import psycopg2
from typing import Dict

processing_store: Dict[str, Dict] = {}

app = FastAPI()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('chatbot.log'),
        logging.StreamHandler()
    ]
)

model = SentenceTransformer('all-mpnet-base-v2')  # Best model for general-purpose semantic matching

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

class get_results(BaseModel):
    phone_number: str
    unique_id:str

# conn1 = psycopg2.connect(
#     dbname="postgres",
#     user="postgres",
#     password="admin",
#     host="localhost",
#     port="5432"
# )

link_url = "https://api.goapl.com"

# cursor1 = conn1.cursor()
# cursor1.execute("ROLLBACK")

def get_all_data(from_number: str):
    try:
        with open("user_data.json", "r") as file:
            data = json.load(file)
        
        return data.get(from_number, None)
            
    except (FileNotFoundError, json.JSONDecodeError):
        return None

def get_stage(from_number: str):
    try:
        with open("user_data.json", "r") as file:
            data = json.load(file)
        
        return data.get(from_number, {})
            
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def check_text_content(text):
    try:
        validation_result = guard.validate(text)
        return {
            'is_valid': validation_result.validation_passed,
            'message': "Your message contains content that violates our community guidelines. Please ensure your message is respectful and appropriate before trying again."
        }
    except Exception as e:
        return {
            'is_valid': False,
            'message': "We encountered an issue processing your message. Please try again with different wording."
        }

def set_stage(stage: str, phone_number: str, com_name: str = '0', mo_name: str = '0', user_name: str = '0', pdf_file: str = '0', vector_file: str = '0', conversation_history: list = [], chunks_file: str = '0', last_uuid: list = [], solution_type: str = '0', rag_no: int = 0, last_time: str = '0'):
    try:
        with open("user_data.json", "r") as file:
            data = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        data = {}

    if phone_number not in data:
        data[phone_number] = {}

    # Update only if the new value is not the default '0'
    if stage != '0':
        data[phone_number]["stage"] = stage
    if com_name != '0':
        data[phone_number]["com_name"] = com_name
    if mo_name != '0':
        data[phone_number]["mo_name"] = mo_name
    if user_name != '0':
        data[phone_number]["user_name"] = user_name
    if pdf_file != '0':
        data[phone_number]["pdf_file"] = pdf_file
    if vector_file != '0':
        data[phone_number]["vector_file"] = vector_file
    if conversation_history:
        data[phone_number]["conversation_history"] = conversation_history
    if chunks_file != '0':
        data[phone_number]["chunks_file"] = chunks_file
    if last_uuid:
        data[phone_number]["last_uuid"] = last_uuid
    if solution_type != '0':
        data[phone_number]["solution_type"] = solution_type
    if rag_no != 0:
        data[phone_number]["rag_no"] = rag_no
    if last_time != 0:
        data[phone_number]["last_time"] = last_time

    with open("user_data.json", "w") as file:
        json.dump(data, file, indent=4)

    return "Stage set successfully"

def get_best_matching_tag(user_query):
    cursor1.execute("SELECT DISTINCT unnest(tags_list) AS tag FROM decision_tree")
    tags = [row[0] for row in cursor1.fetchall()]
    
    if not tags:
        return None
    
    tag_embeddings = model.encode(tags)
    query_embedding = model.encode([user_query])
    
    similarities = cosine_similarity(query_embedding, tag_embeddings)
    best_match_idx = np.argmax(similarities)
    best_match_score = similarities[0][best_match_idx] * 100  # Convert to percentage

    # Check if the best match score is above 80
    if best_match_score < 80:
        return None, None, None, None

    cursor1.execute("SELECT dt_id FROM decision_tree WHERE %s = ANY(tags_list) LIMIT 1", (tags[best_match_idx],))
    result = cursor1.fetchone()

    cursor1.execute("SELECT question_text FROM decision_tree WHERE type_id = 'Issue' and dt_id = %s", (result[0],))
    dt_data = cursor1.fetchone()

    cursor1.execute("SELECT action_id FROM decision_tree WHERE type_id = 'Issue' and dt_id = %s", (result[0],))
    action = cursor1.fetchone()
    
    dt_id = result[0]
    question_text = dt_data[0]
    action = action[0]
    
    if not result:
        return None, None, None, None
    
    if not dt_data:
        return None, None, None, None
    
    if not question_text:
        return None, None, None, None
    
    if not action:
        return None, None, None, None
    
    return tags[best_match_idx], dt_id, question_text, action

def store_user_interaction(phone_number: str, stage: str = '0', solution_number: int = 0, result: dict = None, issue: str = None, dt_id: int = None, action: str = None, yes_id: str = None, user_name: str = None):
    interaction = {
        "phone_number": phone_number,
        "stage": stage,
        "issue": issue,
        "dt_id": dt_id,
        "solution_number": solution_number,
        "timestamp": str(dt.datetime.now()),
        "user_name": user_name,
        "result": result,
        "action": action,
        "yes_id": yes_id,
    }
    
    try:
        with open('user_interactions.json', 'r') as file:
            interactions = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        interactions = []
    
    # Update or append interaction
    updated = False
    for i, existing in enumerate(interactions):
        if existing['phone_number'] == phone_number:
            interactions[i] = interaction
            updated = True
            break
    
    if not updated:
        interactions.append(interaction)
    
    # Save updated interactions
    with open('user_interactions.json', 'w') as file:
        json.dump(interactions, file, indent=4)

def get_user_interaction(phone_number: str) -> dict:
    """Get stored interaction details for a user"""
    try:
        with open('user_interactions.json', 'r') as file:
            interactions = json.load(file)
            for interaction in interactions:
                if interaction['phone_number'] == phone_number:
                    return interaction
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def encodings_process(unique_id: str, pdf_file: str, phone_number: str, com_name: str, mo_name: str, username: str):
    all_chunks = []
    pdf_file = os.path.join("PDFs", pdf_file)
    current_chunks = rag.get_chunks(pdf_file)
    all_chunks.extend(current_chunks)
    chunks = all_chunks
    context_encodings = rag.encode_chunks(chunks)

    # Save chunks to a file
    chunks_filename = f"encodings/chunks_{phone_number}.pkl"
    with open(chunks_filename, 'wb') as f:
        pickle.dump(chunks, f)

    # Save encodings to a file
    encodings_filename = f"encodings/encodings_{phone_number}.npy"
    np.save(encodings_filename, context_encodings)

    # Update vector_file in database
    cursor.execute("""
        UPDATE l1_tree
        SET vector_file = ?, chunks_file = ?
        WHERE phone_number = ?
    """, (encodings_filename, chunks_filename, phone_number))
    conn.commit()
    vector_file = encodings_filename
    set_stage("tech_support", "+91"+phone_number, com_name, mo_name, username, pdf_file, vector_file, chunks_filename)
    result = phone_number
    # Use the phone number as the key
    key = phone_number
    if key in processing_store:
        processing_store[key]["result"] = result
    else:
        logging.error(f"Key {key} not found in processing_store")

@app.post("/get_result")
async def get_result(request:get_results):
    """
    Checks if the background task has completed and returns the result if available.
    """
    key = request.phone_number[3:]
    if key in processing_store:
        if processing_store[key]["result"]:
            return {"message": "Completed", "flag": ""}
        else:
            return {"message": "Processing not complete yet", "flag": "No"}
    else:
        return {"message": "No processing found for the provided details.", "flag": "No"}

@app.post("/webhook")
async def webhook(request: WebhookData, background_tasks: BackgroundTasks):
    phone_number = request.from_number[3:]  # Remove the '+91' prefix
    user_validation = check_text_content(request.message)
    if user_validation['is_valid']:
        logging.info(f"Processing request from {phone_number}")
        if get_stage(request.from_number) == {}:
            phone_number = request.from_number[3:]
            cursor.execute("""
                SELECT user_name
                FROM l1_tree 
                WHERE phone_number = ?
            """, (phone_number,))
            
            result = cursor.fetchone()

            if result:
                phone_number = request.from_number[3:]
                cursor.execute("""
                    SELECT user_name, com_name, mo_name
                    FROM l1_tree 
                    WHERE phone_number = ?
                """, (phone_number,))
                
                result = cursor.fetchone()
                if result:
                    username, com_name, mo_name = result
                    set_stage("data_found", request.from_number, com_name, mo_name, username)
                    return {"message": f"Welcome {username}\nCan you please confirm your this {com_name} {mo_name} is your Model Name?",
                            "flag":""}
                else:
                    set_stage("no_data", request.from_number)
                    return {"message": "No user data found do you enter a new model name?",
                            "flag":"No"}

        elif get_stage(request.from_number)['stage'] == "data_found":
            user_response = request.message.lower()
            yes_variations = ["yes", "yeah", "yep", "sure", "correct", "right", "ok", "okay"]
            no_variations = ["no", "not", "nope", "nah", "wrong", "incorrect"]
            
            # Direct string matching instead of embeddings
            user_response = user_response.strip().lower()
            
            # Check if response contains any yes variations
            max_similarity = 1.0 if any(yes_word in user_response for yes_word in yes_variations) else 0.0
            no_max_similarity = 1.0 if any(no_word in user_response for no_word in no_variations) else 0.0
            
            if max_similarity > 0.7:
                phone_number = request.from_number[3:]
                cursor.execute("""
                    SELECT user_name, com_name, mo_name, pdf_file, vector_file, chunks_file
                    FROM l1_tree 
                    WHERE phone_number = ?
                """, (phone_number,))

                result = cursor.fetchone()
                if result:
                    username, com_name, mo_name, pdf_file, vector_file, chunks_filename = result
                
                if vector_file != '0' and chunks_filename == '0':
                    vector_file = vector_file
                    chunks_filename = chunks_filename
                    set_stage("tech_support", request.from_number, com_name, mo_name, username, pdf_file, vector_file, chunks_filename)
                    return {"message": "Great! I'll use specialized support for your model. What seems to be the problem?",
                            "flag":""}
 
                else:
                    unique_id = str(uuid.uuid4())
                    logging.info(f"Adding background task for {phone_number}")
                    processing_store[phone_number] = {"uid": unique_id, "result": None}
                    background_tasks.add_task(
                        encodings_process,
                        unique_id=request.message,
                        pdf_file=pdf_file,
                        phone_number=phone_number,
                        com_name=com_name,
                        mo_name=mo_name,
                        username=username
                    )
                    return {"message": "Great! I'll use specialized support for your model. What seems to be the problem?",
                            "flag":"Yes"}
 
            elif no_max_similarity > 0.7:
                set_stage("no_data", request.from_number)
                return {"message": "Please let me know your model name",
                        "flag":""}

            else:
                set_stage("data_found", request.from_number)
                return {"message": f"Please Say Yes or No",
                        "flag":""}

        elif get_stage(request.from_number)['stage'] == "no_data":
            set_stage("no_data", request.from_number)
            return {"message": "Please let me know your model name",
                    "flag":"No"}

        elif get_stage(request.from_number)['stage'] == "tech_support":
            stage_data = get_all_data(request.from_number)
            pdf_file = stage_data.get('pdf_file')
            encodings_file = stage_data.get('vector_file')
            chunks_file = stage_data.get('chunks_file')
            conversation_history = stage_data.get('conversation_history', [])

            result, dt_id, question_text, action = get_best_matching_tag(request.message)

            
            if chunks_file != '0':
                with open(chunks_file, 'rb') as f:
                    chunks = pickle.load(f)
            else:
                chunks = []

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)