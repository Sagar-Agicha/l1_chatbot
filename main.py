from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from guardrails import Guard
import pyodbc
from guardrails.hub import ToxicLanguage, ProfanityFree
import history_samba_continuous_function as hm
import rag_samba_continuous_function as rag
import uuid
import requests
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
from typing import Dict
import pytz

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
    uuid_id: str
    session_id: str 
    message: str
    from_number: str

class get_results(BaseModel):
    phone_number: str
    unique_id:str

link_url = "https://api.goapl.com"

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

def set_stage(stage: str, phone_number: str, com_name: str = '0', mo_name: str = '0', user_name: str = '0', pdf_file: str = '0', vector_file: str = '0', conversation_history: list = [], chunks_file: str = '0', last_uuid: list = [], solution_type: str = '0', rag_no: int = 0, last_time: str = '0', session_key: str = '0'):
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
    if session_key != '0':
        data[phone_number]["session_key"] = session_key

    with open("user_data.json", "w") as file:
        json.dump(data, file, indent=4)

    return "Stage set successfully"

def clear_stage(phone_number: str):
    # Clear user_data.json
    try:
        with open("user_data.json", "r") as file:
            data = json.load(file)
        # Remove the entry for this phone number if it exists
        if phone_number in data:
            del data[phone_number]
        with open("user_data.json", "w") as file:
            json.dump(data, file, indent=4)
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    # Clear user_interactions.json
    try:
        with open("user_interactions.json", "r") as file:
            interactions = json.load(file)
        # Remove the entry for this phone number
        interactions = [i for i in interactions if i['phone_number'] != phone_number]
        with open("user_interactions.json", "w") as file:
            json.dump(interactions, file, indent=4)
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    return "Stage cleared successfully"

def get_best_matching_tag(user_query):
    # Fetch all distinct tags and strip extra whitespace
    cursor.execute("""
        SELECT DISTINCT LTRIM(RTRIM(value)) AS tag 
        FROM decision_tree 
        CROSS APPLY STRING_SPLIT(tags_list, ',')
    """)
    rows = cursor.fetchall()
    tags = [row[0] for row in rows]
    
    if not tags:
        return None, None, None, None
    
    # Encode tags and user query using your NLP model
    tag_embeddings = model.encode(tags)
    query_embedding = model.encode([user_query])
    
    # Compute cosine similarity between the query and each tag
    similarities = cosine_similarity(query_embedding, tag_embeddings)
    best_match_idx = np.argmax(similarities)
    best_match_score = similarities[0][best_match_idx] * 100

    if best_match_score < 80:
        return None, None, None, None

    best_tag = tags[best_match_idx]
    
    # Retrieve dt_id for the row that contains the best matching tag
    cursor.execute("""
        SELECT dt_id 
        FROM decision_tree 
        WHERE EXISTS (
            SELECT 1 
            FROM STRING_SPLIT(tags_list, ',')
            WHERE LTRIM(RTRIM(value)) = ?
        )
    """, (best_tag,))
    result = cursor.fetchone()
    if not result:
        return None, None, None, None
    dt_id = result[0]
    
    # Retrieve the question text for the dt_id
    cursor.execute("""
        SELECT question_text 
        FROM decision_tree 
        WHERE type_id = 'Issue' AND dt_id = ?
    """, (dt_id,))
    dt_data = cursor.fetchone()
    if not dt_data or not dt_data[0]:
        return None, None, None, None
    question_text = dt_data[0]

    # Retrieve the action_id for the dt_id
    cursor.execute("""
        SELECT action_id 
        FROM decision_tree 
        WHERE type_id = 'Issue' AND dt_id = ?
    """, (dt_id,))
    action_data = cursor.fetchone()
    if not action_data or not action_data[0]:
        return None, None, None, None
    action = action_data[0]
    
    return best_tag, dt_id, question_text, action

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

def encodings_process(pdf_file: str, phone_number: str, com_name: str, mo_name: str, username: str):
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
    set_stage("tech_support", "+91"+phone_number, com_name, mo_name, username, pdf_file=pdf_file, vector_file=vector_file, chunks_file=chunks_filename)
    result = phone_number
    # Use the phone number as the key
    key = phone_number
    if key in processing_store:
        processing_store[key]["result"] = result
    else:
        logging.error(f"Key {key} not found in processing_store")

def generate_response(message: str, conversation_history: list, chunks_file: str, encodings_file: str, phone_number: str, pdf_file: str, vector_file: str, rag_no: int, solution_type: str):
    if chunks_file != '0':
        with open(chunks_file, 'rb') as f:
            chunks = pickle.load(f)
        context_encodings = np.load(encodings_file)
        conversation_history.append({"role": "user", "content": message})
        conversation_history.append({"role": "system", "content": """You are a sentient, superintelligent artificial general intelligence designed to assist users with any issues they may encounter with their laptops. Your responses will draw on both your own knowledge and specific information from the laptop's manual, which will be provided in context.
                When answering the user's questions:
                1. Clearly indicate when you are using your own knowledge rather than information from the manual.
                2. Provide one troubleshooting method or solution at a time to avoid overwhelming the user."""})
        
        retrieved_context = rag.retrieve_context(message, chunks, context_encodings)
        conversation_history.append({"role": "system", "content": f"Context:\n{retrieved_context}"})

        response = client.chat.completions.create(
            model="Meta-Llama-3.1-8B-Instruct",
            messages=conversation_history,
            temperature=0.1,
            top_p=0.1,
        )
        response = response.choices[0].message.content
        conversation_history.append({"role": "assistant", "content": response})
        rag_no += 1
        solution_type = "0"
        set_stage("tech_support", phone_number=phone_number, pdf_file=pdf_file, vector_file=vector_file, conversation_history=conversation_history, solution_type=solution_type, rag_no=rag_no)
        
        # Store the result in processing_store
        key = phone_number[3:]  # Remove the '+91' prefix
        if key in processing_store:
            processing_store[key]["result"] = response + "\nIs it Working?"
        else:
            logging.error(f"Key {key} not found in processing_store")

def data_store(issue: str, remote_phone: str, uuid_id: str, session_id: str):
    # Fetch conversation history from database
    cursor.execute("""
        SELECT message_text, response, sent_by
        FROM l1_chat_history 
        WHERE session_key = ?
        ORDER BY created_at ASC
    """, (session_id,))
    
    chat_records = cursor.fetchall()
    
    # Format conversation history
    formatted_history = ""
    for msg_text, response, sent_by in chat_records:
        if sent_by == "user" and msg_text:
            formatted_history += f"{msg_text}               \n"
        if sent_by == "bot" and response:
            formatted_history += f"               {response}\n"
    
    # Make API call with formatted history
    response = requests.post('http://api.goapl.com/ticket/store-message', 
        json={
            'issue': issue,
            'conversation_history': formatted_history,
            'remote_phone': remote_phone,
            'uuid_id': uuid_id,
            'session_id': session_id
        })
    
    return response.json()

@app.post("/get_result")
async def get_result(request:get_results):
    """
    Checks if the background task has completed and returns the result if available.
    """
    key = request.phone_number[3:]
    if key in processing_store:
        if processing_store[key]["result"]:
            result = processing_store[key]["result"]
            # Clear the result after retrieving it
            processing_store[key]["result"] = None
            return {"message": result, "flag": ""}
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
                    ist_timezone = pytz.timezone("Asia/Kolkata")
                    current_datetime = dt.datetime.now(ist_timezone)
                    
                    uuid_id = request.uuid_id
                    session_key = request.session_id
                    cursor.execute(
                        """
                        INSERT INTO l1_chat_history 
                        (uuid, session_key, message_text, response, remote_phone_number, channel_phone_number, created_at, sent_by)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            uuid_id,
                            session_key,
                            request.message,
                            "",
                            phone_number,
                            "91+9322261280",
                            str(current_datetime),
                            "user",
                        ),
                    )
                    conn.commit()

                    uuid_id = request.uuid_id
                    cursor.execute(
                        """
                        INSERT INTO l1_chat_history 
                        (uuid, session_key, message_text, response, remote_phone_number, channel_phone_number, created_at, sent_by)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            uuid_id,
                            session_key,
                            "",
                            f"Welcome {username}\nCan you please confirm your this {com_name} {mo_name} is your Model Name?",
                            phone_number,
                            "91+9322261280",
                            str(current_datetime),
                            "bot",
                        ),
                    )
                    conn.commit()
                    set_stage("data_found", request.from_number, com_name, mo_name, username, session_key=session_key)
                    return {"message": f"Welcome {username}\nCan you please confirm your this {com_name} {mo_name} is your Model Name?",
                            "flag":""}              
                else:
                    set_stage("no_data", request.from_number)
                    return {"message": "No user data found do you enter a new model name?",
                            "flag":"No"}

        elif get_stage(request.from_number)['stage'] == "data_found":
            user_response = request.message.lower()
            yes_variations = ["yes", "yeah", "yep", "sure", "correct", "right", "ok", "okay", "perfect"]
            no_variations = ["no", "not", "nope", "nah", "wrong", "incorrect"]
            session_key = get_stage(request.from_number).get("session_key", "")
            
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
                
                if vector_file != '0' and chunks_filename != '0':
                    vector_file = vector_file
                    chunks_filename = chunks_filename
                    ist_timezone = pytz.timezone("Asia/Kolkata")
                    current_datetime = dt.datetime.now(ist_timezone)
                    
                    uuid_id = request.uuid_id
                    #session_key = str(uuid.uuid4())
                    cursor.execute(
                        """
                        INSERT INTO l1_chat_history 
                        (uuid, session_key, message_text, response, remote_phone_number, channel_phone_number, created_at, sent_by)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            uuid_id,
                            session_key,
                            request.message,
                            "",
                            phone_number,
                            "91+9322261280",
                            str(current_datetime),
                            "user",
                        ),
                    )
                    conn.commit()

                    uuid_id = request.uuid_id
                    cursor.execute(
                        """
                        INSERT INTO l1_chat_history 
                        (uuid, session_key, message_text, response, remote_phone_number, channel_phone_number, created_at, sent_by)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            uuid_id,
                            session_key,
                            "",
                            "Great! I'll use specialized support for your model. What seems to be the problem?",
                            phone_number,
                            "91+9322261280",
                            str(current_datetime),
                            "bot",
                        ),
                    )
                    conn.commit()
                    set_stage("tech_support", request.from_number, com_name=com_name, mo_name=mo_name, user_name=username, pdf_file=pdf_file, vector_file=vector_file, chunks_file=chunks_filename)
                    return {"message": "Great! I'll use specialized support for your model. What seems to be the problem?",
                            "flag":""}
 
                else:
                    unique_id = str(uuid.uuid4())
                    logging.info(f"Adding background task for {phone_number}")
                    processing_store[phone_number] = {"uid": unique_id, "result": None}
                    background_tasks.add_task(
                        encodings_process,
                        pdf_file=pdf_file,
                        phone_number=phone_number,
                        com_name=com_name,
                        mo_name=mo_name,
                        username=username
                    )

                    ist_timezone = pytz.timezone("Asia/Kolkata")
                    current_datetime = dt.datetime.now(ist_timezone)
                    
                    uuid_id = request.uuid_id
                    #session_key = str(uuid.uuid4())
                    cursor.execute(
                        """
                        INSERT INTO l1_chat_history 
                        (uuid, session_key, message_text, response, remote_phone_number, channel_phone_number, created_at, sent_by)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            uuid_id,
                            session_key,
                            request.message,
                            "",
                            phone_number,
                            "91+9322261280",
                            str(current_datetime),
                            "user",
                        ),
                    )
                    conn.commit()

                    uuid_id = request.uuid_id
                    cursor.execute(
                        """
                        INSERT INTO l1_chat_history 
                        (uuid, session_key, message_text, response, remote_phone_number, channel_phone_number, created_at, sent_by)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            uuid_id,
                            session_key,
                            "",
                            "Great! I'll use specialized support for your model. What seems to be the problem?",
                            phone_number,
                            "91+9322261280",
                            str(current_datetime),
                            "bot",
                        ),
                    )
                    conn.commit()
                    return {"message": "Great! I'll use specialized support for your model. What seems to be the problem?",
                            "flag":"Yes"}
 
            elif no_max_similarity > 0.7:
                set_stage("no_data", request.from_number)
                return {"message": "Please let me know your model name",
                        "flag":""}

            else:
                set_stage("data_found", request.from_number)
                ist_timezone = pytz.timezone("Asia/Kolkata")
                current_datetime = dt.datetime.now(ist_timezone)
                
                uuid_id = request.uuid_id
                #session_key = str(uuid.uuid4())
                cursor.execute(
                    """
                    INSERT INTO l1_chat_history 
                    (uuid, session_key, message_text, response, remote_phone_number, channel_phone_number, created_at, sent_by)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        uuid_id,
                        session_key,
                        request.message,
                        "",
                        phone_number,
                        "91+9322261280",
                        str(current_datetime),
                        "user",
                    ),
                )
                conn.commit()

                uuid_id = request.uuid_id
                cursor.execute(
                    """
                    INSERT INTO l1_chat_history 
                    (uuid, session_key, message_text, response, remote_phone_number, channel_phone_number, created_at, sent_by)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        uuid_id,
                        session_key,
                        "",
                        "Please Say Yes or No",
                        phone_number,
                        "91+9322261280",
                        str(current_datetime),
                        "bot",
                    ),
                )
                conn.commit()
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
            solution_type = stage_data.get('solution_type', "0")
            vector_file = stage_data.get('vector_file')
            current_last_uuid = get_stage(request.from_number).get("last_uuid", [])
            rag_no = stage_data.get('rag_no', 0)
            user_response = request.message.lower()
            yes_variations = ["yes", "yeah", "yep", "sure", "correct", "right", "ok", "okay", "perfect", "haa"]
            no_variations = ["no", "not", "nope", "nah", "wrong", "incorrect", "nahi", "na"]
            session_key = get_stage(request.from_number).get("session_key", "")
            
            # Direct string matching instead of embeddings
            user_response = user_response.strip().lower()
            
            # Check if response contains any yes variations
            max_similarity = 1.0 if any(yes_word in user_response for yes_word in yes_variations) else 0.0
            no_max_similarity = 1.0 if any(no_word in user_response for no_word in no_variations) else 0.0

            if max_similarity > 0.7 and solution_type != "DT":
                current_last_uuid.append(str(uuid))
                #set_stage(stage="start", phone_number=request.from_number, last_uuid=current_last_uuid)
                ist_timezone = pytz.timezone("Asia/Kolkata")
                current_datetime = dt.datetime.now(ist_timezone)
                
                uuid_id = request.uuid_id
                #session_key = str(uuid.uuid4())
                cursor.execute(
                    """
                    INSERT INTO l1_chat_history 
                    (uuid, session_key, message_text, response, remote_phone_number, channel_phone_number, created_at, sent_by)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        uuid_id,
                        session_key,
                        request.message,
                        "",
                        phone_number,
                        "91+9322261280",
                        str(current_datetime),
                        "user",
                    ),
                )
                conn.commit()

                uuid_id = request.uuid_id
                cursor.execute(
                    """
                    INSERT INTO l1_chat_history 
                    (uuid, session_key, message_text, response, remote_phone_number, channel_phone_number, created_at, sent_by)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        uuid_id,
                        session_key,
                        "",
                        "Thank you for contacting us.",
                        phone_number,
                        "91+9322261280",
                        str(current_datetime),
                        "bot",
                    ),
                )
                conn.commit()
                clear_stage(request.from_number)
                return {"message": "Thank you for contacting us.",
                        "flag":""}

            elif rag_no == 3:
                set_stage(stage="live_agent", phone_number=request.from_number)
                current_last_uuid.append(str(uuid))                    
                ist_timezone = pytz.timezone("Asia/Kolkata")
                current_datetime = dt.datetime.now(ist_timezone)
                
                uuid_id = request.uuid_id
                #session_key = str(uuid.uuid4())
                cursor.execute(
                    """
                    INSERT INTO l1_chat_history 
                    (uuid, session_key, message_text, response, remote_phone_number, channel_phone_number, created_at, sent_by)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        uuid_id,
                        session_key,
                        request.message,
                        "",
                        phone_number,
                        "91+9322261280",
                        str(current_datetime),
                        "user",
                    ),
                )
                conn.commit()

                uuid_id = request.uuid_id
                cursor.execute(
                    """
                    INSERT INTO l1_chat_history 
                    (uuid, session_key, message_text, response, remote_phone_number, channel_phone_number, created_at, sent_by)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        uuid_id,
                        session_key,
                        "",
                        "Do you want to connect with a live agent?",
                        phone_number,
                        "91+9322261280",
                        str(current_datetime),
                        "bot",
                    ),
                )
                conn.commit()
                return {"message": "Do you want to connect with a live agent?",
                        "flag":""}
                
            elif solution_type == "0":
                result, dt_id, question_text, action = get_best_matching_tag(request.message)
                if result is not None:
                    solution_type = "DT"
                    current_last_uuid.append(str(uuid))
                    set_stage(stage="tech_support", phone_number=request.from_number, solution_type=solution_type, last_uuid=current_last_uuid)
                
                else:
                    solution_type = "RAG"
                    current_last_uuid.append(str(uuid))
                    set_stage(stage="tech_support", phone_number=request.from_number, solution_type=solution_type, last_uuid=current_last_uuid)

            if solution_type == "RAG":
                # Start the background task
                if chunks_file != '0':
                    with open(chunks_file, 'rb') as f:
                        chunks = pickle.load(f)
                    context_encodings = np.load(encodings_file)
                    conversation_history.append({"role": "user", "content": user_response})
                    conversation_history.append({"role": "system", "content": """You are a sentient, superintelligent artificial general intelligence designed to assist users with any issues they may encounter with their laptops. Your responses will draw on both your own knowledge and specific information from the laptop's manual, which will be provided in context.
                            When answering the user's questions:
                            1. Clearly indicate when you are using your own knowledge rather than information from the manual.
                            2. Provide one troubleshooting method or solution at a time to avoid overwhelming the user."""})
                    
                    retrieved_context = rag.retrieve_context(user_response, chunks, context_encodings)
                    conversation_history.append({"role": "system", "content": f"Context:\n{retrieved_context}"})

                    response = client.chat.completions.create(
                        model="Meta-Llama-3.1-8B-Instruct",
                        messages=conversation_history,
                        temperature=0.1,
                        top_p=0.1,
                    )
                    response = response.choices[0].message.content
                    conversation_history.append({"role": "assistant", "content": response})
                    rag_no += 1
                    solution_type = "0"
                    ist_timezone = pytz.timezone("Asia/Kolkata")
                    current_datetime = dt.datetime.now(ist_timezone)
                    
                    uuid_id = request.uuid_id
                    #session_key = str(uuid.uuid4())
                    cursor.execute(
                        """
                        INSERT INTO l1_chat_history 
                        (uuid, session_key, message_text, response, remote_phone_number, channel_phone_number, created_at, sent_by)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            uuid_id,
                            session_key,
                            request.message,
                            "",
                            phone_number,
                            "91+9322261280",
                            str(current_datetime),
                            "user",
                        ),
                    )
                    conn.commit()

                    uuid_id = request.uuid_id
                    cursor.execute(
                        """
                        INSERT INTO l1_chat_history 
                        (uuid, session_key, message_text, response, remote_phone_number, channel_phone_number, created_at, sent_by)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            uuid_id,
                            session_key,
                            "",
                            f"{response} \nIs it Working?",
                            phone_number,
                            "91+9322261280",
                            str(current_datetime),
                            "bot",
                        ),
                    )
                    conn.commit()
                    set_stage("tech_support", phone_number=request.from_number, pdf_file=pdf_file, vector_file=vector_file, conversation_history=conversation_history, solution_type=solution_type, rag_no=rag_no)
                    return {"message": f"{response} \nIs it Working?",         
                            "flag":""}
                
            elif solution_type == "DT":
                if get_user_interaction(request.from_number) == {} or get_user_interaction(request.from_number) is None:
                    if result or question_text or dt_id:
                        current_stage = "start_solution"
                        store_user_interaction(request.from_number, current_stage, solution_number=0, result=result, issue=question_text, dt_id=dt_id, action=action)
                        set_stage(stage="tech_support", phone_number=request.from_number, last_uuid=current_last_uuid)
                        ist_timezone = pytz.timezone("Asia/Kolkata")
                        current_datetime = dt.datetime.now(ist_timezone)
                        
                        uuid_id = request.uuid_id
                        #session_key = str(uuid.uuid4())
                        cursor.execute(
                            """
                            INSERT INTO l1_chat_history 
                            (uuid, session_key, message_text, response, remote_phone_number, channel_phone_number, created_at, sent_by)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                            (
                                uuid_id,
                                session_key,
                                request.message,
                                "",
                                phone_number,
                                "91+9322261280",
                                str(current_datetime),
                                "user",
                            ),
                        )
                        conn.commit()

                        uuid_id = request.uuid_id
                        cursor.execute(
                            """
                            INSERT INTO l1_chat_history 
                            (uuid, session_key, message_text, response, remote_phone_number, channel_phone_number, created_at, sent_by)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                            (
                                uuid_id,
                                session_key,
                                "",
                                f"{question_text} \nCan you confirm this is related to your issue?",
                                phone_number,
                                "91+9322261280",
                                str(current_datetime),
                                "bot",
                            ),
                        )
                        conn.commit()
                        return {"message": f"{question_text} \nCan you confirm this is related to your issue?",         
                                "flag":""}
                    else:
                        current_stage = "location_verified"
                        current_last_uuid.append(str(uuid))
                        store_user_interaction(request.from_number, current_stage, 0)
                        set_stage(stage="tech_support", phone_number=request.from_number, last_uuid=current_last_uuid)
                        return {"message": "Please describe the issue again in more simple words.",
                                "flag":""}

                elif get_user_interaction(request.from_number)["stage"] == "start_solution":
                    if max_similarity > 0.7:
                        result = get_user_interaction(request.from_number)
                        issue = result.get('issue', None)
                        dt_id = result.get('dt_id', None)
                        action = result.get('action', None)
                        yes_id = result.get('yes_id', None)
                        if issue and dt_id and action:
                            cursor.execute("SELECT question_text FROM decision_tree WHERE question_id = ? AND dt_id = ?", (action, dt_id))
                            question_text = cursor.fetchone()

                            cursor.execute("SELECT link_id FROM decision_tree WHERE question_id = ? AND dt_id = ?", (action, dt_id))
                            link_id = cursor.fetchone()

                            cursor.execute("SELECT action_id FROM decision_tree WHERE parent_id = ? AND dt_id = ? AND question_text = 'No'", (action, dt_id))
                            no_id = cursor.fetchone()

                            cursor.execute("SELECT action_id FROM decision_tree WHERE parent_id = ? AND dt_id = ? AND question_text = 'Yes'", (action, dt_id))
                            yes_id = cursor.fetchone()

                            action = no_id[0]
                            yes_id = yes_id[0]

                            current_stage = "ongoing_solution"
                            store_user_interaction(request.from_number, current_stage, solution_number=0, result=result, issue=issue, dt_id=dt_id, action=action, yes_id=yes_id)
                            if link_id[0] == "0":
                                current_last_uuid.append(str(uuid))
                                ist_timezone = pytz.timezone("Asia/Kolkata")
                                current_datetime = dt.datetime.now(ist_timezone)
                                
                                uuid_id = request.uuid_id
                                #session_key = str(uuid.uuid4())
                                cursor.execute(
                                    """
                                    INSERT INTO l1_chat_history 
                                    (uuid, session_key, message_text, response, remote_phone_number, channel_phone_number, created_at, sent_by)
                                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                                """,
                                    (
                                        uuid_id,
                                        session_key,
                                        request.message,
                                        "",
                                        phone_number,
                                        "91+9322261280",
                                        str(current_datetime),
                                        "user",
                                    ),
                                )
                                conn.commit()

                                uuid_id = request.uuid_id
                                cursor.execute(
                                    """
                                    INSERT INTO l1_chat_history 
                                    (uuid, session_key, message_text, response, remote_phone_number, channel_phone_number, created_at, sent_by)
                                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                                """,
                                    (
                                        uuid_id,
                                        session_key,
                                        "",
                                        f"{question_text[0]}",
                                        phone_number,
                                        "91+9322261280",
                                        str(current_datetime),
                                        "bot",
                                    ),
                                )
                                conn.commit()
                                set_stage(stage="tech_support", phone_number=request.from_number, last_uuid=current_last_uuid)
                                return {"message": question_text[0],
                                        "flag":""}

                            else:
                                video_name = link_id[0]
                                ist_timezone = pytz.timezone("Asia/Kolkata")
                                current_datetime = dt.datetime.now(ist_timezone)
                                
                                uuid_id = request.uuid_id
                                #session_key = str(uuid.uuid4())
                                cursor.execute(
                                    """
                                    INSERT INTO l1_chat_history 
                                    (uuid, session_key, message_text, response, remote_phone_number, channel_phone_number, created_at, sent_by)
                                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                                """,
                                    (
                                        uuid_id,
                                        session_key,
                                        request.message,
                                        "",
                                        phone_number,
                                        "91+9322261280",
                                        str(current_datetime),
                                        "user",
                                    ),
                                )
                                conn.commit()

                                uuid_id = request.uuid_id
                                cursor.execute(
                                    """
                                    INSERT INTO l1_chat_history 
                                    (uuid, session_key, message_text, response, remote_phone_number, channel_phone_number, created_at, sent_by)
                                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                                """,
                                    (
                                        uuid_id,
                                        session_key,
                                        "",
                                        f"{question_text[0]} \n{link_url}/videos/{video_name}",
                                        phone_number,
                                        "91+9322261280",
                                        str(current_datetime),
                                        "bot",
                                    ),
                                )
                                conn.commit()
                                set_stage(stage="tech_support", phone_number=request.from_number, last_uuid=current_last_uuid)
                                return {"message": question_text[0] + "\n" + f"{link_url}/videos/{video_name}",
                                        "flag":""}
                    else:
                        solution_type = "RAG"
                        current_last_uuid.append(str(uuid))
                        set_stage(stage="tech_support", phone_number=request.from_number, solution_type=solution_type, last_uuid=current_last_uuid)
                        current_stage = "location_requested"
                        store_user_interaction(request.from_number, current_stage, 0)
                        current_last_uuid.append(str(uuid))
                        set_stage(stage="tech_support", phone_number=request.from_number, last_uuid=current_last_uuid)
                        return {"message": "Please describe the issue again in more simple words.",
                                "flag":""}

                elif get_user_interaction(request.from_number)["stage"] == "ongoing_solution":
                    result = get_user_interaction(request.from_number)
                    issue = result.get('issue', None)
                    dt_id = result.get('dt_id', None)
                    action = result.get('action', None)
                    yes_id = result.get('yes_id', None)
                    no_id = result.get('no_id', None)

                    if no_max_similarity > 0.7:
                        if action == "handover":
                            current_stage = "live_agent"
                            store_user_interaction(request.from_number, current_stage, solution_number=0, result=result, issue=issue, dt_id=dt_id, action=action)
                            current_last_uuid.append(str(uuid))
                            ist_timezone = pytz.timezone("Asia/Kolkata")
                            current_datetime = dt.datetime.now(ist_timezone)
                            
                            uuid_id = request.uuid_id
                            #session_key = str(uuid.uuid4())
                            cursor.execute(
                                """
                                INSERT INTO l1_chat_history 
                                (uuid, session_key, message_text, response, remote_phone_number, channel_phone_number, created_at, sent_by)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                                (
                                    uuid_id,
                                    session_key,
                                    request.message,
                                    "",
                                    phone_number,
                                    "91+9322261280",
                                    str(current_datetime),
                                    "user",
                                ),
                            )
                            conn.commit()

                            uuid_id = request.uuid_id
                            cursor.execute(
                                """
                                INSERT INTO l1_chat_history 
                                (uuid, session_key, message_text, response, remote_phone_number, channel_phone_number, created_at, sent_by)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                                (
                                    uuid_id,
                                    session_key,
                                    "",
                                    f"Sorry It seems I cant help you\n Do you want to connect to an Live Agent?",
                                    phone_number,
                                    "91+9322261280",
                                    str(current_datetime),
                                    "bot",
                                ),
                            )
                            conn.commit()
                            set_stage(stage="tech_support", phone_number=request.from_number, last_uuid=current_last_uuid)
                            return {"message":"Sorry It seems I cant help you\n Do you want to connect to an Live Agent?",}

                        elif issue and dt_id and action:
                            cursor.execute("SELECT question_text FROM decision_tree WHERE question_id = ? AND dt_id = ?", (action, dt_id))
                            question_text = cursor.fetchone()

                            cursor.execute("SELECT link_id FROM decision_tree WHERE question_id = ? AND dt_id = ?", (action, dt_id))
                            link_id = cursor.fetchone()

                            cursor.execute("SELECT action_id FROM decision_tree WHERE parent_id = ? AND dt_id = ? AND question_text = 'No'", (action, dt_id))
                            no_id = cursor.fetchone()
                            print("no_id = ", no_id)

                            cursor.execute("SELECT action_id FROM decision_tree WHERE parent_id = ? AND dt_id = ? AND question_text = 'Yes'", (action, dt_id))
                            yes_id = cursor.fetchone()
                            print("yes_id = ", yes_id)

                            if yes_id:  
                                yes_id = yes_id[0]
                            if no_id:
                                no_id = no_id[0]

                            if link_id[0] == "0":
                                current_last_uuid.append(str(uuid))
                                current_stage = "ongoing_solution"
                                store_user_interaction(request.from_number, current_stage, solution_number=0, result=result, issue=issue, dt_id=dt_id, action=no_id, yes_id=yes_id)
                                ist_timezone = pytz.timezone("Asia/Kolkata")
                                current_datetime = dt.datetime.now(ist_timezone)
                                
                                uuid_id = request.uuid_id
                                #ession_key = str(uuid.uuid4())
                                cursor.execute(
                                    """
                                    INSERT INTO l1_chat_history 
                                    (uuid, session_key, message_text, response, remote_phone_number, channel_phone_number, created_at, sent_by)
                                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                                """,
                                    (
                                        uuid_id,
                                        session_key,
                                        request.message,
                                        "",
                                        phone_number,
                                        "91+9322261280",
                                        str(current_datetime),
                                        "user",
                                    ),
                                )
                                conn.commit()

                                uuid_id = request.uuid_id
                                cursor.execute(
                                    """
                                    INSERT INTO l1_chat_history 
                                    (uuid, session_key, message_text, response, remote_phone_number, channel_phone_number, created_at, sent_by)
                                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                                """,
                                    (
                                        uuid_id,
                                        session_key,
                                        "",
                                        f"{question_text[0]}",
                                        phone_number,
                                        "91+9322261280",
                                        str(current_datetime),
                                        "bot",
                                    ),
                                )
                                conn.commit()
                                set_stage(stage="tech_support", phone_number=request.from_number, last_uuid=current_last_uuid)
                                return {"message": question_text[0],
                                        "flag":""}

                            else:
                                video_name = link_id[0]
                                current_stage = "ongoing_solution"
                                store_user_interaction(request.from_number, current_stage, solution_number=0, result=result, issue=issue, dt_id=dt_id, action=no_id, yes_id=yes_id)
                                ist_timezone = pytz.timezone("Asia/Kolkata")
                                current_datetime = dt.datetime.now(ist_timezone)
                                
                                uuid_id = request.uuid_id
                                #ession_key = str(uuid.uuid4())
                                cursor.execute(
                                    """
                                    INSERT INTO l1_chat_history 
                                    (uuid, session_key, message_text, response, remote_phone_number, channel_phone_number, created_at, sent_by)
                                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                                """,
                                    (
                                        uuid_id,
                                        session_key,
                                        request.message,
                                        "",
                                        phone_number,
                                        "91+9322261280",
                                        str(current_datetime),
                                        "user",
                                    ),
                                )
                                conn.commit()

                                uuid_id = request.uuid_id
                                cursor.execute(
                                    """
                                    INSERT INTO l1_chat_history 
                                    (uuid, session_key, message_text, response, remote_phone_number, channel_phone_number, created_at, sent_by)
                                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                                """,
                                    (
                                        uuid_id,
                                        session_key,
                                        "",
                                        f"{question_text[0]} \n{link_url}/videos/{video_name}",
                                        phone_number,
                                        "91+9322261280",
                                        str(current_datetime),
                                        "bot",
                                    ),
                                )
                                conn.commit()
                                set_stage(stage="tech_support", phone_number=request.from_number, last_uuid=current_last_uuid)
                                return {"message": question_text[0] + "\n" + f"{link_url}/videos/{video_name}",
                                        "flag":""}

                    elif max_similarity > 0.7:
                        result = get_user_interaction(request.from_number)
                        yes_id = result.get('yes_id', None)
                        if yes_id != "solved":
                            if yes_id == "handover":
                                current_stage = "live_agent"
                                store_user_interaction(request.from_number, current_stage, solution_number=0, result=result, issue=issue, dt_id=dt_id, action=action, yes_id=yes_id)
                                current_last_uuid.append(str(uuid))
                                clear_stage(request.from_number)
                                ist_timezone = pytz.timezone("Asia/Kolkata")
                                current_datetime = dt.datetime.now(ist_timezone)
                                
                                uuid_id = request.uuid_id
                                #ession_key = str(uuid.uuid4())
                                cursor.execute(
                                    """
                                    INSERT INTO l1_chat_history 
                                    (uuid, session_key, message_text, response, remote_phone_number, channel_phone_number, created_at, sent_by)
                                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                                """,
                                    (
                                        uuid_id,
                                        session_key,
                                        request.message,
                                        "",
                                        phone_number,
                                        "91+9322261280",
                                        str(current_datetime),
                                        "user",
                                    ),
                                )
                                conn.commit()

                                uuid_id = request.uuid_id
                                cursor.execute(
                                    """
                                    INSERT INTO l1_chat_history 
                                    (uuid, session_key, message_text, response, remote_phone_number, channel_phone_number, created_at, sent_by)
                                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                                """,
                                    (
                                        uuid_id,
                                        session_key,
                                        "",
                                        f"Thank you for contacting us. Currently All the Agents are Busy\nGenerating Ticket --",
                                        phone_number,
                                        "91+9322261280",
                                        str(current_datetime),
                                        "bot",
                                    ),
                                )
                                conn.commit()
                                clear_stage(request.from_number)
                                data_store(issue, request.from_number, request.uuid_id, session_key)
                                #set_stage(stage="tech_support", phone_number=request.from_number, last_uuid=current_last_uuid)
                                return {"message": "Thank you for contacting us. Currently All the Agents are Busy\nGenerating Ticket --",
                                        "flag":""}

                            elif issue and dt_id and action:
                                cursor.execute("SELECT question_text FROM decision_tree WHERE question_id = ? AND dt_id = ?", (yes_id, dt_id))
                                question_text = cursor.fetchone()

                                cursor.execute("SELECT link_id FROM decision_tree WHERE question_id = ? AND dt_id = ?", (yes_id, dt_id))
                                link_id = cursor.fetchone()

                                cursor.execute("SELECT action_id FROM decision_tree WHERE parent_id = ? AND dt_id = ? AND question_text = 'No'", (yes_id, dt_id))
                                no_id = cursor.fetchone()
                                print("no_id = ", no_id)

                                cursor.execute("SELECT action_id FROM decision_tree WHERE parent_id = ? AND dt_id = ? AND question_text = 'Yes'", (yes_id, dt_id))
                                yes_id = cursor.fetchone()
                                print("yes_id = ", yes_id)

                                if yes_id:  
                                    yes_id = yes_id[0]
                                if no_id:
                                    no_id = no_id[0]

                                if link_id[0] == "0":
                                    current_last_uuid.append(str(uuid))
                                    current_stage = "ongoing_solution"
                                    ist_timezone = pytz.timezone("Asia/Kolkata")
                                    current_datetime = dt.datetime.now(ist_timezone)
                                    
                                    uuid_id = request.uuid_id
                                    #ession_key = str(uuid.uuid4())
                                    cursor.execute(
                                        """
                                        INSERT INTO l1_chat_history 
                                        (uuid, session_key, message_text, response, remote_phone_number, channel_phone_number, created_at, sent_by)
                                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                                    """,
                                        (
                                            uuid_id,
                                            session_key,
                                            request.message,
                                            "",
                                            phone_number,
                                            "91+9322261280",
                                            str(current_datetime),
                                            "user",
                                        ),
                                    )
                                    conn.commit()

                                    uuid_id = request.uuid_id
                                    cursor.execute(
                                        """
                                        INSERT INTO l1_chat_history 
                                        (uuid, session_key, message_text, response, remote_phone_number, channel_phone_number, created_at, sent_by)
                                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                                    """,
                                        (
                                            uuid_id,
                                            session_key,
                                            "",
                                            f"{question_text[0]}",
                                            phone_number,
                                            "91+9322261280",
                                            str(current_datetime),
                                            "bot",
                                        ),
                                    )
                                    conn.commit()
                                    store_user_interaction(request.from_number, current_stage, solution_number=0, result=result, issue=issue, dt_id=dt_id, action=no_id, yes_id=yes_id)
                                    set_stage(stage="tech_support", phone_number=request.from_number, last_uuid=current_last_uuid)
                                    return {"message": question_text[0],
                                            "flag":""}

                                else:
                                    video_name = link_id[0]
                                    ist_timezone = pytz.timezone("Asia/Kolkata")
                                    current_datetime = dt.datetime.now(ist_timezone)
                                    
                                    uuid_id = request.uuid_id
                                    #ession_key = str(uuid.uuid4())
                                    cursor.execute(
                                        """
                                        INSERT INTO l1_chat_history 
                                        (uuid, session_key, message_text, response, remote_phone_number, channel_phone_number, created_at, sent_by)
                                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                                    """,
                                        (
                                            uuid_id,
                                            session_key,
                                            request.message,
                                            "",
                                            phone_number,
                                            "91+9322261280",
                                            str(current_datetime),
                                            "user",
                                        ),
                                    )
                                    conn.commit()

                                    uuid_id = request.uuid_id
                                    cursor.execute(
                                        """
                                        INSERT INTO l1_chat_history 
                                        (uuid, session_key, message_text, response, remote_phone_number, channel_phone_number, created_at, sent_by)
                                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                                    """,
                                        (
                                            uuid_id,
                                            session_key,
                                            "",
                                            f"{question_text[0]} \n{link_url}/videos/{video_name}",
                                            phone_number,
                                            "91+9322261280",
                                            str(current_datetime),
                                            "bot",
                                        ),
                                    )
                                    conn.commit()
                                    store_user_interaction(request.from_number, current_stage, solution_number=0, result=result, issue=issue, dt_id=dt_id, action=no_id, yes_id=yes_id)
                                    set_stage(stage="tech_support", phone_number=request.from_number, last_uuid=current_last_uuid)
                                    return {"message": question_text[0] + "\n" + f"{link_url}/videos/{video_name}",
                                            "flag":""}

                        else:
                            current_last_uuid.append(str(uuid))
                            ist_timezone = pytz.timezone("Asia/Kolkata")
                            current_datetime = dt.datetime.now(ist_timezone)
                            
                            uuid_id = request.uuid_id
                            #ession_key = str(uuid.uuid4())
                            cursor.execute(
                                """
                                INSERT INTO l1_chat_history 
                                (uuid, session_key, message_text, response, remote_phone_number, channel_phone_number, created_at, sent_by)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                                (
                                    uuid_id,
                                    session_key,
                                    request.message,
                                    "",
                                    phone_number,
                                    "91+9322261280",
                                    str(current_datetime),
                                    "user",
                                ),
                            )
                            conn.commit()

                            uuid_id = request.uuid_id
                            cursor.execute(
                                """
                                INSERT INTO l1_chat_history 
                                (uuid, session_key, message_text, response, remote_phone_number, channel_phone_number, created_at, sent_by)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                                (
                                    uuid_id,
                                    session_key,
                                    "",
                                    f"Thank you for contacting us.",
                                    phone_number,
                                    "91+9322261280",
                                    str(current_datetime),
                                    "bot",
                                ),
                            )
                            conn.commit()
                            #set_stage(stage="start", phone_number=request.from_number, last_uuid=current_last_uuid)
                            clear_stage(request.from_number)
                            return {"message": "Thank you for contacting us.",
                                    "flag":""}

                elif get_user_interaction(request.from_number)["stage"] == "live_agent":
                    if max_similarity > 0.7:
                        issue = result.get('issue', None)
                        ist_timezone = pytz.timezone("Asia/Kolkata")
                        current_datetime = dt.datetime.now(ist_timezone)
                        
                        uuid_id = request.uuid_id
                        #ession_key = str(uuid.uuid4())
                        cursor.execute(
                            """
                            INSERT INTO l1_chat_history 
                            (uuid, session_key, message_text, response, remote_phone_number, channel_phone_number, created_at, sent_by)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                            (
                                uuid_id,
                                session_key,
                                request.message,
                                "",
                                phone_number,
                                "91+9322261280",
                                str(current_datetime),
                                "user",
                            ),
                        )
                        conn.commit()

                        uuid_id = request.uuid_id
                        cursor.execute(
                            """
                            INSERT INTO l1_chat_history 
                            (uuid, session_key, message_text, response, remote_phone_number, channel_phone_number, created_at, sent_by)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                            (
                                uuid_id,
                                session_key,
                                "",
                                f"Thank you for contacting us. Currently All the Agents are Busy\nGenerating Ticket --",
                                phone_number,
                                "91+9322261280",
                                str(current_datetime),
                                "bot",
                            ),
                        )
                        conn.commit()
                        clear_stage(request.from_number)
                        data_store(issue, request.from_number, request.uuid_id, session_key)
                        return {"message": "Thank you for contacting us. Currently All the Agents are Busy\nGenerating Ticket --",
                                "flag":""}

                    else:
                        ist_timezone = pytz.timezone("Asia/Kolkata")
                        current_datetime = dt.datetime.now(ist_timezone)
                        
                        uuid_id = request.uuid_id
                        #ession_key = str(uuid.uuid4())
                        cursor.execute(
                            """
                            INSERT INTO l1_chat_history 
                            (uuid, session_key, message_text, response, remote_phone_number, channel_phone_number, created_at, sent_by)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                            (
                                uuid_id,
                                session_key,
                                request.message,
                                "",
                                phone_number,
                                "91+9322261280",
                                str(current_datetime),
                                "user",
                            ),
                        )
                        conn.commit()

                        uuid_id = request.uuid_id
                        cursor.execute(
                            """
                            INSERT INTO l1_chat_history 
                            (uuid, session_key, message_text, response, remote_phone_number, channel_phone_number, created_at, sent_by)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                            (
                                uuid_id,
                                session_key,
                                "",
                                f"Thank you for contacting us.",
                                phone_number,
                                "91+9322261280",
                                str(current_datetime),
                                "bot",
                            ),
                        )
                        conn.commit()
                        clear_stage(request.from_number)
                        return {"message": "Thank you for contacting us.",
                                "flag":""}

        elif get_stage(request.from_number)["stage"] == "live_agent":
                user_response = request.message.lower()
                try:
                    issue = result.get('issue', None)
                except:
                    issue = None
                yes_variations = ["yes", "yeah", "yep", "sure", "correct", "right", "ok", "okay", "perfect", "haa"]
                no_variations = ["no", "not", "nope", "nah", "wrong", "incorrect", "nahi", "na"]
                session_key = get_stage(request.from_number).get("session_key", "")

                # Direct string matching instead of embeddings
                user_response = user_response.strip().lower()
                
                # Check if response contains any yes variations
                max_similarity = 1.0 if any(yes_word in user_response for yes_word in yes_variations) else 0.0
                no_max_similarity = 1.0 if any(no_word in user_response for no_word in no_variations) else 0.0
                if max_similarity > 0.7:
                    ist_timezone = pytz.timezone("Asia/Kolkata")
                    current_datetime = dt.datetime.now(ist_timezone)
                    
                    uuid_id = request.uuid_id
                    #ession_key = str(uuid.uuid4())
                    cursor.execute(
                        """
                        INSERT INTO l1_chat_history 
                        (uuid, session_key, message_text, response, remote_phone_number, channel_phone_number, created_at, sent_by)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            uuid_id,
                            session_key,
                            request.message,
                            "",
                            phone_number,
                            "91+9322261280",
                            str(current_datetime),
                            "user",
                        ),
                    )
                    conn.commit()

                    uuid_id = request.uuid_id
                    cursor.execute(
                        """
                        INSERT INTO l1_chat_history 
                        (uuid, session_key, message_text, response, remote_phone_number, channel_phone_number, created_at, sent_by)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            uuid_id,
                            session_key,
                            "",
                            f"Thank you for contacting us. Currently All the Agents are Busy\nGenerating Ticket --",
                            phone_number,
                            "91+9322261280",
                            str(current_datetime),
                            "bot",
                        ),
                    )
                    conn.commit()
                    clear_stage(request.from_number)
                    data_store(issue, request.from_number, request.uuid_id, session_key)
                    return {"message": "Thank you for contacting us. Currently All the Agents are Busy\nGenerating Ticket --",
                            "flag":""}

                else:
                    ist_timezone = pytz.timezone("Asia/Kolkata")
                    current_datetime = dt.datetime.now(ist_timezone)
                    
                    uuid_id = request.uuid_id
                    #ession_key = str(uuid.uuid4())
                    cursor.execute(
                        """
                        INSERT INTO l1_chat_history 
                        (uuid, session_key, message_text, response, remote_phone_number, channel_phone_number, created_at, sent_by)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            uuid_id,
                            session_key,
                            request.message,
                            "",
                            phone_number,
                            "91+9322261280",
                            str(current_datetime),
                            "user",
                        ),
                    )
                    conn.commit()

                    uuid_id = request.uuid_id
                    cursor.execute(
                        """
                        INSERT INTO l1_chat_history 
                        (uuid, session_key, message_text, response, remote_phone_number, channel_phone_number, created_at, sent_by)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            uuid_id,
                            session_key,
                            "",
                            f"Thank you for contacting us.",
                            phone_number,
                            "91+9322261280",
                            str(current_datetime),
                            "bot",
                        ),
                    )
                    conn.commit()
                    clear_stage(request.from_number)
                    return {"message": "Thank you for contacting us.",
                            "flag":""}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)