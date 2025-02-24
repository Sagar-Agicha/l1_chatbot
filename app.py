import os
import json
import pickle
import requests
import datetime as dt
import numpy as np
import openai
import pyodbc
import psycopg2
from guardrails import Guard
from guardrails.hub import ToxicLanguage, ProfanityFree
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import rag_samba_continuous_function as rag
import pytz

client = openai.OpenAI(
    api_key= os.getenv("OPENAI_API_KEY"),
    base_url="https://api.sambanova.ai/v1",
)

server = os.getenv("SERVER")
database = os.getenv("DATABASE")
username = os.getenv("UID") 
password = os.getenv("PWD")

conn1 = psycopg2.connect(
    dbname="postgres",
    user="postgres",
    password="admin",
    host="localhost",
    port="5432"
)

link_url = "https://api.goapl.com"

cursor1 = conn1.cursor()
cursor1.execute("ROLLBACK")

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

model = SentenceTransformer('all-mpnet-base-v2')  # Best model for general-purpose semantic matching

whatsapp_number = "+919322261280"
api_key = "UAK6026c101-7d0a-4109-be98-fe7674f6f3ed"
receive_url = f"https://api.p.2chat.io/open/whatsapp/messages/{whatsapp_number}"

def send_msg(phone_number, txt):
    send_url = "https://api.p.2chat.io/open/whatsapp/send-message"
    headers = {"X-User-API-Key": api_key, "Content-Type": "application/json"}

    payload = {
        "from_number": whatsapp_number,  
        "to_number": phone_number,
        "text": txt,
    }

    payload_json = json.dumps(payload)

    requests.post(send_url, headers=headers, data=payload_json)

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

print("Program Started")
while True:
    headers = {"X-User-API-Key": api_key, "Content-Type": "application/json"}

    response = requests.get(receive_url, headers=headers)

    if response.status_code == 200:
        messages_data = response.json().get("messages", [])

        for message_data in messages_data:
            id = message_data.get("id", "0")
            uuid = message_data.get("uuid", "0")
            session_key = message_data.get("session_key", "0")
            message_text = message_data.get("message", {}).get("text", None)
            created_at = dt.datetime.strptime(
                message_data.get("created_at", ""), "%Y-%m-%dT%H:%M:%S"
            )
            remote_phone_number = message_data.get("remote_phone_number", None)
            sent_by = message_data.get("sent_by", None)

            if message_text is None:
                continue

            created_at = str(created_at)

            if sent_by == "user":
                user_validation = check_text_content(message_text)
                if user_validation['is_valid']:
                    if get_stage(remote_phone_number) == {} or get_stage(remote_phone_number)["stage"] == "start" and str(uuid) not in get_stage(remote_phone_number)["last_uuid"]:
                        phone_number = remote_phone_number[3:]
                        cursor.execute("""
                            SELECT user_name, com_name, mo_name
                            FROM l1_tree 
                            WHERE phone_number = ?
                        """, (phone_number,))
                        
                        result = cursor.fetchone()
                        if result:
                            username, com_name, mo_name = result
                            current_last_uuid = get_stage(remote_phone_number).get("last_uuid", [])
                            current_last_uuid.append(str(uuid))
                            set_stage(stage="data_found", phone_number=remote_phone_number, com_name=com_name, mo_name=mo_name, user_name=username, last_uuid=current_last_uuid, last_time=created_at)
                            send_msg(remote_phone_number, f"Welcome {username}\nCan you please confirm your this {com_name} {mo_name} is your Model Name?")
                      
                    elif get_stage(remote_phone_number)["stage"] == "data_found" and str(uuid) not in get_stage(remote_phone_number)["last_uuid"]:
                        if message_text.lower() == "yes":
                            phone_number = remote_phone_number[3:]
                            cursor.execute("""
                                SELECT user_name, com_name, mo_name, pdf_file, vector_file, chunks_file
                                FROM l1_tree 
                                WHERE phone_number = ?
                            """, (phone_number,))

                            result = cursor.fetchone()
                            if result:
                                user_name, com_name, mo_name, pdf_file, vector_file, chunks_filename = result
                        
                            if vector_file != '0' and chunks_filename != '0':
                                vector_file = vector_file
                                chunks_filename = chunks_filename
        
                            else:
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

                            current_stage_data = get_stage(remote_phone_number)
                            current_last_uuid = current_stage_data.get("last_uuid", [])
                            current_last_uuid.append(str(uuid))
                            set_stage(stage="tech_support", phone_number=remote_phone_number, com_name=com_name, mo_name=mo_name, user_name=user_name, pdf_file=pdf_file, vector_file=vector_file, chunks_file=chunks_filename, last_uuid=current_last_uuid, last_time=created_at)
                            send_msg(remote_phone_number, "Great! I'll use specialized support for your model. What seems to be the problem?")
                        else:
                            current_stage_data = get_stage(remote_phone_number)
                            current_last_uuid = current_stage_data.get("last_uuid", [])
                            current_last_uuid.append(str(uuid))
                            set_stage("no_data", remote_phone_number, last_uuid=current_last_uuid)
                            send_msg(remote_phone_number, "Please let me know your model name")

                    elif get_stage(remote_phone_number)["stage"] == "no_data" and str(uuid) not in get_stage(remote_phone_number)["last_uuid"]:
                        current_stage_data = get_stage(remote_phone_number)
                        current_last_uuid = current_stage_data.get("last_uuid", [])
                        current_last_uuid.append(str(uuid))
                        set_stage(stage="no_data", phone_number=remote_phone_number, last_uuid=current_last_uuid, last_time=created_at)
                        send_msg(remote_phone_number, "Please let me know your model name")

                    elif get_stage(remote_phone_number)["stage"] == "tech_support" and str(uuid) not in get_stage(remote_phone_number)["last_uuid"]:
                        stage_data = get_all_data(remote_phone_number)
                        pdf_file = stage_data.get('pdf_file')
                        encodings_file = stage_data.get('vector_file')
                        com_name = stage_data.get('com_name')
                        mo_name = stage_data.get('mo_name')
                        user_name = stage_data.get('user_name')
                        chunks_file = stage_data.get('chunks_file')
                        conversation_history = stage_data.get('conversation_history', [])
                        solution_type = stage_data.get('solution_type', "0")
                        vector_file = stage_data.get('vector_file')
                        current_last_uuid = get_stage(remote_phone_number).get("last_uuid", [])
                        rag_no = stage_data.get('rag_no', 0)

                        if message_text.lower() == "yes" and solution_type != "DT":
                            send_msg(remote_phone_number, "Thank you for contacting us.")
                            current_last_uuid.append(str(uuid))
                            set_stage(stage="start", phone_number=remote_phone_number, last_uuid=current_last_uuid, last_time=created_at)

                        elif rag_no == 3:
                            set_stage(stage="live_agent", phone_number=remote_phone_number)
                            current_last_uuid.append(str(uuid))
                            send_msg(remote_phone_number, "Do you want to connect with a live agent?", last_uuid=current_last_uuid, last_time=created_at)
                            
                        elif solution_type == "0":
                            result, dt_id, question_text, action = get_best_matching_tag(message_text)
                            if result is not None:
                                solution_type = "DT"
                                current_last_uuid.append(str(uuid))
                                set_stage(stage="tech_support", phone_number=remote_phone_number, solution_type=solution_type, last_uuid=current_last_uuid, last_time=created_at)
                            
                            else:
                                solution_type = "RAG"
                                current_last_uuid.append(str(uuid))
                                set_stage(stage="tech_support", phone_number=remote_phone_number, solution_type=solution_type, last_uuid=current_last_uuid, last_time=created_at)

                        if solution_type == "RAG":
                            if chunks_file != '0':
                                with open(chunks_file, 'rb') as f:
                                    chunks = pickle.load(f)
                            else:
                                chunks = []

                            # Load the saved encodings
                            context_encodings = np.load(encodings_file)
                            conversation_history.append({"role": "user", "content": message_text})
                            conversation_history.append({"role": "system", "content": """You are a sentient, superintelligent artificial general intelligence designed to assist users with any issues they may encounter with their laptops. Your responses will draw on both your own knowledge and specific information from the laptop's manual, which will be provided in context.
                                    When answering the user's questions:
                                    1. Clearly indicate when you are using your own knowledge rather than information from the manual.
                                    2. Provide one troubleshooting method or solution at a time to avoid overwhelming the user."""})
                            
                            retrieved_context = rag.retrieve_context(message_text, chunks, context_encodings)
                            conversation_history.append({"role": "system", "content": f"Context:\n{retrieved_context}"})

                            response = client.chat.completions.create(
                                model="Meta-Llama-3.1-8B-Instruct",
                                messages=conversation_history,
                                temperature=0.1,
                                top_p=0.1,
                            )
                            response = response.choices[0].message.content

                            current_stage_data = get_stage(remote_phone_number)
                            current_last_uuid = current_stage_data.get("last_uuid", [])
                            current_last_uuid.append(str(uuid))
                            conversation_history.append({"role": "assistant", "content": response})
                            rag_no += 1
                            solution_type = "0"
                            current_last_uuid.append(str(uuid))
                            set_stage("tech_support", phone_number=remote_phone_number, com_name=com_name, mo_name=mo_name, user_name=username, pdf_file=pdf_file, vector_file=vector_file, conversation_history=conversation_history, last_uuid=current_last_uuid, solution_type=solution_type, rag_no=rag_no, last_time=created_at)
                            send_msg(remote_phone_number, response + "\nIs it Working?")

                        elif solution_type == "DT":
                            if get_user_interaction(remote_phone_number) == {}:
                                if result or question_text or dt_id:
                                    send_msg(remote_phone_number, f"{question_text} \nCan you confirm this is related to your issue?")
                                    current_stage = "start_solution"
                                    current_last_uuid.append(str(uuid))
                                    store_user_interaction(remote_phone_number, current_stage, solution_number=0, result=result, issue=question_text, dt_id=dt_id, action=action)
                                    set_stage(stage="tech_support", phone_number=remote_phone_number, last_time=created_at, last_uuid=current_last_uuid)
                                else:
                                    send_msg(remote_phone_number, "Please describe the issue again in more simple words.")
                                    current_stage = "location_verified"
                                    current_last_uuid.append(str(uuid))
                                    store_user_interaction(remote_phone_number, current_stage, 0)
                                    set_stage(stage="tech_support", phone_number=remote_phone_number, last_time=created_at, last_uuid=current_last_uuid)

                            elif get_user_interaction(remote_phone_number)["stage"] == "start_solution":
                                if message_text.lower() == "yes":
                                    result = get_user_interaction(remote_phone_number)
                                    issue = result.get('issue', None)
                                    dt_id = result.get('dt_id', None)
                                    action = result.get('action', None)
                                    yes_id = result.get('yes_id', None)
                                    if issue and dt_id and action:
                                        cursor1.execute("SELECT question_text FROM decision_tree WHERE question_id = %s and dt_id = %s", (action, dt_id))
                                        question_text = cursor1.fetchone()

                                        cursor1.execute("SELECT link_id FROM decision_tree WHERE question_id = %s and dt_id = %s", (action, dt_id))
                                        link_id = cursor1.fetchone()

                                        cursor1.execute("SELECT action_id FROM decision_tree WHERE parent_id = %s and dt_id = %s and question_text = 'No'", (action, dt_id))
                                        no_id = cursor1.fetchone()

                                        cursor1.execute("SELECT action_id FROM decision_tree WHERE parent_id = %s and dt_id = %s and question_text = 'Yes'", (action, dt_id))
                                        yes_id = cursor1.fetchone()

                                        action = no_id[0]
                                        yes_id = yes_id[0]

                                        current_stage = "ongoing_solution"
                                        store_user_interaction(remote_phone_number, current_stage, solution_number=0, result=result, issue=issue, dt_id=dt_id, action=action, yes_id=yes_id)
                                        if link_id[0] == "0":
                                            send_msg(remote_phone_number, question_text[0])
                                            current_last_uuid.append(str(uuid))
                                            set_stage(stage="tech_support", phone_number=remote_phone_number, last_uuid=current_last_uuid, last_time=created_at)

                                        else:
                                            video_name = link_id[0]
                                            send_msg(remote_phone_number, question_text[0])
                                            send_msg(remote_phone_number, f"{link_url}/videos/{video_name}")
                                            current_last_uuid.append(str(uuid))
                                            set_stage(stage="tech_support", phone_number=remote_phone_number, last_uuid=current_last_uuid, last_time=created_at)
                                else:
                                    solution_type = "RAG"
                                    current_last_uuid.append(str(uuid))
                                    set_stage(stage="tech_support", phone_number=remote_phone_number, solution_type=solution_type, last_uuid=current_last_uuid, last_time=created_at)
                                    current_stage = "location_requested"
                                    store_user_interaction(remote_phone_number, current_stage, 0)
                                    current_last_uuid.append(str(uuid))
                                    set_stage(stage="tech_support", phone_number=remote_phone_number, last_uuid=current_last_uuid, last_time=created_at)

                            elif get_user_interaction(remote_phone_number)["stage"] == "ongoing_solution":
                                result = get_user_interaction(remote_phone_number)
                                issue = result.get('issue', None)
                                dt_id = result.get('dt_id', None)
                                action = result.get('action', None)
                                yes_id = result.get('yes_id', None)
                                no_id = result.get('no_id', None)

                                if message_text.lower() == "no":
                                    if action == "handover":
                                        current_stage = "live_agent"
                                        send_msg(remote_phone_number, "Do you want to connect with a live agent?")
                                        store_user_interaction(remote_phone_number, current_stage, solution_number=0, result=result, issue=issue, dt_id=dt_id, action=action)
                                        current_last_uuid.append(str(uuid))
                                        set_stage(stage="tech_support", phone_number=remote_phone_number, last_uuid=current_last_uuid, last_time=created_at)

                                    elif issue and dt_id and action:
                                        cursor1.execute("SELECT question_text FROM decision_tree WHERE question_id = %s and dt_id = %s", (action, dt_id))
                                        question_text = cursor1.fetchone()

                                        cursor1.execute("SELECT link_id FROM decision_tree WHERE question_id = %s and dt_id = %s", (action, dt_id))
                                        link_id = cursor1.fetchone()

                                        cursor1.execute("SELECT action_id FROM decision_tree WHERE parent_id = %s and dt_id = %s and question_text = 'No'", (action, dt_id))
                                        no_id = cursor1.fetchone()

                                        cursor1.execute("SELECT action_id FROM decision_tree WHERE parent_id = %s and dt_id = %s and question_text = 'Yes'", (action, dt_id))
                                        yes_id = cursor1.fetchone()

                                        if yes_id:
                                            yes_id = yes_id[0]
                                        if no_id:
                                            no_id = no_id[0]

                                        if link_id[0] == "0":
                                            send_msg(remote_phone_number, question_text[0])
                                            current_last_uuid.append(str(uuid))
                                            set_stage(stage="tech_support", phone_number=remote_phone_number, last_uuid=current_last_uuid, last_time=created_at)
                                        else:
                                            video_name = link_id[0]
                                            send_msg(remote_phone_number, question_text[0])
                                            send_msg(remote_phone_number, f"{link_url}/videos/{video_name}")
                                            current_last_uuid.append(str(uuid))
                                            set_stage(stage="tech_support", phone_number=remote_phone_number, last_uuid=current_last_uuid, last_time=created_at)

                                        current_stage = "ongoing_solution"
                                        store_user_interaction(remote_phone_number, current_stage, solution_number=0, result=result, issue=issue, dt_id=dt_id, action=no_id, yes_id=yes_id)

                                elif message_text.lower() == "yes":
                                    result = get_user_interaction(remote_phone_number)
                                    yes_id = result.get('yes_id', None)
                                    if yes_id != "solved":
                                        if yes_id == "handover":
                                            current_stage = "live_agent"
                                            send_msg(remote_phone_number, "Do you want to connect with a live agent?")
                                            store_user_interaction(remote_phone_number, current_stage, solution_number=0, result=result, issue=issue, dt_id=dt_id, action=action, yes_id=yes_id)
                                            current_last_uuid.append(str(uuid))
                                            set_stage(stage="tech_support", phone_number=remote_phone_number, last_uuid=current_last_uuid, last_time=created_at)

                                        elif issue and dt_id and action:
                                            cursor1.execute("SELECT question_text FROM decision_tree WHERE question_id = %s and dt_id = %s", (yes_id, dt_id))
                                            question_text = cursor1.fetchone()

                                            cursor1.execute("SELECT link_id FROM decision_tree WHERE question_id = %s and dt_id = %s", (yes_id, dt_id))
                                            link_id = cursor1.fetchone()

                                            cursor1.execute("SELECT action_id FROM decision_tree WHERE parent_id = %s and dt_id = %s and question_text = 'No'", (yes_id, dt_id))
                                            no_id = cursor1.fetchone()
                                            print("no_id = ", no_id)

                                            cursor1.execute("SELECT action_id FROM decision_tree WHERE parent_id = %s and dt_id = %s and question_text = 'Yes'", (yes_id, dt_id))
                                            yes_id = cursor1.fetchone()
                                            print("yes_id = ", yes_id)

                                            if yes_id:  
                                                yes_id = yes_id[0]
                                            if no_id:
                                                no_id = no_id[0]

                                            if link_id[0] == "0":
                                                send_msg(remote_phone_number, question_text[0])
                                                current_last_uuid.append(str(uuid))
                                                current_stage = "ongoing_solution"
                                                store_user_interaction(remote_phone_number, current_stage, solution_number=0, result=result, issue=issue, dt_id=dt_id, action=no_id, yes_id=yes_id)
                                                set_stage(stage="tech_support", phone_number=remote_phone_number, last_uuid=current_last_uuid, last_time=created_at)

                                            else:
                                                video_name = link_id[0]
                                                send_msg(remote_phone_number, question_text[0])
                                                send_msg(remote_phone_number, f"{link_url}/videos/{video_name}")
                                                current_last_uuid.append(str(uuid))
                                                current_stage = "ongoing_solution"
                                                store_user_interaction(remote_phone_number, current_stage, solution_number=0, result=result, issue=issue, dt_id=dt_id, action=no_id, yes_id=yes_id)
                                                set_stage(stage="tech_support", phone_number=remote_phone_number, last_uuid=current_last_uuid, last_time=created_at)

                                    else:
                                        send_msg(remote_phone_number, "Thank you for contacting us.")
                                        current_last_uuid.append(str(uuid))
                                        set_stage(stage="start", phone_number=remote_phone_number, last_uuid=current_last_uuid, last_time=created_at)

                            elif get_user_interaction(remote_phone_number)["stage"] == "live_agent":
                                if message_text.lower() == "yes":
                                    send_msg(remote_phone_number, "Thank you for contacting us. We will connect you with a live agent shortly.")

                                else:
                                    send_msg(remote_phone_number, "Thank you for contacting us.")
                    
                    elif get_stage(remote_phone_number)["stage"] == "live_agent" and str(uuid) not in get_stage(remote_phone_number)["last_uuid"]:
                        if message_text.lower() == "yes":
                            send_msg(remote_phone_number, "Thank you for contacting us. We will connect you with a live agent shortly.")
                            current_last_uuid.append(str(uuid))
                            set_stage(stage="start", phone_number=remote_phone_number, last_uuid=current_last_uuid, last_time=created_at)

                        elif message_text.lower() == "no":
                            send_msg(remote_phone_number, "Thank you for contacting us.")
                            current_last_uuid.append(str(uuid))
                            set_stage(stage="start", phone_number=remote_phone_number, last_uuid=current_last_uuid, last_time=created_at)