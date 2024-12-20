import uuid
import tensorflow as tf
import tensorflow_hub as hub
from transformers import GPT2Tokenizer, GPTNeoForCausalLM
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from sentence_transformers import SentenceTransformer, util
from models import *
import re
import pandas as pd
from happytransformer import HappyTextToText, TTSettings
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import nltk
nltk.download("punkt")
nltk.download('punkt_tab')

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict to specific domains here
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load GPT-Neo model and tokenizer for text generation
gpt_neo_model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")
tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")

# Load Sentence-BERT model for semantic similarity
similarity_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Load the Gramformer model
gramformer_model = HappyTextToText("T5", "C:\\Users\\OMEN\\Desktop\\LinguifyAI_Backend\\gramformer_model")

# Load BrevityBot model and tokenizer
model_dir = "C:\\Users\\OMEN\\Desktop\\LinguifyAI_Backend\\t5-summarization-final"
model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained(model_dir)

# Store session context for each user
user_context = {}

# Generate a unique user ID using UUID (Version 4)
def generate_user_id():
    return str(uuid.uuid4())

# Token counting utility to prevent exceeding limits
def count_tokens(text):
    tokens = tokenizer.encode(text)
    return len(tokens)

# Semantic similarity check function
def is_message_related(new_message, previous_message, threshold=0.7):
    if not previous_message:
        return True  # If no previous message, assume it's related
    
    # Calculate embeddings for both messages
    embeddings = similarity_model.encode([new_message, previous_message], convert_to_tensor=True)
    similarity_score = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()

    return similarity_score >= threshold

# Function to manage conversation context within token limits
def update_context(user_id, new_message, max_tokens=2048):
    if user_id not in user_context:
        user_context[user_id] = []

    context = user_context[user_id]

    # Check if the new message is related to the last context message
    if context and not is_message_related(new_message, context[-1]):
        context.clear()  # Clear context if the message is unrelated

    context.append(new_message)

    # Remove old messages if the token limit is exceeded
    total_tokens = count_tokens(' '.join(context))
    while total_tokens > max_tokens:
        context.pop(0)
        total_tokens = count_tokens(' '.join(context))

    return ' '.join(context)

def clean_response(prompt, response):
    # Find where the prompt ends and extract the remaining part of the response
    if response.startswith(prompt):
        response = response[len(prompt):].strip()
    response = re.sub(r'[|\\-`_@]', '', response)  # Remove pipes, backslashes, dashes, underscores, backticks, and @ symbols    
    if not response.endswith('.'):
        response = response.rsplit('.', 1)[0]    
    return response.strip()  # Trim any leading/trailing whitespace

def format_response(response):
    # Replace newline characters and any excess whitespace
    response = response.replace("\n", " ").strip()
    response = re.sub(r'\s+',' ', response)
    return ' '.join(response.split())  # Clean up any extra spaces

# GPT-Neo: Text generation based on prompt and token management
def generate_response(user_id, prompt, include_context=True, max_response_length=300):

    if include_context:
        context = update_context(user_id, prompt)
    else:
        context = prompt  # Ignore previous context for unrelated questions

    # tokenizer.pad_token = tokenizer.eos_token    
    inputs = tokenizer(context, return_tensors="pt",truncation=True
                        # padding=True,
                        )
    # Check cancellation during processing

    outputs = gpt_neo_model.generate(
        inputs["input_ids"],
        # attention_mask=inputs['attention_mask'],
        max_length=max_response_length
        ,do_sample=True,temperature=0.9
    #    ,temperature=0.7,top_p=0.9,top_k=50, 
    # ,pad_token_id=tokenizer.eos_token_id
            )
    print(f"Model Output: {outputs}")
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Decoded Response: {response}")

    # Clean up the response
    cleaned_response = clean_response(prompt, response)
    formatted_response = format_response(cleaned_response)

    # Update context with the model's response if related
    if include_context:
        update_context(user_id, formatted_response)
    return formatted_response   

df = pd.read_csv("C:\\Users\\OMEN\\Desktop\\LinguifyAI_Backend\\grammar_mcq_data\\cleaned_questions_with_text_answers.csv")

## Cleaning the Data

# corrupted_indices = list(range(90, 95)) + list(range(115, 120)) +  list(range(159, 164)) + list(range(204, 209)) + list(range(229, 234)) + list(range(254, 259)) + list(range(279, 284)) + list(range(303, 308)) + list(range(327, 332)) + list(range(351, 356)) + list(range(376, 381)) + list(range(401, 406)) + list(range(421, 426)) + list(range(446, 451)) + list(range(471, 476)) + list(range(496, 501)) + list(range(521, 526)) + list(range(546, 551)) + list(range(571, 576)) + list(range(596, 601)) + list(range(621, 626)) + list(range(646, 651)) + list(range(671, 676)) + list(range(696, 701)) + list(range(721, 726)) + list(range(746, 751)) + list(range(771, 776)) + list(range(796, 801)) + list(range(821, 826)) + list(range(846, 851)) + list(range(871, 876)) + list(range(898, 903)) + list(range(923, 928)) + list(range(948, 953)) + list(range(973, 978)) + list(range(998, 1003)) + list(range(1023, 1028)) + list(range(1048, 1053)) + list(range(1073, 1078)) + list(range(1098, 1103)) + list(range(1123, 1128)) + list(range(1148, 1153)) + list(range(1173, 1178)) + list(range(1198, 1203)) + list(range(1223, 1228)) + list(range(1248, 1253)) + list(range(1273, 1278)) + list(range(1298, 1303)) + list(range(1323, 1328)) + list(range(1348, 1353)) + list(range(1373, 1378)) + list(range(1398, 1403)) + list(range(1423, 1428)) + list(range(1448, 1453)) + list(range(1473, 1478)) + list(range(1498, 1503)) + list(range(1523, 1528)) + list(range(1548, 1553)) + list(range(1573, 1578)) + list(range(1598, 1603)) + list(range(1623, 1628)) + list(range(1660, 1670)) + list(range(1710, 1720)) + list(range(1760, 1770)) + list(range(1810, 1820)) + list(range(1860, 1870)) + list(range(1910, 1921)) + list(range(1960, 1970)) + list(range(1998, 1999)) + list(range(2011, 2020)) + list(range(2060, 2070)) + list(range(2110, 2120)) + list(range(2148, 2153)) + list(range(2173, 2178)) + list(range(2198, 2203)) + list(range(2223, 2228)) + list(range(2248, 2253)) + list(range(2273, 2278)) + list(range(2298, 2303)) + list(range(2323, 2328)) + list(range(2348, 2353)) + list(range(2373, 2378)) + list(range(2396, 2401)) + list(range(2426, 2431)) + list(range(2456, 2461)) + list(range(2506, 2511)) + list(range(2531, 2536)) + list(range(2556, 2561)) + list(range(2566, 2576))

# Drop the corrupted rows
# df_cleaned = df.drop(index=corrupted_indices, errors='ignore')

# Reset the index to ensure consistency
# df_cleaned.reset_index(drop=True, inplace=True)
##

# Add the answer_text column
# def map_answer_to_choice(row):
#     choice_mapping = {
#         1.0: row['choice_1'],
#         2.0: row['choice_2'],
#         3.0: row['choice_3'],
#         4.0: row['choice_4']
#     }
#     return choice_mapping.get(row['answer'], None)

# df['answer_text'] = df.apply(map_answer_to_choice, axis=1)

# df.to_csv("C:\\Users\\OMEN\\Desktop\\LinguifyAI_Backend\\grammar_mcq_data\\cleaned_questions_with_text_answers.csv", index=False)

# Configure model settings
gramformer_settings = TTSettings(
    do_sample=True,
    top_k=20,
    temperature=0.7,
    min_length=1,
    max_length=100,
    num_beams=5,
    early_stopping=True,
)

# Grammar correction function
def correct_grammar(text: str) -> str:
    sentences = nltk.sent_tokenize(text)
    corrected_sentences = [gramformer_model.generate_text(sentence, args=gramformer_settings).text for sentence in sentences]
    return " ".join(corrected_sentences)

# Summarization Function
def summarize_text(input_text: str) -> str:
    try:
        inputs = ["summarize: " + input_text]
        tokenized_inputs = tokenizer(inputs, max_length=512, truncation=True, return_tensors="pt")
        outputs = model.generate(**tokenized_inputs, max_length=64, num_beams=4)
        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return summary
    except Exception as e:
        raise Exception(f"Summarization failed: {str(e)}")

# Route for text input (GPT Neo)
@app.post("/text")
async def chat_text( text_input : TextInput):
    if not text_input.user_id:
        text_input.user_id = generate_user_id()  

    response = generate_response(text_input.user_id, text_input.input_text, include_context = text_input.is_related)
    return {"response": response, "user_id": text_input.user_id}

# Route to get a random question
@app.get("/random_question")
def get_random_question():
    random_row = df.sample(n=1).iloc[0]
    question_data = {
        "question": random_row["question"],
        "choices": {
            "1": random_row["choice_1"],
            "2": random_row["choice_2"],
            "3": random_row["choice_3"],
            "4": random_row["choice_4"]
        }
    }
    return question_data

# Route to validate user answer
@app.post("/validate_answer")
def validate_answer(user_answer: UserAnswer):
    # Find the question in the dataset
    matching_row = df[df["question"] == user_answer.question]
    
    if matching_row.empty:
        raise HTTPException(status_code=404, detail="Question not found")
    
    correct_answer = matching_row.iloc[0]["answer_text"]
    
    if user_answer.user_choice == correct_answer:
        return {
            "correct": True,
            "message": "Awesome! Your answer is correct!",
            "correct_answer": correct_answer
        }
    else:
        return {
            "correct": False,
            "message": "Oops! Try again.",
            "correct_answer": correct_answer
        }
    

@app.post("/correct_grammar")
async def correct_grammar_route(request: GrammarRequest):
    try:
        corrected_text = correct_grammar(request.text)
        return {"corrected_text": corrected_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")  

# Route for text summarization
@app.post("/summarize")
async def summarize_route(request: SummarizationRequest):
    try:
        summary = summarize_text(request.text)
        return {"summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")       
    

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 