# LinguifyAI_Backend

LinguifyAI Backend is the AI-powered backend for the LinguifyAI language learning application. Built using FastAPI, it provides API endpoints for grammar correction, summarization, grammar quizzes, and chatbot interactions.

## Features & API Endpoints

### Grammar Tools
- *Grammar Correction*: Fixes sentence grammar errors using a T5-based model.  
  *Endpoint*: /grammar-correction
- *Grammar Quiz*: Generates multiple-choice grammar questions from a dataset.  
  *Endpoint*: /random_question
- *Answer Validation*: Checks if a user’s selected answer is correct.  
  *Endpoint*: /validate_answer

### Summarization
- Summarizes input text using a fine-tuned T5 model.  
  *Endpoint*: /summarize

### ChatPro AI
- Provides chatbot responses using GPT-Neo with conversational memory.  
  *Endpoint*: /text

## Technology Stack
- FastAPI (Backend framework)
- Sentence-BERT (Semantic similarity)
- GPT-Neo (Chatbot responses)
- T5 (Grammar correction & summarization)
- HappyTextToText (Grammar correction model interface)
- TensorFlow/PyTorch (AI model deployment)
- Uvicorn (ASGI server for FastAPI)
- Pydantic (Data validation & serialization)
- NLTK (Sentence tokenization)
- Pandas (Data handling for grammar quiz)
