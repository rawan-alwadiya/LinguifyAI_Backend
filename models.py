from pydantic import BaseModel

class TextInput(BaseModel):
    input_text: str
    user_id: str = None
    is_related: bool = True

class UserAnswer(BaseModel):
    question: str
    user_choice: str    

class GrammarRequest(BaseModel):
    text: str  # Input text from Flutter

class SummarizationRequest(BaseModel):
    text: str  # Input text to summarize    
