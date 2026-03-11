from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import joblib
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

class JobData(BaseModel):
    title: str = ""
    description: str = ""
    requirements: str = ""
    benefits: str = ""
    company_profile: str = ""
    location: Optional[str] = None
    department: Optional[str] = None
    salary_range: Optional[str] = None
    employment_type: Optional[str] = None
    required_experience: Optional[str] = None
    required_education: Optional[str] = None

app = FastAPI(
    title="Job Fraud Detection API",
    description="API for predicting job posting fraud probability using Random Forest model",
    version="1.0.0"
)

model = None
preprocessor = None

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    words = [
        lemmatizer.lemmatize(w)
        for w in text.split()
        if w not in stop_words
    ]
    return " ".join(words)

def load_model():
    global model, preprocessor
    try:
        model_data = joblib.load('model/randomforest_model.pkl')
        model = model_data['model']
        preprocessor = model_data['preprocessor']
        return True
    except Exception:
        return False

@app.on_event("startup")
async def startup_event():
    if not load_model():
        pass

@app.get("/")
async def root():
    return {
        "message": "Job Fraud Detection API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "POST endpoint for fraud prediction",
            "/health": "GET endpoint to check API health"
        },
        "model_info": {
            "type": "Random Forest Classifier",
            "threshold": 0.30,
            "output_range": "0.0 to 1.0 (fraud probability)"
        }
    }

@app.get("/health")
async def health_check():
    if model is None or preprocessor is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict")
async def predict_fraud(job_data: JobData):
    try:
        if model is None or preprocessor is None:
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        input_data = pd.DataFrame({
            'title': [job_data.title],
            'description': [job_data.description],
            'requirements': [job_data.requirements],
            'benefits': [job_data.benefits],
            'company_profile': [job_data.company_profile],
            'location': [job_data.location or "missing"],
            'department': [job_data.department or "missing"],
            'salary_range': [job_data.salary_range or "missing"],
            'employment_type': [job_data.employment_type or "missing"],
            'required_experience': [job_data.required_experience or "missing"],
            'required_education': [job_data.required_education or "missing"]
        })
        
        input_data['combined_text'] = (
            input_data['title'] + " " +
            input_data['description'] + " " +
            input_data['requirements'] + " " +
            input_data['benefits'] + " " +
            input_data['company_profile']
        ).apply(clean_text)
        
        X_transformed = preprocessor.transform(input_data)
        probabilities = model.predict_proba(X_transformed)
        fraud_score = float(probabilities[0][1])
        
        return {"fraudScore": fraud_score}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

