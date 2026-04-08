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
    has_company_logo: Optional[int] = 0   # 0 or 1

app = FastAPI(
    title="Job Fraud Detection API",
    description="API for predicting job posting fraud probability using Random Forest model",
    version="1.1.0"
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
        print("Model failed to load")
@app.get("/")
async def root():
    return {
        "message": "Job Fraud Detection API",
        "version": "1.1.0",
        "features_used": [
            "text (title, description, requirements, benefits, company_profile)",
            "categorical (location, department, salary_range, employment_type, required_experience, required_education)",
            "numeric (has_company_logo)"
        ]
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
            'required_education': [job_data.required_education or "missing"],
            'has_company_logo': [job_data.has_company_logo if job_data.has_company_logo is not None else 0]
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


        p = float(probabilities[0][1])

        if p >= 0.5:
            fraud_score = 1
        else:
            fraud_score=p*2.5
        
        return {
            "fraudScore": fraud_score,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")