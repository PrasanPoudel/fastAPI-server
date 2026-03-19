# Job Fraud Detection FastAPI Application

A comprehensive guide for setting up and running the Job Fraud Detection API using FastAPI.

## Overview

This application provides a REST API for predicting job posting fraud probability using a Random Forest machine learning model. The API analyzes job posting text data to detect potential fraudulent job listings.

## Project Structure

```
fastapi-server/
├── app.py              # Main FastAPI application
├── server.py           # Server startup script
├── requirements.txt    # Python dependencies
├── test_api.py         # API testing script
├── GUIDE.md           # This documentation file
└── src/
    └── models/
        └── randomforest_model.pkl  # Pre-trained ML model
```

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## Installation and Setup

### 1. Create Virtual Environment

It's recommended to use a virtual environment to isolate dependencies:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 2. Install Dependencies

Install all required packages from the requirements.txt file:

```bash
pip install -r requirements.txt
```

**Required packages:**

- `fastapi==0.104.1` - Web framework
- `uvicorn[standard]==0.24.0` - ASGI server
- `pandas==2.2.2` - Data manipulation
- `numpy==1.26.4` - Numerical computing
- `scikit-learn==1.4.2` - Machine learning
- `joblib==1.4.2` - Model serialization
- `pydantic==2.5.0` - Data validation
- `python-multipart==0.0.6` - Form data handling
- `nltk`
- `requests`

### 3. Download NLTK Data

The application uses NLTK for text preprocessing. The first time you run the server, it will automatically download required NLTK data packages:

- `punkt` - Tokenizer
- `stopwords` - Stop words list
- `wordnet` - WordNet lemmatizer

## Running the Application

### Start the Server

```bash
python server.py
```

The server will start and display information:

```
============================================================
Job Fraud Detection FastAPI Server
============================================================

Starting FastAPI server...
Server will be available at: http://localhost:5000
API endpoint: http://localhost:5000/predict
Health check: http://localhost:5000/health
Documentation: http://localhost:5000/docs

Press Ctrl+C to stop the server
------------------------------------------------------------
```

### Server Configuration

- **Host**: `0.0.0.0` (accessible from any network interface)
- **Port**: `5000`
- **Auto-reload**: Disabled (for production stability)
- **Log level**: Info

## API Endpoints

### 1. Root Endpoint

**GET** `/`

Returns API information and available endpoints.

**Response:**

```json
{
  "message": "Job Fraud Detection API",
  "version": "1.1.0",
  "features_used": [
    "text (title, description, requirements, benefits, company_profile)",
    "categorical (location, department, salary_range, employment_type, required_experience, required_education)",
    "numeric (has_company_logo)"
  ]
}
```

### 2. Health Check

**GET** `/health`

Checks if the model is loaded and the API is ready.

**Response:**

```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### 3. Fraud Prediction

**POST** `/predict`

Analyzes job posting data and returns fraud probability score.

**Request Body:**

```json
{
  "title": "Software Engineer",
  "description": "We are looking for a talented software engineer...",
  "requirements": "Bachelor's degree in Computer Science...",
  "benefits": "Health insurance, 401k matching...",
  "company_profile": "We are a leading technology company...",
  "location": "San Francisco, CA",
  "department": "Engineering",
  "salary_range": "80000-120000",
  "employment_type": "Full-time",
  "required_experience": "3-5 years",
  "required_education": "Bachelor's Degree"
}
```

**Response:**

```json
{
  "fraudScore": 0.23
}
```

**Response Fields:**

- `fraudScore`: Float between 0.0 and 1.0 representing fraud probability
  - 0.0 = Very unlikely to be fraudulent
  - 1.0 = Very likely to be fraudulent
  - Threshold of 0.30 is typically used to flag suspicious postings

## Testing the API

### Using the Test Script

A test script is provided to verify the API functionality:

```bash
python test_api.py
```

This script sends sample job data to the `/predict` endpoint and displays the response.

### Manual Testing

You can also test the API manually using curl:

```bash
# Health check
curl http://localhost:5000/health

# Fraud prediction
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Software Engineer",
    "description": "We are looking for a talented software engineer...",
    "requirements": "Bachelor'\''s degree in Computer Science...",
    "benefits": "Health insurance, 401k matching...",
    "company_profile": "We are a leading technology company...",
    "location": "San Francisco, CA",
    "department": "Engineering",
    "salary_range": "80000-120000",
    "employment_type": "Full-time",
    "required_experience": "3-5 years",
    "required_education": "Bachelor'\''s Degree"
  }'
```

### Interactive Documentation

FastAPI automatically generates interactive API documentation:

- **Swagger UI**: http://localhost:5000/docs
- **ReDoc**: http://localhost:5000/redoc

These provide a web interface to explore and test all API endpoints.

## Model Information

### Machine Learning Model

- **Type**: Random Forest Classifier
- **Input**: Job posting text data (title, description, requirements, etc.)
- **Output**: Fraud probability score (0.0 to 1.0)
- **Preprocessing**: Text cleaning, stop word removal, lemmatization
- **Features**: TF-IDF vectorization of combined text fields

### Model File

The pre-trained model is located at:

- `src/models/randomforest_model.pkl`

This file contains both the trained model and the text preprocessor.

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError: No module named 'nltk'**
   - Solution: Run `pip install -r requirements.txt` to install all dependencies

2. **Model not loaded error**
   - Ensure `src/models/randomforest_model.pkl` exists
   - Check that the model file is not corrupted

3. **Port already in use**
   - Change the port in `server.py` or stop other services using port 5000

4. **NLTK data download issues**
   - The application will automatically download required NLTK data on first run
   - If issues persist, manually download: `python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"`

### Development Mode

For development with auto-reload, modify `server.py`:

```python
uvicorn.run(
    "app:app",
    host="0.0.0.0",
    port=5000,
    reload=True,  # Enable auto-reload
    log_level="info"
)
```

## Production Deployment

### Environment Variables

For production, consider using environment variables for configuration:

```python
import os

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 5000))
DEBUG = os.getenv("DEBUG", False)
```

### Security Considerations

- Use HTTPS in production
- Implement API authentication if needed
- Set appropriate CORS policies
- Monitor and log API usage

### Docker Deployment

Create a `Dockerfile`:

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["python", "server.py"]
```

Build and run:

```bash
docker build -t job-fraud-api .
docker run -p 5000:5000 job-fraud-api
```

## Performance Optimization

### Model Loading

The model is loaded once at startup. For high-traffic applications:

- Consider lazy loading if startup time is critical
- Monitor memory usage with the loaded model
- Implement model caching strategies if needed

### Text Processing

Text preprocessing happens on each request. For optimization:

- Consider caching preprocessed text for similar inputs
- Monitor response times under load
- Use async processing for heavy text analysis if needed

## Contributing

1. Set up the development environment as described above
2. Make your changes
3. Test using `test_api.py`
4. Verify the API documentation at http://localhost:5000/docs

## License

This project is for educational and demonstration purposes. Ensure compliance with data usage policies when deploying in production environments.

## Support

For issues or questions:

1. Check the troubleshooting section above
2. Review the API documentation at http://localhost:5000/docs
3. Examine the test script for usage examples
