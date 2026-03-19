import requests

def test_api():
    legitimate_job = {
        "title": "Software Engineer",
        "description": "We are looking for a talented software engineer to join our team. You will work on developing web applications using modern technologies.",
        "requirements": "Bachelor's degree in Computer Science, 3+ years experience with Python and JavaScript, knowledge of React and Node.js",
        "benefits": "Health insurance, 401k matching, flexible work hours, remote work options",
        "company_profile": "TechCorp is a leading software development company specializing in web applications and cloud services",
        "location": "US, CA, San Francisco",
        "department": "Engineering",
        "salary_range": "75000-100000",
        "employment_type": "Full-time",
        "required_experience": "Mid-Senior level",
        "required_education": "Bachelor's Degree"
    }
    
    fraudulent_job = {
        "title": "Work From Home Data Entry",
        "description": "Earn $5000 per week from home! No experience required. Just click the link to get started.",
        "requirements": "Must have internet connection and email address",
        "benefits": "Unlimited earning potential, no boss, work anytime",
        "company_profile": "We are a global company with millions of satisfied customers",
        "location": "Remote",
        "department": "Data Entry",
        "salary_range": "5000-10000",
        "employment_type": "Part-time",
        "required_experience": "Not Applicable",
        "required_education": "High School or equivalent"
    }
    base_url = "http://localhost:5000"
    
    print("Testing Job Fraud Detection API")
    print("=" * 50)
    
    print("\n1. Testing health check...")
    try:
        response = requests.get(f"{base_url}/health")
        print(f"Health check: {response.status_code} - {response.json()}")
    except requests.exceptions.ConnectionError:
        print("API is not running. Please start the server first:")
        print("   python app.py")
        return
    except Exception as e:
        print(f"Health check failed: {e}")
        return
    
    print("\n2. Testing legitimate job posting...")
    try:
        response = requests.post(f"{base_url}/predict", json=legitimate_job)
        result = response.json()
        print(f"Legitimate job fraud score: {result['fraudScore']:.4f}")
        print(f"Status: {'Low Risk' if result['fraudScore'] < 0.3 else 'High Risk'}")
    except Exception as e:
        print(f"Legitimate job test failed: {e}")
    
    print("\n3. Testing potentially fraudulent job posting...")
    try:
        response = requests.post(f"{base_url}/predict", json=fraudulent_job)
        result = response.json()
        print(f"Fraudulent job fraud score: {result['fraudScore']:.4f}")
        print(f"Status: {'Low Risk' if result['fraudScore'] < 0.3 else 'High Risk'}")
    except Exception as e:
        print(f"Fraudulent job test failed: {e}")
    
    print("\n" + "=" * 50)
    print("Test completed!")

if __name__ == "__main__":
    test_api()
