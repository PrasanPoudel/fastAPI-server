import requests

def test_api():
    jobs = [
    {
        "title": "Senior Backend Software Engineer",
        "description": "We are seeking an experienced backend engineer to design, build, and maintain scalable APIs and microservices. You will collaborate with cross-functional teams, participate in code reviews, and contribute to system architecture decisions.",
        "requirements": "Bachelor’s degree in Computer Science or related field, 5+ years of experience in backend development, strong knowledge of Python, Django, REST APIs, and cloud platforms like AWS.",
        "benefits": "Health, dental, and vision insurance, paid time off, remote work flexibility, annual bonus, learning and development budget.",
        "company_profile": "InnovateTech Solutions is a mid-sized SaaS company focused on building cloud-based enterprise tools for global clients.",
        "location": "US, NY, New York",
        "department": "Engineering",
        "salary_range": "110000-140000",
        "employment_type": "Full-time",
        "required_experience": "Senior level",
        "required_education": "Bachelor's Degree",
        "has_company_logo": 1
    },
    {
        "title": "Frontend Developer (React)",
        "description": "Develop and maintain user-facing features using React.js. Work closely with designers to implement responsive UI/UX and optimize applications for performance.",
        "requirements": "3+ years of frontend development experience, proficiency in JavaScript, React, HTML, CSS, familiarity with REST APIs.",
        "benefits": "Flexible working hours, remote-friendly, health insurance, team retreats.",
        "company_profile": "BrightApps is a digital product agency delivering modern web and mobile solutions to startups and enterprises.",
        "location": "UK, London",
        "department": "Engineering",
        "salary_range": "60000-80000",
        "employment_type": "Full-time",
        "required_experience": "Mid level",
        "required_education": "Bachelor's Degree"
    },
    {
        "title": "Data Analyst",
        "description": "Analyze structured and unstructured data to provide actionable insights. Create dashboards, reports, and collaborate with business teams.",
        "requirements": "Bachelor’s degree in Statistics, Mathematics, or related field, experience with SQL, Python, and data visualization tools like Tableau or Power BI.",
        "benefits": "Health insurance, retirement plan, paid leave, hybrid work setup.",
        "company_profile": "DataInsights Corp specializes in business intelligence and analytics solutions for retail and finance sectors.",
        "location": "Canada, Toronto",
        "department": "Analytics",
        "salary_range": "70000-90000",
        "employment_type": "Full-time",
        "required_experience": "Mid level",
        "required_education": "Bachelor's Degree"
    },
    {
        "title": "DevOps Engineer",
        "description": "Manage CI/CD pipelines, monitor system performance, and ensure high availability of applications. Work with cloud infrastructure and automation tools.",
        "requirements": "4+ years of DevOps experience, strong knowledge of Docker, Kubernetes, AWS, CI/CD tools like Jenkins or GitHub Actions.",
        "benefits": "Remote work, health benefits, stock options, flexible hours.",
        "company_profile": "CloudNet Systems provides scalable cloud infrastructure and DevOps consulting services.",
        "location": "Germany, Berlin",
        "department": "Operations",
        "salary_range": "90000-120000",
        "employment_type": "Full-time",
        "required_experience": "Mid-Senior level",
        "required_education": "Bachelor's Degree"
    },
    {
        "title": "Mobile Application Developer",
        "description": "Design and build mobile applications for Android and iOS platforms. Collaborate with product managers and designers to deliver high-quality apps.",
        "requirements": "Experience with Flutter or React Native, knowledge of mobile app architecture, debugging, and deployment processes.",
        "benefits": "Health insurance, paid holidays, flexible schedule, remote work options.",
        "company_profile": "NextGen Apps builds innovative mobile solutions for e-commerce and fintech startups.",
        "location": "India, Bangalore",
        "department": "Engineering",
        "salary_range": "50000-70000",
        "employment_type": "Full-time",
        "required_experience": "Mid level",
        "required_education": "Bachelor's Degree"
    },

    {
        "title": "Work From Home Data Entry Clerk",
        "description": "Earn up to $300 daily by entering simple data online. No skills required. Start immediately with guaranteed earnings.",
        "requirements": "No experience needed. Just basic computer knowledge and internet access.",
        "benefits": "Daily payments, bonuses, free training, flexible hours.",
        "company_profile": "Global Data Solutions is expanding rapidly and offering opportunities worldwide.",
        "location": "Anywhere",
        "department": "Clerical",
        "salary_range": "30000-90000 per month",
        "employment_type": "Full-time / Remote",
        "required_experience": "None",
        "required_education": "Not required"
    },
    {
        "title": "Online Assistant - Immediate Start",
        "description": "Assist clients online and get paid instantly. Limited positions available. Fast hiring process with no interview.",
        "requirements": "No degree required, no prior experience needed.",
        "benefits": "Instant payouts, performance bonuses, free laptop provided.",
        "company_profile": "We are a fast-growing global company with unlimited earning potential.",
        "location": "Remote",
        "department": "Support",
        "salary_range": "40000-80000 per month",
        "employment_type": "Full-time",
        "required_experience": "None",
        "required_education": "Not required"
    },
    {
        "title": "Social Media Manager (Remote)",
        "description": "Manage social media accounts and earn money quickly. No prior experience required. Easy tasks and high income potential.",
        "requirements": "Basic understanding of social media platforms.",
        "benefits": "High commissions, daily bonuses, flexible work schedule.",
        "company_profile": "A reputed international company offering online income opportunities.",
        "location": "Worldwide",
        "department": "Marketing",
        "salary_range": "50000-100000 per month",
        "employment_type": "Remote",
        "required_experience": "Entry level",
        "required_education": "Not required"
    },
    {
        "title": "Part-Time Online Job Opportunity",
        "description": "Make money online in your spare time. Simple tasks with guaranteed income. Sign up today and start earning.",
        "requirements": "No experience or qualifications required.",
        "benefits": "Weekly payouts, referral bonuses, flexible hours.",
        "company_profile": "An international platform connecting workers with easy online jobs.",
        "location": "Anywhere",
        "department": "General",
        "salary_range": "20000-60000 per month",
        "employment_type": "Part-time",
        "required_experience": "None",
        "required_education": "Not required"
    },
    {
        "title": "Remote Customer Service Representative",
        "description": "Provide customer support from home and earn high income. Immediate hiring with guaranteed salary.",
        "requirements": "No prior experience needed. Training provided.",
        "benefits": "High salary, bonuses, work from home, flexible schedule.",
        "company_profile": "A global service provider with rapid growth and high-paying roles.",
        "location": "Remote",
        "department": "Customer Support",
        "salary_range": "60000-120000 per month",
        "employment_type": "Full-time",
        "required_experience": "Entry level",
        "required_education": "Not required"
    }
]
    base_url = "http://localhost:5000"
    
    print("Testing Job Fraud Detection API")
    print("=" * 50)
    
    # 1. Health check
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

    # 2. Test all jobs
    print("\n2. Testing all job postings...\n")

    for i, job in enumerate(jobs, 1):
        try:
            response = requests.post(f"{base_url}/predict", json=job)
            result = response.json()
            score = result['fraudScore']

            print(f"{i}. {job['title']}")
            print(f"   Fraud Score: {score:.4f}")
            print(f"   Status: {'Low Risk' if score < 0.3 else 'High Risk'}\n")

        except Exception as e:
            print(f"{i}. Error testing job: {e}\n")

    print("=" * 50)
    print("Test completed!")

if __name__ == "__main__":
    test_api()