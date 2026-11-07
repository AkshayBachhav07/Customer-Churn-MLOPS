Customer churn prediction enables telecom operators to proactively detect accounts at risk of contract cancellation. The system processes customer tenure, billing tiers, internet service configurations, payment modes, and support escalations to compute churn probability:

Machine Learning Pipeline (train.py): Trains a RandomForestClassifier paired with standard numerical feature scalers (StandardScaler) wrapped in a Scikit-Learn Pipeline. Stratified sampling is applied to balance customer churn ratios.

API Microservice (app.py): A high-throughput FastAPI application exposing /health and /predict endpoints. Request attributes are validated using Pydantic schemas.

Testing & Quality Assurance (test_app.py): Executes automated unit and regression tests using pytest and httpx, verifying health checks, payload bounds, and model inference responses.

Containerization (Dockerfile): Packages the application inside a python:3.11-slim container, exposing port 8000.

CI/CD Pipeline (.github/workflows/ci-cd.yml): Automatically sets up Python, installs dependencies, verifies training reproducibility, runs test suites, and validates Docker builds on every push to main.


Customer-Churn-MLOPS/
│
├── .github/
│   └── workflows/
│       └── ci-cd.yml          # GitHub Actions workflow (test suite & Docker verification)
│
├── data/
│   └── dataset.csv            # Tabular customer subscription and support dataset
│
├── models/
│   └── model.pkl              # Serialized Scikit-learn Pipeline (Scaler + RandomForest)
│
├── app.py                     # FastAPI server handling /health and /predict endpoints
├── train.py                   # Data synthesizer, model training, and artifact exporter
├── test_app.py                # Automated Pytest suite for API endpoints and model validation
├── requirements.txt           # Production and test dependencies
├── Dockerfile                 # Multi-stage container definition
├── .gitignore                 # Cache and virtual environment ignores
└── README.md                  # Project documentation and API contract guide


You can save and commit this documentation directly to your repository:

PowerShell
Set-Content -Path "README.md" -Value @'
# Customer Churn Prediction System (End-to-End MLOps)

An end-to-end Machine Learning Operations (MLOps) microservice designed to predict telecom customer churn risk. The service evaluates subscriber tenure, monthly charges, contract structure, internet connection type, payment automation, and customer support ticket history to forecast churn probability in real time.

---

## 📌 Repository Architecture

```text
Customer-Churn-MLOPS/
├── .github/
│   └── workflows/
│       └── ci-cd.yml          # Automated CI/CD (lint, train, test, Docker build)
├── data/
│   └── dataset.csv            # Tabular customer subscription dataset
├── models/
│   └── model.pkl              # Serialized Scikit-learn Pipeline
├── app.py                     # FastAPI REST API with Pydantic validation
├── train.py                   # Data ingestion, preprocessing, training & export
├── test_app.py                # Automated Pytest suite for API & model logic
├── requirements.txt           # Pinned production and test dependencies
├── Dockerfile                 # Container build specification
└── .gitignore                 # Environment and cache ignores
🛠️ Tech Stack
Language: Python 3.11

Machine Learning: Scikit-learn, Pandas, NumPy, Joblib

API Framework: FastAPI, Uvicorn, Pydantic

Testing: Pytest, HTTPX

DevOps & Cloud: Docker, GitHub Actions, Render

🚀 Local Setup & Execution
1. Clone the Repository
Bash
git clone [https://github.com/AkshayBachhav07/Customer-Churn-MLOPS.git](https://github.com/AkshayBachhav07/Customer-Churn-MLOPS.git)
cd Customer-Churn-MLOPS
2. Create and Activate Virtual Environment
Bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
3. Install Dependencies
Bash
pip install -r requirements.txt
4. Train the Model
Generates data/dataset.csv and serializes the pipeline to models/model.pkl:

Bash
python train.py
5. Run Automated Tests
Bash
pytest test_app.py -v
6. Launch the Local API Server
Bash
uvicorn app:app --reload
Open http://127.0.0.1:8000/docs to access the interactive Swagger UI.

🐳 Docker Execution
Build Container
Bash
docker build -t customer-churn-api:latest .
Run Container
Bash
docker run -d -p 8000:8000 --name churn-service customer-churn-api:latest
Access the containerized API at http://localhost:8000/docs.

📡 API Specification & Contract
1. Health Check
Endpoint: GET /health

Response:

JSON
{
  "status": "healthy",
  "model_artifact_loaded": true
}
2. Predict Customer Churn
Endpoint: POST /predict

Sample Request:

JSON
{
  "tenure_months": 2,
  "monthly_charges": 105.0,
  "contract_type": 0,
  "internet_service": 1,
  "payment_method": 0,
  "support_tickets_count": 4
}
Sample Response:

JSON
{
  "churn_prediction": 1,
  "risk_level": "High Risk (Will Churn)",
  "churn_probability": 0.8421,
  "status": "Success"
}
🔄 CI/CD Workflow
The .github/workflows/ci-cd.yml pipeline triggers on all pushes and pull requests to main:

Checks out repository code.

Configures Python 3.11 environment with pip dependency caching.

Installs requirements from requirements.txt.

Runs train.py to confirm deterministic model artifact generation.

Executes the pytest test suite.

Builds the Docker image to ensure deployment readiness.
'@

git add README.md
git commit -m "docs: add comprehensive project overview, architecture, and API documentation"
git push origin main
