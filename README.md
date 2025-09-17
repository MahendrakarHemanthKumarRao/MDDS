Multi-Disease Detection System (MDDS)

A comprehensive health screening and recommendation platform that leverages Machine Learning (ML) to predict multiple high-mortality diseases and provide personalized dietary, exercise, and doctor recommendations.

🚀 Features

Predicts six critical diseases:

Heart Disease

Liver Disease

Kidney Disease

Pneumonia

Breast Cancer

Brain Tumor

Diet Recommendation Engine:

Calculates BMI and daily calorie needs

Recommends meal plans based on calorie intake per meal

Personalized Guidance:

Food suggestions

Exercise recommendations

Doctor suggestions

Automated PDF Report Generation with results and recommendations

Database Integration to store and fetch patient history by unique Patient ID

User-Friendly Interface built with Streamlit (frontend) and Flask/FastAPI (backend)


⚙️ Installation & Setup

Clone the repository

git clone https://github.com/<your-username>/mdds.git
cd mdds


Create a virtual environment

python -m venv venv
source venv/bin/activate    # Linux/Mac
venv\Scripts\activate       # Windows


Install dependencies

pip install -r requirements.txt


Run the backend

python app.py   # Flask app


Run the frontend (Streamlit)

streamlit run frontend.py
