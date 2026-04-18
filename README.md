# FairLens AI - Unbiased Decision Dashboard

**Hack2Skill Solution Challenge 2026 | Unbiased AI Decision Track**

An interactive Streamlit dashboard that detects and mitigates gender bias in a loan approval ML model — with live predictions, fairness metrics, bias toggle, and real-time explainability.

---

## Quick Start (Local)

```bash
pip install -r requirements.txt
python data/generate_dataset.py
python model/train.py
streamlit run app.py
```

Open: http://localhost:8501

---

## Deploy to Google Cloud Run

### Prerequisites
- Google Cloud project with billing enabled
- `gcloud` CLI authenticated

### Steps (run in Cloud Shell or terminal)

```bash
# 1. Clone the repo
git clone https://github.com/Gowtham280103/aiapp.git
cd aiapp

# 2. Set your project
gcloud config set project gen-lang-client-0903375444

# 3. Enable required APIs
gcloud services enable run.googleapis.com cloudbuild.googleapis.com

# 4. Build and deploy in one command
gcloud run deploy fairlensai \
  --source . \
  --region us-central1 \
  --platform managed \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --timeout 300 \
  --port 8080
```

Cloud Run will build the Docker image, train the models, and give you a public HTTPS URL.

---

## Deploy to Streamlit Cloud (Easiest)

1. Push this repo to GitHub
2. Go to https://share.streamlit.io
3. Click **New app** → select `Gowtham280103/aisolution` → `app.py`
4. Click **Deploy**

> Note: Add a `setup.sh` startup hook if Streamlit Cloud needs model pre-training (see below).

---

## Project Structure

```
aisolution/
├── app.py                    # Main Streamlit dashboard
├── requirements.txt
├── Dockerfile                # Cloud Run deployment
├── .streamlit/config.toml    # Streamlit server config
├── data/
│   ├── generate_dataset.py   # Synthetic biased loan dataset (1000 rows)
│   └── loan_data.csv         # Auto-generated
└── model/
    ├── train.py              # Trains biased + debiased models
    ├── metrics.py            # Fairness metrics engine
    ├── biased_model.pkl      # Auto-generated
    ├── debiased_model.pkl    # Auto-generated
    └── test_data.csv         # Auto-generated
```

---

## Results

| Metric | Biased Model | Debiased Model | Improvement |
|--------|-------------|----------------|-------------|
| Demographic Parity Diff | 0.382 | 0.082 | **-78%** |
| Equal Opportunity Diff | 0.436 | 0.093 | **-79%** |
| Accuracy Gap | 0.228 | 0.035 | **-85%** |
| Fairness Score | 35/100 | 78/100 | **+43 pts** |

---

## Dashboard Features

- **Bias Toggle** — switch between biased and debiased model live
- **Live Prediction** — real-time loan decision with probability gauges
- **Fairness Score** — 0-100 gauge with badges (Bias Detected / Fair Model Achieved)
- **Confusion Matrices** — per gender group (Female / Male)
- **Feature Influence** — shows gender coefficient drop after debiasing
- **Auto Insights** — dynamic messages based on current model and applicant
- **Before vs After** — side-by-side comparison with improvement progress bars
