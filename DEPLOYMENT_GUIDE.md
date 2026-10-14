# Deployment Guide for Heart Disease ML Web App

This guide explains how to run the web app locally and how to deploy it to get a **free public URL link** that anyone can access on the internet.

---

## 1. Running the App Locally

To run the web app on your computer:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch Streamlit
streamlit run app.py
```

The app will open automatically in your browser at:
**`http://localhost:8501`**

---

## 2. Deploy to Streamlit Community Cloud (Recommended - Free & 1-Click)

Streamlit Community Cloud is 100% free and gives you a public link like:
`https://heart-disease-ml.streamlit.app` (or custom name).

### Step-by-Step Instructions:

1. **Commit and Push the app files to your GitHub repository:**
   In your terminal or Git bash, run:
   ```bash
   git add app.py requirements.txt .streamlit/ test_app_logic.py DEPLOYMENT_GUIDE.md README.md
   git commit -m "Add interactive Streamlit web application and deployment config"
   git push origin main
   ```

2. **Open Streamlit Community Cloud:**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click **Sign in with GitHub** (use your GitHub account: `deekshithakarnati`).

3. **Deploy App:**
   - Click **Create App** (or **New app**).
   - Fill in:
     - **Repository:** `deekshithakarnati/Heart-Disease-ML`
     - **Branch:** `main`
     - **Main file path:** `app.py`
     - **App URL (optional):** Choose your custom URL, e.g., `heart-disease-risk-prediction`
   - Click **Deploy!**

4. **Your Live Link:**
   Within 1-2 minutes, Streamlit will install the requirements and deploy your app. You will get a permanent public link like:
   👉 **`https://heart-disease-risk-prediction.streamlit.app`**

---

## 3. Alternative: Deploy to Hugging Face Spaces (Free)

Hugging Face Spaces is another free hosting option for Machine Learning apps:

1. Go to [huggingface.co/spaces](https://huggingface.co/spaces) and sign up / log in.
2. Click **Create new Space**.
3. Choose:
   - **Space SDK:** `Streamlit`
   - **License:** `Open Source (e.g. MIT)`
   - **Space Hardware:** `Free (CPU basic)`
4. Either clone your GitHub repo into the Space or push the files (`app.py`, `heart_disease_model.pkl`, `requirements.txt`).
5. Your Space will be live instantly at:
   👉 **`https://huggingface.co/spaces/deekshithakarnati/heart-disease-ml`**

---

## 4. Features Included in the Web App

- 🩺 **Full 25-Feature Clinical Input:** All clinical vitals, lab results, and lifestyle parameters.
- 🎯 **3 Quick Preset Patient Profiles:** One-click load for Healthy, Borderline, and High Risk archetypes.
- 📊 **Model & Feature Insights Tab:** Feature importance chart and performance metrics (88.17% accuracy).
- 🔍 **Clinical Factor Analysis:** Automatically flags abnormal values like high blood pressure, LDL, and low max heart rate.
- 📱 **Mobile & Desktop Responsive:** Works smoothly on phones, tablets, and laptops.
