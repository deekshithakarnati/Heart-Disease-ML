# Deployment Platform Documentation

This document outlines the deployment platform architecture, technical specifications, selection rationale, and configuration details for the Heart Disease Risk Prediction web application.

---

## 1. Selected Platform Overview

| Category | Specification |
|----------|---------------|
| **Primary Cloud Platform** | **Streamlit Community Cloud** (PaaS / Serverless ML Hosting) |
| **Alternative Cloud Platform** | **Hugging Face Spaces** (Docker / Streamlit SDK) |
| **Local Runtime** | Streamlit Local Server (`http://localhost:8501`) |
| **Repository Integration** | GitHub (`deekshithakarnati/Heart-Disease-ML`) |
| **Branch Deployed** | `main` |
| **Entry Point Script** | `app.py` |
| **Default Port** | `8501` |
| **SSL / HTTPS** | Automatically managed with TLS certificates |
| **Hosting Cost** | 100% Free / Open Source Tier |

---

## 2. Platform Selection Rationale

The project selected **Streamlit Community Cloud** as the primary deployment environment due to the following strategic and technical advantages:

1. **Direct GitHub Synchronization:**
   Streamlit Cloud links directly to the `deekshithakarnati/Heart-Disease-ML` repository. Any code updates or model re-trains pushed to the `main` branch trigger automated zero-downtime re-deployments (Continuous Deployment / CD).

2. **Native Machine Learning Support:**
   Unlike traditional web hosting platforms that require custom API gateways, frontend-backend routing, or heavy JavaScript setups, Streamlit executes native Python code with high-performance C-extensions (`numpy`, `scikit-learn`, `joblib`).

3. **In-Memory Model Caching:**
   Using Streamlit's `@st.cache_resource` decorator, the 11.5 MB `heart_disease_model.pkl` pipeline is loaded into memory only once upon container initialization, ensuring sub-second inference latency for end users.

4. **Production-Ready UI Components:**
   Provides responsive input sliders, numerical spinners, dropdown menus, progress bars, and matplotlib visualization containers out of the box without requiring separate React/Angular codebases.

---

## 3. Platform Architecture Flow

```
+-------------------------------------------------------------+
|                     GitHub Repository                       |
|          deekshithakarnati/Heart-Disease-ML                 |
|  - app.py                  - requirements.txt               |
|  - heart_disease_model.pkl - .streamlit/config.toml         |
+------------------------------+------------------------------+
                               | Git Webhook (Auto CD)
                               v
+-------------------------------------------------------------+
|                 Streamlit Community Cloud                   |
|  1. Spins up isolated Linux container (Debian-based)        |
|  2. Provisions Python runtime environment                   |
|  3. Runs `pip install -r requirements.txt`                  |
|  4. Boots Uvicorn / Tornado web server on port 8501         |
|  5. Loads pipeline via `@st.cache_resource`                 |
+------------------------------+------------------------------+
                               | Secure HTTPS / WebSocket
                               v
+-------------------------------------------------------------+
|                     End-User Browser                        |
|  - Accessible on Desktop, Tablet, and Mobile                |
|  - Live URL: https://<custom-subdomain>.streamlit.app       |
+-------------------------------------------------------------+
```

---

## 4. Environment & Dependency Specifications

All runtime dependencies are explicitly defined in `requirements.txt`:

```text
streamlit>=1.30.0
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
joblib>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

### Server & UI Theme Configuration (`.streamlit/config.toml`)
```toml
[theme]
primaryColor = "#E63946"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F8F9FA"
textColor = "#1D3557"
font = "sans serif"

[server]
headless = true
enableCORS = false
enableXsrfProtection = true
```

---

## 5. Deployment Step-by-Step Guide

### Deploying to Streamlit Community Cloud (Recommended):

1. **Visit Streamlit Cloud:** Navigate to [share.streamlit.io](https://share.streamlit.io).
2. **Authorize GitHub:** Sign in using your GitHub account (`deekshithakarnati`).
3. **Initiate New App:** Click the **"Create app"** button.
4. **Configure Settings:**
   - **Repository:** `deekshithakarnati/Heart-Disease-ML`
   - **Branch:** `main`
   - **Main file path:** `app.py`
   - **App URL:** (Optional) choose a unique custom subdomain (e.g. `heart-disease-risk-prediction`)
5. **Launch:** Click **"Deploy!"**.
6. The platform completes provisioning in ~60 seconds and assigns your public live link.

---

### Alternative: Deploying to Hugging Face Spaces:

1. Navigate to [huggingface.co/spaces](https://huggingface.co/spaces).
2. Click **Create new Space**.
3. Set Space name (e.g., `heart-disease-prediction`).
4. Select **Streamlit** as the Space SDK.
5. Clone or sync the files (`app.py`, `requirements.txt`, `heart_disease_model.pkl`).
6. The app builds automatically and is hosted at `https://huggingface.co/spaces/<user>/<space-name>`.
