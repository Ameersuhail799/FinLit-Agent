# FinLit AI — Institutional Risk & Capital Defense

> **Problem Statement 7 — AI Agent for Digital Financial Literacy**
> Built on **IBM Granite 3.0 (8B Instruct)** · **watsonx.ai RAG** · **Flask Multi-Agent Architecture**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.0-lightgrey?logo=flask)](https://flask.palletsprojects.com)
[![IBM Granite](https://img.shields.io/badge/IBM%20Granite-3.0%208B-0530AD?logo=ibm)](https://www.ibm.com/products/watsonx-ai)
[![watsonx.ai](https://img.shields.io/badge/watsonx.ai-RAG%20Pipeline-6929C4?logo=ibm)](https://www.ibm.com/watsonx)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## Overview

FinLit AI is a production-grade, multi-agent financial literacy platform designed to protect and educate Indian digital consumers. It combines **retrieval-augmented generation (RAG)** over a curated RBI/NPCI knowledge base with **IBM Granite 3.0** for conversational guidance, a **14-rule heuristic scam detector**, and a **50/30/20 capital allocation engine** — all surfaced through an institutional-grade dark dashboard UI.

**Live Repository:** [https://github.com/Ameersuhail799/FinLit-Agent](https://github.com/Ameersuhail799/FinLit-Agent)

---

## Multi-Agent Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│                    VernacularResponse Agent                     │
│              (Orchestrator — IBM Granite 3.0 8B)               │
│                                                                 │
│  Intent Detection  ──────────────────────────────────────────  │
│       │                    │                    │               │
│       ▼                    ▼                    ▼               │
│ ┌───────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│ │  Knowledge    │  │  ScamGuard   │  │   BudgetPlanner      │  │
│ │  Retrieval    │  │   Agent      │  │   Agent              │  │
│ │   Agent       │  │              │  │                      │  │
│ │               │  │ 14-Rule      │  │ 50/30/20 Framework   │  │
│ │ TF-IDF RAG    │  │ Heuristic    │  │ EMI Amortisation     │  │
│ │ Top-K=3       │  │ Risk Scoring │  │ Emergency Fund       │  │
│ │ Cosine≥0.05   │  │ 0–100 Score  │  │ Health Score         │  │
│ └───────────────┘  └──────────────┘  └──────────────────────┘  │
│       │                    │                    │               │
│       └────────────────────┴────────────────────┘              │
│                            │                                   │
│                    Granite Generates                           │
│                  Contextual Response                           │
└─────────────────────────────────────────────────────────────────┘
                             │
                             ▼
                    Structured JSON Response
             { response, intent, context_used,
               scam_analysis?, chart_data? }
```

### Agent Descriptions

| Agent | Role | Technology |
|---|---|---|
| **KnowledgeRetrievalAgent** | Chunks `financial_literacy_kb.md` into passages, builds a TF-IDF matrix at init, retrieves top-3 relevant passages per query via cosine similarity | `scikit-learn` TF-IDF + cosine similarity |
| **ScamGuardAgent** | Evaluates any text against 14 weighted heuristic rules (fake APKs, QR traps, OTP harvesting, digital arrest, escrow fraud). Returns `risk_level`, `risk_score` (0–100), `warnings[]`, and `action_advice` | Pure Python regex scoring |
| **BudgetPlannerAgent** | Accepts monthly income + optional expense buckets, computes exact INR allocations for Needs/Wants/Savings, 3- and 6-month emergency funds, SIP/PPF investment split, EMI amortisation, and a financial health score | Amortisation formula + heuristics |
| **VernacularResponseAgent** | Orchestrates all three agents, builds a structured prompt, and calls IBM Granite via `ibm-watsonx-ai` SDK. Falls back to deterministic rule-based responses when credentials are not configured | `ibm-watsonx-ai` SDK |

---

## Core Features

### Advisory Console (Tab 1)
- Real-time conversational interface powered by IBM Granite 3.0
- **Agent Observability Rail** — live IDLE / ACTIVE / DONE / ERROR badges per agent
- Quick-prompt pills: UPI Scam SMS, 50/30/20 Budget, Safe Loan Rate, Digital Arrest
- RAG context: TF-IDF retrieval from RBI/NPCI knowledge base
- Reference chips linking to NPCI, RBI, and cybercrime.gov.in

### Threat Detection Engine (Tab 2)
- Paste any suspicious SMS, WhatsApp message, or email
- Animated SVG half-gauge (0–100 risk score with colour gradient)
- Numbered threat vector cards per triggered heuristic rule
- Escalation protocol with national helpline (1930) and cybercrime.gov.in links
- Sample vectors: Digital Arrest, Malicious APK, QR Reverse Trap

### Capital Allocator (Tab 3)
- Monthly income input with 4 income presets (30K / 60K / 1L / 2.5L)
- Optional fixed liabilities accordion (housing, utilities, groceries, EMIs, dining, entertainment)
- **Chart.js donut chart** — live 50% Needs / 30% Wants / 20% Savings breakdown
- Emergency runway (3-month + 6-month targets), SIP/PPF split, surplus calculation
- Loan amortisation matrix (12- and 24-month EMIs at 14% p.a.)
- Financial health score (0–100) with coloured progress bar

---

## Tech Stack

| Layer | Technology |
|---|---|
| **LLM** | IBM Granite 3.0 8B Instruct (`ibm/granite-3-8b-instruct`) via watsonx.ai |
| **RAG** | TF-IDF vectorisation + cosine similarity (`scikit-learn`) |
| **Backend** | Python 3.10+ / Flask 3.0 |
| **Frontend** | Vanilla JS (ES2022) + Chart.js 4.4 + inline SVG sprite |
| **Styling** | Pure CSS — 8px grid design system, dark `#090D16` palette |
| **Config** | `python-dotenv` |
| **Knowledge Base** | `financial_literacy_kb.md` — UPI security, cybercrime threats, budgeting, CIBIL |

---

## Local Installation

### Prerequisites
- Python 3.10 or higher
- An IBM Cloud account with watsonx.ai access (optional — falls back to local rules without credentials)

### 1 — Clone the repository

```bash
git clone https://github.com/Ameersuhail799/FinLit-Agent.git
cd FinLit-Agent
```

### 2 — Create and activate a virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### 4 — Configure environment variables

```bash
cp .env.example .env
```

Open `.env` and fill in your IBM watsonx.ai credentials:

```ini
WATSONX_APIKEY=your_ibm_cloud_api_key_here
WATSONX_PROJECT_ID=your_watsonx_project_id_here
WATSONX_URL=https://us-south.ml.cloud.ibm.com
MODEL_ID=ibm/granite-3-8b-instruct
FLASK_SECRET_KEY=change-this-in-production
```

> **Note:** If credentials are left as placeholders, the app runs in **local fallback mode** — all four agents still work using deterministic rule-based responses.

### 5 — Start the server

```bash
python app.py
```

Visit [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/chat` | Multi-agent conversational endpoint |
| `POST` | `/api/analyze-scam` | Dedicated scam forensic analyser |
| `POST` | `/api/calculate-budget` | 50/30/20 budget + EMI computation |
| `GET` | `/api/health` | System health, KB chunk count, watsonx connection status |

### Example — Scam Analysis

```bash
curl -X POST http://127.0.0.1:5000/api/analyze-scam \
  -H "Content-Type: application/json" \
  -d '{"text": "Install SBI_Rewards.apk and enter OTP to claim cashback"}'
```

```json
{
  "risk_level": "High",
  "risk_score": 93,
  "warnings": [
    "Suspicious APK / side-loaded app installation requested",
    "OTP harvesting attempt detected"
  ],
  "action_advice": "[ALERT: HIGH RISK] STOP — Do NOT install. Report to 1930 / cybercrime.gov.in"
}
```

---

## Project Structure

```
FinLit-Agent/
├── app.py                    # Flask app + all 4 agent classes
├── financial_literacy_kb.md  # RAG knowledge base (RBI/NPCI/cybercrime)
├── requirements.txt
├── .env.example              # Environment variable template
├── .gitignore
├── README.md
├── templates/
│   └── index.html            # Single-page institutional dashboard
└── static/
    ├── style.css             # 8px-grid dark design system
    └── app.js                # Tab navigation, chat, gauge, Chart.js
```

---

## Compliance & Governance

- All responses cite **RBI** and **NPCI** framework guidelines
- Scam escalation always references **National Cybercrime Helpline: 1930** and **cybercrime.gov.in**
- IBM Granite hallucination checks enforced via structured prompt engineering
- No user data is persisted — all processing is stateless per request

---

## Problem Statement 7 — Alignment

| Requirement | Implementation |
|---|---|
| AI Agent for Digital Financial Literacy | 4-agent RAG pipeline with IBM Granite 3.0 |
| Scam / fraud detection | ScamGuard — 14 heuristic rules, risk score 0–100 |
| Budget planning | BudgetPlanner — 50/30/20, EMI, emergency fund |
| Vernacular / simplified output | VernacularResponseAgent with jargon explanations |
| IBM Granite on watsonx.ai | `ibm-watsonx-ai` SDK, `ibm/granite-3-8b-instruct` |
| RAG pipeline | TF-IDF + cosine retrieval over financial KB |
| Interactive UI | Three-tab dashboard — Advisory, Threat Detection, Capital Allocator |

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

*FinLit AI — Problem Statement 7 — IBM Granite on watsonx.ai · Built for Digital India*
