"""
FinLit-Agent: AI Agent for Digital Financial Literacy
Problem Statement 7 — IBM Granite on watsonx.ai + RAG
"""

import os
import re
import json
import math
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import requests
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------
app = Flask(__name__)

# ---------------------------------------------------------------------------
# IBM watsonx.ai — pure requests implementation
# No third-party SDK required; works with only the `requests` package.
# ---------------------------------------------------------------------------

_WX_API_KEY     = os.getenv("IBM_CLOUD_API_KEY", "")
_WX_PROJECT_ID  = os.getenv("WATSONX_PROJECT_ID", "")
_WX_URL         = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com").rstrip("/")
_WX_MODEL_ID    = os.getenv("MODEL_ID", "ibm/granite-3-8b-instruct")

_IAM_TOKEN_URL  = "https://iam.cloud.ibm.com/identity/token"
_INFER_URL      = f"{_WX_URL}/ml/v1/text/generation?version=2023-05-29"

# Cached IAM token state
_iam_token: Optional[str] = None
_iam_token_expiry: float = 0.0


def _get_iam_token() -> Optional[str]:
    """Exchange the IBM Cloud API key for a short-lived IAM Bearer token.
    Automatically refreshes when the cached token is within 60 s of expiry.
    Returns None and logs a warning if the exchange fails.
    """
    global _iam_token, _iam_token_expiry

    if not _WX_API_KEY or _WX_API_KEY == "your_ibm_cloud_api_key_here":
        return None

    # Return cached token if still valid
    if _iam_token and time.time() < _iam_token_expiry - 60:
        return _iam_token

    try:
        resp = requests.post(
            _IAM_TOKEN_URL,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data={
                "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
                "apikey": _WX_API_KEY,
            },
            timeout=15,
        )
        resp.raise_for_status()
        payload = resp.json()
        _iam_token = payload["access_token"]
        # IBM tokens expire in 3600 s; store the absolute expiry timestamp
        _iam_token_expiry = time.time() + int(payload.get("expires_in", 3600))
        logger.info("[watsonx.ai] IAM token acquired (expires in %ds).",
                    int(payload.get("expires_in", 3600)))
        return _iam_token
    except Exception as exc:
        logger.warning("[watsonx.ai] IAM token exchange failed: %s", exc)
        _iam_token = None
        return None


def _watsonx_generate(prompt: str) -> Optional[str]:
    """Send a generation request to the watsonx.ai REST endpoint.
    Returns the generated text string, or None on failure.
    """
    token = _get_iam_token()
    if not token or not _WX_PROJECT_ID:
        return None

    if _WX_PROJECT_ID == "your_watsonx_project_id_here":
        return None

    payload = {
        "model_id": _WX_MODEL_ID,
        "input": prompt,
        "parameters": {
            "decoding_method": "greedy",
            "temperature": 0.2,
            "repetition_penalty": 1.1,
            "max_new_tokens": 400,
            "stop_sequences": [],
        },
        "project_id": _WX_PROJECT_ID,
    }

    try:
        resp = requests.post(
            _INFER_URL,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        generated = (
            data.get("results", [{}])[0]
            .get("generated_text", "")
            .strip()
        )
        if generated:
            logger.info("[watsonx.ai] Successfully generated response with Granite 3.0")
        return generated or None
    except Exception as exc:
        logger.warning("[watsonx.ai] Inference request failed: %s", exc)
        return None


def _watsonx_ready() -> bool:
    """Returns True when real credentials are present (not placeholders)."""
    return bool(
        _WX_API_KEY
        and _WX_API_KEY != "your_ibm_cloud_api_key_here"
        and _WX_PROJECT_ID
        and _WX_PROJECT_ID != "your_watsonx_project_id_here"
    )


# Log startup status
if _watsonx_ready():
    logger.info("[watsonx.ai] Credentials loaded — Granite 3.0 active on %s", _WX_URL)
else:
    logger.warning("[watsonx.ai] No credentials — running in local-fallback mode.")

# ===========================================================================
# AGENT 1: KnowledgeRetrievalAgent
# ===========================================================================
class KnowledgeRetrievalAgent:
    """Chunks financial_literacy_kb.md and retrieves relevant passages via TF-IDF."""

    KB_PATH = Path("financial_literacy_kb.md")
    TOP_K = 3

    def __init__(self):
        self._chunks: List[str] = []
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._matrix = None
        self._load()

    def _load(self):
        if not self.KB_PATH.exists():
            logger.warning("KB file not found: %s", self.KB_PATH)
            return
        text = self.KB_PATH.read_text(encoding="utf-8")
        self._chunks = self._chunk(text)
        self._vectorizer = TfidfVectorizer(stop_words="english")
        self._matrix = self._vectorizer.fit_transform(self._chunks)
        logger.info("KnowledgeRetrievalAgent: indexed %d chunks.", len(self._chunks))

    def _chunk(self, text: str) -> List[str]:
        """Split on '##' section headers, keeping each section as one chunk."""
        sections = re.split(r"(?=^## )", text, flags=re.MULTILINE)
        chunks = [s.strip() for s in sections if s.strip()]
        # Also add bullet-level sub-chunks for finer granularity
        sub = []
        for chunk in chunks:
            bullets = re.split(r"\n- ", chunk)
            if len(bullets) > 1:
                header = bullets[0]
                for b in bullets[1:]:
                    sub.append(f"{header}\n- {b.strip()}")
        return chunks + sub

    def retrieve(self, query: str) -> List[str]:
        if not self._vectorizer or self._matrix is None:
            return []
        q_vec = self._vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self._matrix).flatten()
        top_idx = np.argsort(sims)[::-1][: self.TOP_K]
        return [self._chunks[i] for i in top_idx if sims[i] > 0.05]


# ===========================================================================
# AGENT 2: ScamGuardAgent
# ===========================================================================
class ScamGuardAgent:
    """Detects financial scam patterns and returns a structured threat payload."""

    # Weighted red-flag rules: (regex_pattern, score_contribution, warning_text)
    # Weights are intentionally high so compound threats score into "High" reliably.
    RULES = [
        # Tier 1 — Critical (single match can indicate High threat)
        (r"upi.?pin.*receiv|receive.*upi.?pin|scan.*qr.*(refund|cashback|receiv|credit)|qr.*(get|receiv).*(money|cash|refund)",
         80, "Reverse UPI/QR Trap — Scanning a QR code or entering your UPI PIN NEVER receives money. This is a textbook scam."),
        (r"digital.?arrest|cbi.*officer|cyber.?police.*call|customs.*(parcel|package|narcotics)|enforcement.*video",
         75, "Digital Arrest Impersonation — Genuine CBI/Police/Customs NEVER conduct arrests or proceedings via video call. This is a fraud."),
        (r"\.apk\b|download.*app.*link|install.*apk|sideload|sbi.?reward.*apk|quicksupport.*apk",
         60, "Malicious APK Download — Installing sideloaded APKs from unknown sources can silently steal your OTPs and banking credentials."),
        # Tier 2 — High-risk signals
        (r"otp|one.?time.?password",
         30, "OTP Solicitation — Legitimate banks, UPI apps, and government agencies NEVER ask for your OTP via call, SMS, or chat."),
        (r"upi.?pin|enter.*pin|verify.*pin|share.*pin",
         30, "UPI PIN Solicitation — Your UPI PIN is a debit-only key. No refund, cashback, or credit transaction ever requires it."),
        (r"urgent|immediately|tonight|expires|last.?chance|within.*hour|action.?required|suspended.?now",
         25, "High-Pressure Urgency Tactic — Artificial urgency is a classic manipulation technique to bypass your caution."),
        (r"kyc.*verif|verify.*account|update.*kyc|pan.*card.*verif|aadhaar.*verif",
         25, "Unsolicited KYC Demand — Banks send KYC requests only through official apps or branches, never via SMS links or calls."),
        (r"electricity.*bill|power.*cut|connection.*block|ebill|wbsedcl|bescom|discom",
         20, "Utility Bill Impersonation — Fraudsters pose as electricity boards to push fake payment links or APK installs."),
        (r"anydesk|teamviewer|screen.?shar|remote.*access|remote.*desktop",
         22, "Screen-Sharing Request — Granting remote access allows attackers to capture your banking credentials in real time."),
        (r"escrow|safe.?account|secure.?transfer|government.*wallet",
         20, "Fake Escrow/Safe Account — No legitimate agency, court, or bank transfers funds to an 'escrow' via chat or call."),
        # Tier 3 — Supporting signals
        (r"refund.*click|cashback.*link|prize.*claim|lottery.*won",
         18, "Fake Refund/Prize Link — Clicking these links leads to phishing pages or APK downloads designed to steal credentials."),
        (r"account.*block|sim.*deactivat|service.*suspend|number.*cancel",
         15, "Account/SIM Suspension Threat — A common urgency lure; always verify through the official operator or bank helpline."),
        (r"loan.*instant.*approv|0%.?interest|guaranteed.*return|double.*money",
         15, "Predatory Lending/Investment Red Flag — Guaranteed returns and instant approvals violate RBI regulations."),
        (r"whatsapp.*officer|telegram.*support|video.?call.*official",
         12, "Unofficial Communication Channel — Regulated institutions do not conduct official business via WhatsApp or Telegram."),
    ]

    # Escalation advice templates — NO emojis; plain ASCII markers only
    _ESCALATION = (
        "[ACTION] Call **National Cybercrime Helpline 1930** immediately.\n"
        "[ACTION] File a report at **cybercrime.gov.in** within the first 2 hours "
        "('Golden Period') to freeze fraudulent transfers.\n"
        "[RULE] Do NOT share any OTP, UPI PIN, Aadhaar, or bank details with anyone."
    )

    def analyze(self, text: str) -> Dict[str, Any]:
        text_lower = text.lower()
        raw_score = 0
        warnings: List[str] = []

        for pattern, weight, warning in self.RULES:
            if re.search(pattern, text_lower):
                raw_score += weight
                warnings.append(warning)

        num_warnings = len(warnings)

        # Determine risk level using both score and warning count
        if raw_score >= 60 or num_warnings >= 2:
            risk_level = "High"
            # Normalise into the 85–98 band; more signals push score higher
            risk_score = min(85 + (num_warnings * 2) + (raw_score // 20), 98)
        elif raw_score >= 30 or num_warnings == 1:
            risk_level = "Medium"
            # Normalise into the 50–70 band
            risk_score = min(50 + (num_warnings * 5) + (raw_score // 10), 70)
        else:
            risk_level = "Safe"
            risk_score = min(raw_score, 24)

        # Action advice — never produce a "no indicators" message when warnings exist
        if risk_level == "High":
            action_advice = (
                "[ALERT: HIGH RISK] **STOP — This message shows strong scam indicators. DO NOT proceed.**\n\n"
                + self._ESCALATION
            )
        elif risk_level == "Medium":
            action_advice = (
                "[ALERT: MEDIUM RISK] **Caution — Suspicious signals detected. "
                "Treat this message with scepticism.**\n\n"
                "Do NOT click any links, install any apps, or share credentials.\n"
                "Verify the sender's identity through the official bank/operator helpline.\n\n"
                + self._ESCALATION
            )
        else:
            # Safe — no warnings, genuinely clean text
            action_advice = (
                "[VERIFIED: SAFE] No scam indicators found in this message.\n"
                "Stay vigilant: never share OTPs, UPI PINs, or Aadhaar details with anyone, "
                "even if they claim to be from your bank or a government agency."
            )

        return {
            "risk_level": risk_level,
            "risk_score": risk_score,
            "warnings": warnings,
            "action_advice": action_advice,
        }


# ===========================================================================
# AGENT 3: BudgetPlannerAgent
# ===========================================================================
class BudgetPlannerAgent:
    """Generates 50/30/20 budgets and EMI breakdowns."""

    def calculate(self, income: float, expenses: Dict[str, float]) -> Dict[str, Any]:
        needs_target = income * 0.50
        wants_target = income * 0.30
        savings_target = income * 0.20

        total_expenses = sum(expenses.values()) if expenses else 0
        existing_emis = expenses.get("emis", 0)

        # Adjust needs for declared EMIs
        actual_needs = max(expenses.get("housing", 0) + expenses.get("utilities", 0)
                           + expenses.get("groceries", 0) + existing_emis, 0)
        actual_wants = max(expenses.get("dining", 0) + expenses.get("entertainment", 0)
                           + expenses.get("subscriptions", 0), 0)
        actual_savings = max(income - actual_needs - actual_wants, 0)

        emergency_fund = savings_target * 3  # 3-month target
        sip_suggestion = savings_target * 0.60  # 60% of savings bucket into SIP
        ppf_suggestion = savings_target * 0.40  # 40% into PPF

        # Simple EMI formula: EMI = P * r * (1+r)^n / ((1+r)^n - 1)
        def emi_calc(principal: float, annual_rate: float, months: int) -> float:
            if annual_rate == 0:
                return principal / months
            r = annual_rate / 12 / 100
            return principal * r * (1 + r) ** months / ((1 + r) ** months - 1)

        sample_loan = income * 2  # 2x monthly as sample personal loan
        sample_emi_12 = emi_calc(sample_loan, 14.0, 12)
        sample_emi_24 = emi_calc(sample_loan, 14.0, 24)

        surplus = income - total_expenses if total_expenses else savings_target
        health_score = self._health_score(income, actual_needs, actual_wants, actual_savings)

        return {
            "income": income,
            "allocations": {
                "needs": {"target": round(needs_target, 2), "actual": round(actual_needs, 2), "percent": 50},
                "wants": {"target": round(wants_target, 2), "actual": round(actual_wants, 2), "percent": 30},
                "savings": {"target": round(savings_target, 2), "actual": round(actual_savings, 2), "percent": 20},
            },
            "emergency_fund": {
                "target_3_months": round(emergency_fund, 2),
                "target_6_months": round(emergency_fund * 2, 2),
                "recommended_instrument": "High-interest savings account or Liquid Mutual Fund",
            },
            "investment_split": {
                "sip_mutual_fund": round(sip_suggestion, 2),
                "ppf": round(ppf_suggestion, 2),
            },
            "emi_examples": {
                "loan_amount": round(sample_loan, 2),
                "rate_pa": 14.0,
                "emi_12_months": round(sample_emi_12, 2),
                "emi_24_months": round(sample_emi_24, 2),
            },
            "surplus": round(surplus, 2),
            "financial_health_score": health_score,
            "chart_data": {
                "labels": ["Needs (50%)", "Wants (30%)", "Savings (20%)"],
                "values": [round(needs_target, 2), round(wants_target, 2), round(savings_target, 2)],
                "colors": ["#3b82f6", "#f59e0b", "#10b981"],
            },
        }

    def _health_score(self, income, needs, wants, savings) -> int:
        score = 100
        if needs > income * 0.55:
            score -= 20
        if wants > income * 0.35:
            score -= 20
        if savings < income * 0.10:
            score -= 30
        elif savings < income * 0.20:
            score -= 10
        return max(score, 0)


# ===========================================================================
# AGENT 4: VernacularResponseAgent (Orchestrator)
# ===========================================================================
class VernacularResponseAgent:
    """Orchestrates multi-agent retrieval + IBM Granite generation (or local fallback)."""

    SYSTEM_PROMPT = (
        "You are FinLit-Advisor, an expert AI assistant for digital financial literacy in India. "
        "You help users understand UPI safety, scam prevention, budgeting, loans, and investment basics. "
        "Always explain financial jargon in simple terms. Keep responses concise, factual, and actionable. "
        "Use ₹ for rupees and reference RBI/NPCI guidelines where relevant."
    )

    def __init__(self, kb_agent: KnowledgeRetrievalAgent, scam_agent: ScamGuardAgent, budget_agent: BudgetPlannerAgent):
        self.kb = kb_agent
        self.scam = scam_agent
        self.budget = budget_agent

    def respond(self, user_message: str) -> Dict[str, Any]:
        intent = self._detect_intent(user_message)
        context_chunks = self.kb.retrieve(user_message)
        context = "\n\n".join(context_chunks)

        prompt = self._build_prompt(user_message, context, intent)
        response_text = self._generate(prompt)

        # Attach structured side-data for certain intents
        side_data: Dict[str, Any] = {"intent": intent}
        if intent == "scam_check":
            side_data["scam_analysis"] = self.scam.analyze(user_message)

        return {"response": response_text, "context_used": len(context_chunks) > 0, **side_data}

    # Scam-related keywords that must ALWAYS route to scam_check — checked first,
    # before any budget/EMI keyword matching, to prevent misrouting.
    _SCAM_KEYWORDS = [
        "scam", "fraud", "fake", "otp", "apk", "phishing", "arrest", "qr code",
        "digital arrest", "cbi", "cyber police", "customs", "narcotics",
        "anydesk", "teamviewer", "screen share", "sideload", "install app",
        "download link", "kyc verify", "verify account", "upi pin",
        "cashback link", "refund link", "prize claim", "lottery", "escrow",
        "electricity bill scam", "sim deactivat", "account block",
    ]

    # Budget/investment keywords — only matched when no scam signal is present.
    _BUDGET_KEYWORDS = [
        "budget", "50/30/20", "salary", "income", "expense",
        "saving", "investment", "sip", "ppf", "mutual fund",
    ]

    def _detect_intent(self, text: str) -> str:
        text_lower = text.lower()

        # Priority 1: scam signals — evaluated BEFORE any financial keyword checks
        if any(kw in text_lower for kw in self._SCAM_KEYWORDS):
            return "scam_check"

        # Priority 2: run ScamGuardAgent's own pattern engine as a secondary gate;
        # if it fires even one rule the query is scam-related, not budgeting.
        quick = self.scam.analyze(text)
        if quick["risk_level"] != "Safe":
            return "scam_check"

        # Priority 3: budget / investment (only when no scam signal)
        if any(kw in text_lower for kw in self._BUDGET_KEYWORDS):
            return "budget"

        # Priority 4: loan / credit
        if any(w in text_lower for w in ["loan", "interest", "cibil", "credit score", "emi rate"]):
            return "loan_advice"

        # Priority 5: UPI payment safety (non-scam generic questions)
        if any(w in text_lower for w in ["upi", "pin", "payment", "transfer"]):
            return "upi_safety"

        return "general"

    def _build_prompt(self, user_msg: str, context: str, intent: str) -> str:
        parts = [self.SYSTEM_PROMPT]
        if context:
            parts.append(f"\n[Knowledge Base Context]\n{context}")
        parts.append(f"\n[User Question]\n{user_msg}")
        parts.append("\n[Answer]")
        return "\n".join(parts)

    def _generate(self, prompt: str) -> str:
        """Try watsonx.ai REST inference first; fall back to local rules on failure."""
        result = _watsonx_generate(prompt)
        if result:
            return result
        return self._local_fallback(prompt)

    def _local_fallback(self, prompt: str) -> str:
        """Rule-based fallback when IBM credentials are not configured.

        Intent is already resolved before this method is called, so we check
        the *intent* embedded in the prompt context rather than re-parsing
        raw financial keywords that could overlap with scam texts.
        """
        lower = prompt.lower()

        # ── Scam-related fallbacks (checked FIRST, highest priority) ──────────
        scam_signals = [
            "digital arrest", "cbi", "cyber police", "customs", "narcotics",
            "apk", "anydesk", "teamviewer", "escrow", "scam", "fraud",
            "otp", "upi pin", "kyc", "phish", "electricity bill scam",
            "qr.*receiv", "scan.*get.*money",
        ]
        if any(re.search(sig, lower) for sig in scam_signals):
            # Run the scam engine on the original user message (last line after [User Question])
            user_msg_match = re.search(r"\[User Question\]\n(.+)", prompt, re.DOTALL)
            if user_msg_match:
                user_text = user_msg_match.group(1).split("\n[Answer]")[0].strip()
                analysis = self.scam.analyze(user_text)
                risk = analysis["risk_level"]
                score = analysis["risk_score"]
                warns = analysis["warnings"]
                warn_text = "\n".join(f"• {w}" for w in warns) if warns else ""
                if risk == "High":
                    return (
                        f"[ALERT: HIGH RISK] **Score: {score}/100**\n\n"
                        + (f"**Red Flags Found:**\n{warn_text}\n\n" if warn_text else "")
                        + "**Immediate actions required:**\n"
                        "• Do NOT click any link, install any app, or share any credentials.\n"
                        "• Block the sender on all platforms immediately.\n"
                        "• Call **National Cybercrime Helpline: 1930** to report and freeze any transfers.\n"
                        "• File a complaint at **cybercrime.gov.in** within 2 hours."
                    )
                elif risk == "Medium":
                    return (
                        f"[ALERT: MEDIUM RISK] **Score: {score}/100**\n\n"
                        + (f"**Suspicious signals:**\n{warn_text}\n\n" if warn_text else "")
                        + "• Do not click links or install apps from this message.\n"
                        "• Verify by calling the official bank/operator helpline directly.\n"
                        "• If in doubt, report to **1930** or **cybercrime.gov.in**."
                    )

            # Generic scam safety (fallback if regex extraction fails)
            return (
                "**Scam Safety Advisory:**\n"
                "• Genuine agencies — banks, CBI, police, electricity boards — NEVER ask for OTPs, "
                "UPI PINs, or Aadhaar via call, SMS, or chat.\n"
                "• QR codes and UPI links only *send* money — they cannot receive it.\n"
                "• Real enforcement agencies never conduct arrests via WhatsApp or Skype video.\n\n"
                "**Report immediately:** Call **1930** or visit **cybercrime.gov.in**."
            )

        # ── UPI safety (non-scam generic questions) ───────────────────────────
        if "upi" in lower and "pin" in lower:
            return (
                "Your **UPI PIN** is a debit-only authentication key. "
                "You should **never** enter it to receive money, claim a refund, or verify your account. "
                "Any request for your UPI PIN is a scam — block the sender and dial **1930** immediately."
            )

        # ── Budget / savings (only reached when no scam signal matches above) ──
        if "50/30/20" in lower or "budget" in lower:
            return (
                "The **50/30/20 Rule** is a simple budgeting framework:\n"
                "• **50% Needs** — rent, EMIs, groceries, utilities, health insurance.\n"
                "• **30% Wants** — dining out, travel, subscriptions, entertainment.\n"
                "• **20% Savings** — emergency fund (3–6 months of expenses), SIP mutual funds, PPF.\n\n"
                "Use the **Budget Planner** tab to compute your exact ₹ allocations."
            )

        # ── Loans / credit ────────────────────────────────────────────────────
        if "loan" in lower or "interest" in lower:
            return (
                "**Safe personal loan rates** from regulated banks: **10.5% – 18% APR**. "
                "Anything above 36% APR is predatory. "
                "Avoid apps that demand contact-list permissions — this violates RBI guidelines. "
                "Check your **CIBIL score** (aim for 750+) to access the best rates."
            )
        if "cibil" in lower or "credit score" in lower:
            return (
                "A **CIBIL score** ranges from 300 to 900. "
                "750+ gets you prime loan rates and faster approvals. "
                "Improve it by paying EMIs on time, keeping credit utilisation below 30%, "
                "and avoiding multiple simultaneous loan applications."
            )

        return (
            "I'm your **FinLit-Advisor**. I can help you with:\n"
            "• **UPI & Payment Safety** — protecting your transactions\n"
            "• **Scam Detection** — identifying fraud patterns\n"
            "• **Budget Planning** — 50/30/20 rule for your salary\n"
            "• **Loan & Credit** — understanding interest rates and CIBIL\n\n"
            "Please ask a specific question and I'll guide you!"
        )


# ===========================================================================
# Initialise all agents
# ===========================================================================
kb_agent = KnowledgeRetrievalAgent()
scam_agent = ScamGuardAgent()
budget_agent = BudgetPlannerAgent()
orchestrator = VernacularResponseAgent(kb_agent, scam_agent, budget_agent)

# ===========================================================================
# Flask Routes
# ===========================================================================

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    user_message = (data.get("message") or "").strip()
    if not user_message:
        return jsonify({"error": "message is required"}), 400

    try:
        result = orchestrator.respond(user_message)
        return jsonify(result)
    except Exception as exc:
        logger.exception("Chat error")
        return jsonify({"error": str(exc)}), 500


@app.route("/api/analyze-scam", methods=["POST"])
def analyze_scam():
    data = request.get_json(force=True)
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "text is required"}), 400

    try:
        analysis = scam_agent.analyze(text)
        return jsonify(analysis)
    except Exception as exc:
        logger.exception("Scam analysis error")
        return jsonify({"error": str(exc)}), 500


@app.route("/api/calculate-budget", methods=["POST"])
def calculate_budget():
    data = request.get_json(force=True)
    try:
        income = float(data.get("income", 0))
        if income <= 0:
            return jsonify({"error": "income must be a positive number"}), 400

        expenses = {
            "housing": float(data.get("housing", 0)),
            "utilities": float(data.get("utilities", 0)),
            "groceries": float(data.get("groceries", 0)),
            "emis": float(data.get("emis", 0)),
            "dining": float(data.get("dining", 0)),
            "entertainment": float(data.get("entertainment", 0)),
            "subscriptions": float(data.get("subscriptions", 0)),
        }

        result = budget_agent.calculate(income, expenses)
        return jsonify(result)
    except (ValueError, TypeError) as exc:
        return jsonify({"error": f"Invalid numeric input: {exc}"}), 400
    except Exception as exc:
        logger.exception("Budget calculation error")
        return jsonify({"error": str(exc)}), 500


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model": _WX_MODEL_ID,
        "watsonx_connected": _watsonx_ready(),
        "kb_chunks": len(kb_agent._chunks),
    })


# ===========================================================================
# Entry point
# ===========================================================================
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "False").lower() in ("true", "1")
    app.run(host="0.0.0.0", port=port, debug=debug)
