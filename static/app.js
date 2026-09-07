/**
 * FinLit AI — app.js
 * Matches: templates/index.html (ws-* IDs, tab-pill classes, i-* SVG sprites)
 * API: /api/chat  /api/analyze-scam  /api/calculate-budget  /api/health
 */

"use strict";

// ===========================================================================
// UTILITIES
// ===========================================================================

/** Escape all HTML special characters before inserting into the DOM. */
function escapeHTML(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

/** Indian-locale number formatter — no emoji, just digits + commas. */
const fmt   = (n) => new Intl.NumberFormat("en-IN", { maximumFractionDigits: 0 }).format(n);
const fmtRs = (n) => "\u20B9" + fmt(n);   // ₹ via unicode — no literal symbol

/** Current time as HH:MM:SS string. */
const nowStr = () => new Date().toLocaleTimeString("en-IN", { hour12: false });

/**
 * Convert a small subset of markdown used by the backend into safe HTML.
 * Bold (**…**), inline code (`…`), and newlines only.
 * All raw text is escaped BEFORE substitution so injected HTML is safe.
 */
function mdToHtml(text) {
  let s = escapeHTML(String(text));
  s = s.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
  s = s.replace(/`([^`]+)`/g, "<code>$1</code>");
  s = s.replace(/\n/g, "<br>");
  return s;
}

/** Build an inline SVG <use> reference — prevents any XSS in id. */
const svg = (id, cls = "") =>
  `<svg aria-hidden="true"${cls ? ` class="${escapeHTML(cls)}"` : ""}><use href="#${escapeHTML(id)}"/></svg>`;

// ===========================================================================
// ERROR TOAST
// ===========================================================================

const errorToast = document.getElementById("errorToast");
let _toastTimer = null;

function showError(message) {
  if (!errorToast) return;
  errorToast.textContent = message;
  errorToast.removeAttribute("hidden");
  clearTimeout(_toastTimer);
  _toastTimer = setTimeout(() => errorToast.setAttribute("hidden", ""), 7000);
}

// ===========================================================================
// HEALTH CHECK — populates telemetry on boot
// ===========================================================================

async function checkHealth() {
  try {
    const res  = await fetch("/api/health");
    const data = await res.json();
    const chunksEl = document.getElementById("kbChunks");
    if (chunksEl) chunksEl.textContent = data.kb_chunks ?? "—";
    const statusEl = document.getElementById("healthStatus");
    if (statusEl) {
      statusEl.textContent = data.watsonx_connected
        ? "Granite 3.0 Connected"
        : "TF-IDF Vector RAG";
    }
  } catch {
    // Non-fatal — telemetry is decorative
  }
}

// ===========================================================================
// BOOT — set init timestamp
// ===========================================================================

const initTimeEl = document.getElementById("initTime");
if (initTimeEl) initTimeEl.textContent = nowStr();

checkHealth();

// ===========================================================================
// TAB NAVIGATION
// ===========================================================================

const tabPills   = document.querySelectorAll(".tab-pill[data-tab]");
const workspaces = document.querySelectorAll(".workspace");

function activateTab(tabName) {
  tabPills.forEach((btn) => {
    const isActive = btn.dataset.tab === tabName;
    btn.classList.toggle("tab-pill--active", isActive);
    btn.setAttribute("aria-selected", isActive ? "true" : "false");
  });
  workspaces.forEach((ws) => {
    const isActive = ws.id === `ws-${tabName}`;
    ws.classList.toggle("workspace--active", isActive);
    if (isActive) {
      ws.removeAttribute("hidden");
    } else {
      ws.setAttribute("hidden", "");
    }
  });
}

tabPills.forEach((btn) => {
  btn.addEventListener("click", () => activateTab(btn.dataset.tab));
});

// ===========================================================================
// AGENT OBSERVABILITY RAIL
// ===========================================================================

/**
 * Set the state badge for a named agent.
 * @param {string} id   - One of: kb | scam | budget | granite
 * @param {string} state - idle | active | done | error
 */
function setAgentState(id, state) {
  const badge = document.getElementById(`state-${id}`);
  if (!badge) return;
  // Remove all modifier classes then apply the new one
  badge.className = badge.className
    .replace(/\bstate-badge--\S+/g, "")
    .trim();
  badge.classList.add(`state-badge--${state}`);
  const labels = { idle: "IDLE", active: "ACTIVE", done: "DONE", error: "ERROR" };
  badge.textContent = labels[state] ?? state.toUpperCase();
}

function resetAgents(delay = 3000) {
  setTimeout(() => {
    ["kb", "scam", "budget", "granite"].forEach((id) => setAgentState(id, "idle"));
  }, delay);
}

// ===========================================================================
// ADVISORY CONSOLE — Tab 1
// ===========================================================================

const chatStream  = document.getElementById("chatMessages");
const chatInput   = document.getElementById("chatInput");
const chatForm    = document.getElementById("chatForm");
const sendBtn     = document.getElementById("sendBtn");
const statusLbl   = document.getElementById("agentStatusLabel");

function setStatusLabel(text, dotClass = "live-dot--green") {
  if (!statusLbl) return;
  const dot = statusLbl.querySelector(".live-dot");
  if (dot) {
    dot.className = dot.className.replace(/\blive-dot--\S+/g, "").trim();
    dot.classList.add(dotClass);
  }
  const span = statusLbl.querySelectorAll("span")[1];
  if (span) span.textContent = text;
}

/**
 * Append a message bubble to the chat stream.
 * @param {'user'|'bot'|'system'} role
 * @param {string} html   - Already-sanitised HTML
 * @param {object|null} scamData - Optional scam side-data to render inline
 */
function appendMsg(role, html, scamData = null) {
  if (!chatStream) return;
  const article = document.createElement("article");
  article.className = `msg msg--${role}`;

  let inner = `
    <header class="msg-meta">
      <span class="msg-role">${role === "user" ? "YOU" : role === "bot" ? "FINLIT AI" : "SYSTEM"}</span>
      <time class="msg-time">${nowStr()}</time>
    </header>
    <div class="msg-body">${html}</div>`;

  // Inline scam card when the chat response includes scam analysis
  if (scamData) {
    const lvl   = escapeHTML(scamData.risk_level  ?? "Unknown");
    const score = Number(scamData.risk_score ?? 0);
    const warns = Array.isArray(scamData.warnings) ? scamData.warnings : [];
    const modCls = lvl === "High" ? "scam-card--high"
                 : lvl === "Medium" ? "scam-card--medium"
                 : "scam-card--safe";
    const warnHtml = warns.length
      ? warns.map((w) => `<li>${escapeHTML(w)}</li>`).join("")
      : "<li>No specific red flags triggered.</li>";

    inner += `
    <div class="scam-card ${modCls}">
      <div class="scam-card__header">
        ${svg("i-shield")} Threat Score: <strong>${score}/100</strong>
        &nbsp;&mdash;&nbsp;<strong>${lvl}</strong>
      </div>
      <ul class="scam-card__list">${warnHtml}</ul>
    </div>`;
  }

  article.innerHTML = inner;
  chatStream.appendChild(article);
  chatStream.scrollTop = chatStream.scrollHeight;
  return article;
}

/** Append a pulsing "thinking" indicator. Returns the element so it can be removed. */
function appendTyping() {
  if (!chatStream) return null;
  const el = document.createElement("div");
  el.className = "typing-indicator";
  el.setAttribute("aria-label", "Agent processing");
  el.innerHTML = "<span></span><span></span><span></span>";
  chatStream.appendChild(el);
  chatStream.scrollTop = chatStream.scrollHeight;
  return el;
}

/** Disable / re-enable the send UI. */
function setSendLocked(locked) {
  if (sendBtn) sendBtn.disabled = locked;
  if (chatInput) chatInput.disabled = locked;
}

async function sendMessage(text) {
  text = (text || "").trim();
  if (!text) return;

  // Append user bubble
  appendMsg("user", mdToHtml(text));
  if (chatInput) chatInput.value = "";
  setSendLocked(true);
  setStatusLabel("Processing…", "live-dot--amber");

  // Activate agents visually
  setAgentState("kb", "active");
  setAgentState("granite", "active");

  const typing = appendTyping();

  try {
    const res  = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text }),
    });
    const data = await res.json();

    if (typing) typing.remove();

    if (!res.ok || data.error) {
      throw new Error(data.error ?? `HTTP ${res.status}`);
    }

    // Update agent states based on intent
    const intent = data.intent ?? "general";
    setAgentState("kb", data.context_used ? "done" : "idle");
    if (intent === "scam_check") setAgentState("scam", "done");
    if (intent === "budget")     setAgentState("budget", "done");
    setAgentState("granite", "done");

    const scamData = data.scam_analysis ?? null;
    appendMsg("bot", mdToHtml(data.response ?? "No response received."), scamData);
    setStatusLabel("All Agents Ready", "live-dot--green");

  } catch (err) {
    if (typing) typing.remove();
    setAgentState("kb", "error");
    setAgentState("granite", "error");
    showError("Chat error: " + err.message);
    appendMsg("bot", `<span class="err-inline">Request failed: ${escapeHTML(err.message)}</span>`);
    setStatusLabel("Error — retry", "live-dot--red");
  } finally {
    setSendLocked(false);
    resetAgents(3500);
  }
}

// Form submit
if (chatForm) {
  chatForm.addEventListener("submit", (e) => {
    e.preventDefault();
    sendMessage(chatInput?.value ?? "");
  });
}

// Shift+Enter = newline, Enter = send
if (chatInput) {
  chatInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage(chatInput.value);
    }
  });
  // Auto-grow textarea
  chatInput.addEventListener("input", () => {
    chatInput.style.height = "auto";
    chatInput.style.height = Math.min(chatInput.scrollHeight, 120) + "px";
  });
}

// Quick-prompt pills
document.querySelectorAll(".q-chip[data-prompt]").forEach((chip) => {
  chip.addEventListener("click", () => sendMessage(chip.dataset.prompt));
});

// ===========================================================================
// THREAT DETECTION ENGINE — Tab 2
// ===========================================================================

const scamInput       = document.getElementById("scamInput");
const analyzeScamBtn  = document.getElementById("analyzeScamBtn");
const clearScamBtn    = document.getElementById("clearScamBtn");
const threatResult    = document.getElementById("threatResult");
const threatPlaceholder = document.getElementById("threatPlaceholder");
const gaugeArc        = document.getElementById("gaugeArc");
const gaugeNeedle     = document.getElementById("gaugeNeedle");
const gaugeScoreTxt   = document.getElementById("gaugeScore");
const riskLevelBadge  = document.getElementById("riskLevelBadge");
const vectorsSection  = document.getElementById("vectorsSection");
const vectorList      = document.getElementById("vectorList");
const escalationBody  = document.getElementById("escalationBody");
const charCount       = document.getElementById("charCount");

const GAUGE_ARC_LEN = 283;   // stroke-dasharray on the SVG arc path

// Character counter
if (scamInput && charCount) {
  scamInput.addEventListener("input", () => {
    charCount.textContent = scamInput.value.length;
  });
}

/**
 * Animate the SVG risk gauge from 0 to the target score.
 * The arc goes from full-hidden (offset=283) to partially-revealed.
 * The needle rotates from -90deg (left end) to +90deg (right end).
 */
function animateGauge(score) {
  const clampedScore = Math.min(100, Math.max(0, score));
  const targetOffset = GAUGE_ARC_LEN - (GAUGE_ARC_LEN * clampedScore) / 100;
  // Needle: -90deg = 0, +90deg = 100
  const targetAngle  = -90 + (180 * clampedScore) / 100;

  let current = 0;
  const duration = 800;  // ms
  const startTime = performance.now();

  function step(ts) {
    const elapsed  = ts - startTime;
    const progress = Math.min(elapsed / duration, 1);
    // Ease-out cubic
    const eased    = 1 - Math.pow(1 - progress, 3);
    const now      = current + (clampedScore - current) * eased;

    const offset = GAUGE_ARC_LEN - (GAUGE_ARC_LEN * now) / 100;
    const angle  = -90 + (180 * now) / 100;

    if (gaugeArc)    gaugeArc.setAttribute("stroke-dashoffset", offset.toFixed(2));
    if (gaugeNeedle) gaugeNeedle.setAttribute("transform", `rotate(${angle.toFixed(2)}) translate(0, 0)`);
    if (gaugeScoreTxt) gaugeScoreTxt.textContent = Math.round(now);

    if (progress < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}

/** Apply colour modifier class to the risk level badge. */
function setRiskBadgeClass(level) {
  if (!riskLevelBadge) return;
  riskLevelBadge.className = riskLevelBadge.className
    .replace(/\brisk-level-badge--\S+/g, "")
    .trim();
  const modCls = level === "High"   ? "risk-level-badge--high"
               : level === "Medium" ? "risk-level-badge--medium"
               :                      "risk-level-badge--safe";
  riskLevelBadge.classList.add(modCls);
  riskLevelBadge.textContent = level;
}

function renderThreatResult(data) {
  const score  = Number(data.risk_score  ?? 0);
  const level  = data.risk_level  ?? "Unknown";
  const warns  = Array.isArray(data.warnings)    ? data.warnings    : [];
  const advice = data.action_advice ?? "";

  // Show diagnostic card, hide placeholder
  if (threatResult)     threatResult.removeAttribute("hidden");
  if (threatPlaceholder) threatPlaceholder.setAttribute("hidden", "");

  // Animate gauge
  animateGauge(score);

  // Risk level badge
  setRiskBadgeClass(level);

  // Threat vector list
  if (warns.length > 0) {
    if (vectorsSection) vectorsSection.removeAttribute("hidden");
    if (vectorList) {
      vectorList.innerHTML = warns.map((w, i) =>
        `<div class="vector-item">
          <span class="vector-num">${String(i + 1).padStart(2, "0")}</span>
          <span class="vector-text">${escapeHTML(w)}</span>
        </div>`
      ).join("");
    }
  } else {
    if (vectorsSection) vectorsSection.setAttribute("hidden", "");
    if (vectorList) vectorList.innerHTML = "";
  }

  // Escalation protocol / action advice
  if (escalationBody) {
    escalationBody.innerHTML = mdToHtml(advice);
  }
}

if (analyzeScamBtn) {
  analyzeScamBtn.addEventListener("click", async () => {
    const text = (scamInput?.value ?? "").trim();
    if (!text) {
      showError("Please paste a message to analyse.");
      return;
    }
    analyzeScamBtn.disabled = true;
    analyzeScamBtn.textContent = "Scanning…";

    try {
      const res  = await fetch("/api/analyze-scam", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      const data = await res.json();
      if (!res.ok || data.error) throw new Error(data.error ?? `HTTP ${res.status}`);
      renderThreatResult(data);
    } catch (err) {
      showError("Forensic scan failed: " + err.message);
    } finally {
      analyzeScamBtn.disabled = false;
      analyzeScamBtn.innerHTML = `${svg("i-shield")} Execute Forensic Scan`;
    }
  });
}

if (clearScamBtn) {
  clearScamBtn.addEventListener("click", () => {
    if (scamInput)        { scamInput.value = ""; }
    if (charCount)        charCount.textContent = "0";
    if (threatResult)     threatResult.setAttribute("hidden", "");
    if (threatPlaceholder) threatPlaceholder.removeAttribute("hidden");
    if (gaugeArc)         gaugeArc.setAttribute("stroke-dashoffset", String(GAUGE_ARC_LEN));
    if (gaugeNeedle)      gaugeNeedle.setAttribute("transform", "rotate(-90)");
    if (gaugeScoreTxt)    gaugeScoreTxt.textContent = "0";
    if (riskLevelBadge)   riskLevelBadge.textContent = "—";
    if (vectorList)       vectorList.innerHTML = "";
    if (vectorsSection)   vectorsSection.setAttribute("hidden", "");
    if (escalationBody)   escalationBody.innerHTML = "";
  });
}

// Sample scam message buttons
document.querySelectorAll(".sample-btn[data-scam]").forEach((btn) => {
  btn.addEventListener("click", () => {
    if (scamInput) {
      scamInput.value = btn.dataset.scam;
      if (charCount) charCount.textContent = scamInput.value.length;
    }
  });
});

// ===========================================================================
// CAPITAL ALLOCATOR — Tab 3
// ===========================================================================

const calcBudgetBtn   = document.getElementById("calcBudgetBtn");
const incomeInput     = document.getElementById("incomeInput");
const incomeDisplay   = document.getElementById("incomeDisplay");
const needsAmt        = document.getElementById("needsAmt");
const wantsAmt        = document.getElementById("wantsAmt");
const savingsAmt      = document.getElementById("savingsAmt");
const runwayBody      = document.getElementById("runwayBody");
const emiBody         = document.getElementById("emiBody");

/** Singleton Chart.js instance — destroyed before re-creating. */
window._budgetChart = null;

function renderDonutChart(labels, values, colors) {
  const canvas = document.getElementById("budgetChart");
  if (!canvas) return;

  if (window._budgetChart) {
    window._budgetChart.destroy();
    window._budgetChart = null;
  }

  window._budgetChart = new Chart(canvas, {
    type: "doughnut",
    data: {
      labels,
      datasets: [{
        data: values,
        backgroundColor: colors,
        borderColor: "#0F172A",
        borderWidth: 3,
        hoverBorderColor: "#F8FAFC",
        hoverBorderWidth: 2,
      }],
    },
    options: {
      cutout: "68%",
      responsive: true,
      maintainAspectRatio: true,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (ctx) => ` ${fmtRs(ctx.parsed)}`,
          },
          backgroundColor: "#162032",
          titleColor: "#94A3B8",
          bodyColor: "#F8FAFC",
          borderColor: "#1E293B",
          borderWidth: 1,
        },
      },
      animation: { duration: 900, easing: "easeOutQuart" },
    },
  });
}

/** Build a single .data-row element string. */
function dataRow(key, value, highlight = false) {
  const valClass = highlight ? "data-val data-val--highlight" : "data-val";
  return `<div class="data-row">
    <span class="data-key">${escapeHTML(key)}</span>
    <span class="${valClass}">${escapeHTML(String(value))}</span>
  </div>`;
}

/** Build the health score bar HTML. */
function healthBar(score) {
  const pct  = Math.min(100, Math.max(0, score));
  const cls  = pct >= 70 ? "health-fill--good"
             : pct >= 40 ? "health-fill--fair"
             :              "health-fill--poor";
  return `<div class="health-row">
    <span class="data-key">Financial Health Score</span>
    <div class="health-track">
      <div class="health-fill ${cls}" style="width:${pct}%"></div>
    </div>
    <span class="data-val">${pct}/100</span>
  </div>`;
}

function renderBudget(data) {
  const income     = Number(data.income ?? 0);
  const alloc      = data.allocations ?? {};
  const ef         = data.emergency_fund ?? {};
  const inv        = data.investment_split ?? {};
  const emi        = data.emi_examples ?? {};
  const surplus    = Number(data.surplus ?? 0);
  const healthScore = Number(data.financial_health_score ?? 0);
  const chartData  = data.chart_data ?? {};

  // Income badge
  if (incomeDisplay) incomeDisplay.textContent = fmtRs(income) + " / mo";

  // Allocation amounts
  if (needsAmt)   needsAmt.textContent   = fmtRs(alloc.needs?.target   ?? 0);
  if (wantsAmt)   wantsAmt.textContent   = fmtRs(alloc.wants?.target   ?? 0);
  if (savingsAmt) savingsAmt.textContent = fmtRs(alloc.savings?.target ?? 0);

  // Donut chart
  renderDonutChart(
    chartData.labels ?? ["Needs", "Wants", "Savings"],
    chartData.values ?? [alloc.needs?.target, alloc.wants?.target, alloc.savings?.target],
    chartData.colors ?? ["#3b82f6", "#f59e0b", "#10b981"]
  );

  // Emergency runway + investment split card
  if (runwayBody) {
    runwayBody.innerHTML = [
      dataRow("3-Month Emergency Fund",  fmtRs(ef.target_3_months  ?? 0), true),
      dataRow("6-Month Emergency Fund",  fmtRs(ef.target_6_months  ?? 0)),
      dataRow("Recommended Instrument",  ef.recommended_instrument ?? "—"),
      dataRow("SIP Mutual Fund",         fmtRs(inv.sip_mutual_fund ?? 0), true),
      dataRow("PPF Contribution",        fmtRs(inv.ppf ?? 0)),
      dataRow("Monthly Surplus",         fmtRs(surplus)),
      healthBar(healthScore),
    ].join("");
  }

  // EMI amortisation matrix
  if (emiBody) {
    emiBody.innerHTML = [
      dataRow("Sample Loan Amount",      fmtRs(emi.loan_amount  ?? 0), true),
      dataRow("Interest Rate (p.a.)",    (emi.rate_pa ?? 14.0) + "%"),
      dataRow("EMI — 12-Month Tenure",   fmtRs(emi.emi_12_months ?? 0), true),
      dataRow("EMI — 24-Month Tenure",   fmtRs(emi.emi_24_months ?? 0)),
      dataRow("Total Interest (12M)",    fmtRs(((emi.emi_12_months ?? 0) * 12) - (emi.loan_amount ?? 0))),
      dataRow("Total Interest (24M)",    fmtRs(((emi.emi_24_months ?? 0) * 24) - (emi.loan_amount ?? 0))),
      dataRow("Safe EMI / Income Ratio", "&lt;= 40% of take-home"),
    ].join("");
  }
}

if (calcBudgetBtn) {
  calcBudgetBtn.addEventListener("click", async () => {
    const income = parseFloat(incomeInput?.value ?? 0);
    if (!income || income <= 0) {
      showError("Please enter a valid monthly income.");
      return;
    }

    calcBudgetBtn.disabled = true;
    calcBudgetBtn.innerHTML = `${svg("i-pie")} Computing…`;

    const payload = {
      income,
      housing:       parseFloat(document.getElementById("expHousing")?.value      ?? 0) || 0,
      utilities:     parseFloat(document.getElementById("expUtilities")?.value    ?? 0) || 0,
      groceries:     parseFloat(document.getElementById("expGroceries")?.value    ?? 0) || 0,
      emis:          parseFloat(document.getElementById("expEmis")?.value          ?? 0) || 0,
      dining:        parseFloat(document.getElementById("expDining")?.value        ?? 0) || 0,
      entertainment: parseFloat(document.getElementById("expEntertainment")?.value ?? 0) || 0,
    };

    try {
      const res  = await fetch("/api/calculate-budget", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (!res.ok || data.error) throw new Error(data.error ?? `HTTP ${res.status}`);
      renderBudget(data);
    } catch (err) {
      showError("Budget calculation failed: " + err.message);
    } finally {
      calcBudgetBtn.disabled = false;
      calcBudgetBtn.innerHTML = `${svg("i-pie")} Compute Allocation`;
    }
  });
}

// Income preset buttons
document.querySelectorAll(".preset-btn[data-income]").forEach((btn) => {
  btn.addEventListener("click", () => {
    if (incomeInput) {
      incomeInput.value = btn.dataset.income;
      // Auto-trigger calculation
      calcBudgetBtn?.click();
    }
  });
});
