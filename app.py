"""Fake News Debater frontend."""
from __future__ import annotations

import hashlib
import html
import json as json_lib
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

import streamlit as st

from agents.claim_extractor import extract_claims
from agents.falsifier_agent import falsify_claims
from agents.judge_agent import judge_debate
from agents.verifier_agent import verify_claims
from config import validate_config, MIN_ARTICLE_LENGTH, MAX_ARTICLE_LENGTH
from tools.article_scraper import scrape_article

SAMPLE_ARTICLES = {
    "Test: Obvious Fake News": (
        "BREAKING: Scientists at NASA have confirmed that the moon is actually made of cheese. "
        "According to Dr. James Robertson, head of NASA's Lunar Research Division, samples brought "
        "back by the Apollo missions were reanalyzed using advanced spectroscopy and found to contain "
        "high levels of casein and whey proteins. The discovery was published in the journal Nature "
        "on March 15, 2025. NASA Administrator Bill Nelson held a press conference at the Kennedy "
        "Space Center confirming the findings and announcing a $50 billion mission to harvest lunar "
        "cheese by 2030. The European Space Agency has independently verified these results."
    ),
    "Test: Real-ish News": (
        "The World Health Organization reported on January 12, 2025, that global life expectancy "
        "has increased by 6 years since 2000, reaching an average of 73.4 years. The report, "
        "published in The Lancet, attributes the improvement to better access to healthcare in "
        "developing nations, reduced child mortality rates, and advances in treating infectious diseases."
    ),
    "Test: Misleading Mix": (
        "A new study from Stanford University shows that drinking 8 glasses of water per day can "
        "reduce cancer risk by 45%. The research, led by Dr. Sarah Chen of the Stanford School of "
        "Medicine, followed 50,000 participants over 10 years. Critics point out that the study was "
        "funded by Evian and relied heavily on self-reported water intake."
    ),
}

st.set_page_config(page_title="Fake News Debater", page_icon="FND", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=IBM+Plex+Mono:wght@400;500;600&display=swap');
:root{
--bg:#071019;--bg2:#0d1824;--panel:rgba(255,255,255,.045);--line:rgba(255,255,255,.10);
--text:#f3f6f8;--muted:#95a4b5;--dim:#647387;--accent:#ff6b3d;--good:#35c58e;--bad:#ff7373;--warn:#f3ba4c;
}
html,body,[class*="css"]{font-family:'Manrope',sans-serif}
.stApp{background:
radial-gradient(circle at 18% 0%, rgba(255,107,61,.16), transparent 24%),
radial-gradient(circle at 82% 12%, rgba(94,204,255,.12), transparent 20%),
linear-gradient(180deg,#071019 0%,#09131c 100%);color:var(--text)}
.block-container{max-width:1160px;padding-top:0!important;padding-bottom:3.5rem!important}
header[data-testid="stHeader"]{background:transparent!important} #MainMenu,footer,.stDeployButton{visibility:hidden}
section[data-testid="stSidebar"]{background:linear-gradient(180deg,rgba(7,16,25,.96),rgba(12,23,34,.96))!important;border-right:1px solid var(--line)!important}
section[data-testid="stSidebar"] .stMarkdown p,section[data-testid="stSidebar"] .stCaption,section[data-testid="stSidebar"] li{color:var(--muted)!important}
.hero{width:100vw;margin-left:calc(50% - 50vw);min-height:78svh;padding:6.5rem 0 3.2rem;position:relative;overflow:hidden;border-bottom:1px solid rgba(255,255,255,.06);
background:linear-gradient(120deg,rgba(6,13,20,.95) 0%,rgba(6,13,20,.45) 46%,rgba(6,13,20,.88) 100%),linear-gradient(135deg,#071019 0%,#142537 48%,#09131c 100%)}
.hero::before{content:"";position:absolute;inset:0;background:repeating-linear-gradient(90deg,transparent 0,transparent 56px,rgba(255,255,255,.02) 56px,rgba(255,255,255,.02) 57px)}
.hero-grid{position:relative;z-index:1;max-width:1160px;margin:0 auto;padding:0 1.2rem;display:grid;grid-template-columns:minmax(0,1.12fr) minmax(280px,.7fr);gap:2rem;align-items:end}
.kicker,.eyebrow,.mini{letter-spacing:.16em;text-transform:uppercase}.kicker{color:#ffd1c3;font-size:.76rem;margin-bottom:.9rem}.eyebrow{color:rgba(255,255,255,.66);font-size:.9rem;margin-bottom:.8rem}
.hero h1{margin:0;max-width:7ch;font-size:clamp(3.4rem,8vw,7rem);line-height:.92;letter-spacing:-.055em}.hero p{max-width:34rem;color:var(--muted);font-size:1rem;line-height:1.7;margin:1.15rem 0 1.6rem}
.hero-rail{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:.8rem;max-width:31rem}.rail{padding-top:.85rem;border-top:1px solid rgba(255,255,255,.15)}.rail strong{display:block;font-size:1.45rem}.rail span{font-size:.76rem;color:var(--muted);letter-spacing:.12em;text-transform:uppercase}
.hero-side{min-height:380px;border-left:1px solid rgba(255,255,255,.08);padding-left:1.2rem}.signal{display:grid;grid-template-columns:72px 1fr;gap:.7rem;align-items:center;margin-bottom:.8rem}.mini{font-size:.68rem;color:rgba(255,255,255,.54)}
.bar{height:8px;border-radius:999px;background:rgba(255,255,255,.08);overflow:hidden}.bar span{display:block;height:100%;border-radius:999px;background:linear-gradient(90deg,rgba(255,255,255,.25),var(--accent))}
.quote{margin-top:2rem;max-width:18rem;color:#f8d8cd;font-size:1.15rem;line-height:1.35}.quote-sub{margin-top:.5rem;color:rgba(255,255,255,.54);font-size:.82rem;line-height:1.55}
.section{padding-top:1.6rem}.head{display:flex;justify-content:space-between;gap:1rem;align-items:end;padding-bottom:.8rem;margin-bottom:1rem;border-bottom:1px solid var(--line)}
.label{font-size:.72rem;letter-spacing:.15em;text-transform:uppercase;color:rgba(255,255,255,.56)}.title{font-size:1.42rem;font-weight:700;letter-spacing:-.03em}.copy{max-width:34rem;color:var(--muted);font-size:.92rem;line-height:1.6}
.workspace{display:grid;grid-template-columns:minmax(0,1.1fr) minmax(290px,.74fr);gap:1.35rem}.panel{padding:1.2rem 1.25rem;border:1px solid var(--line);border-radius:24px;background:linear-gradient(180deg,rgba(255,255,255,.045),rgba(255,255,255,.025));backdrop-filter:blur(10px);transition:transform .22s ease,border-color .22s ease,background .22s ease}
.panel.side{background:linear-gradient(180deg,rgba(255,255,255,.05),rgba(255,255,255,.025)),linear-gradient(135deg,rgba(255,107,61,.08),transparent 60%)}
.flow{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:.8rem;margin-top:1rem}.flow-item{padding-top:.85rem;border-top:1px solid rgba(255,255,255,.15)}.flow-item strong{display:block;font-size:.95rem;margin-bottom:.22rem}.flow-item span{color:var(--muted);font-size:.84rem;line-height:1.55}
.stTabs [data-baseweb="tab-list"]{gap:.45rem;margin-bottom:.8rem}.stTabs button[data-baseweb="tab"]{border-radius:999px!important;border:1px solid var(--line)!important;background:rgba(255,255,255,.03)!important;color:var(--muted)!important;padding:.45rem 1rem!important;font-weight:600!important}
.stTabs button[data-baseweb="tab"][aria-selected="true"]{background:rgba(255,107,61,.12)!important;border-color:rgba(255,107,61,.38)!important;color:#ffd6ca!important}
.stTextArea textarea,.stTextInput input{border-radius:18px!important;border:1px solid var(--line)!important;background:rgba(255,255,255,.03)!important;color:var(--text)!important}
.stTextArea textarea:focus,.stTextInput input:focus{border-color:rgba(255,107,61,.55)!important;box-shadow:0 0 0 4px rgba(255,107,61,.14)!important}
.stButton>button{border-radius:999px!important;min-height:2.8rem!important;font-weight:700!important;transition:all .22s ease!important}
.stButton>button[kind="primary"]{background:linear-gradient(135deg,#ff6b3d,#ff8c67)!important;color:#21130d!important;border:none!important;box-shadow:0 14px 34px rgba(255,107,61,.22)!important}
.info{padding-bottom:.95rem;margin-bottom:.95rem;border-bottom:1px solid rgba(255,255,255,.08)} .info:last-child{border-bottom:none;margin-bottom:0;padding-bottom:0}
.info strong{display:block;font-size:.74rem;letter-spacing:.14em;text-transform:uppercase;color:rgba(255,255,255,.56);margin-bottom:.35rem}.info span{color:var(--muted);font-size:.9rem;line-height:1.62}
.strip{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:1rem;margin-top:.35rem}.strip-item{padding-top:.9rem;border-top:1px solid var(--line)}.strip-item strong{display:block;font-size:1rem;margin-bottom:.25rem}.strip-item span{color:var(--muted);font-size:.9rem;line-height:1.58}
.divider{height:1px;margin:2rem 0 1.5rem;background:linear-gradient(90deg,transparent 0%, rgba(255,255,255,.16) 50%, transparent 100%)}
.claim,.break{display:grid;gap:.75rem}.row{display:grid;grid-template-columns:42px 1fr auto auto;gap:.75rem;align-items:start;padding:.92rem 0;border-bottom:1px solid rgba(255,255,255,.07)}.row:last-child{border-bottom:none}
.idx{width:42px;height:42px;border-radius:50%;display:inline-flex;align-items:center;justify-content:center;background:rgba(255,255,255,.06);font-weight:800;color:#ffd6ca}
.text{color:var(--text);font-size:.95rem;line-height:1.58}.reason{margin-top:.28rem;color:var(--muted);font-size:.84rem;line-height:1.6}.tag,.pill,.conf{font-size:.7rem;letter-spacing:.12em;text-transform:uppercase}
.tag,.pill{padding:.38rem .65rem;border-radius:999px;border:1px solid var(--line)}.tag.high{color:#ffd2c3;border-color:rgba(255,107,61,.34);background:rgba(255,107,61,.11)}.tag.medium,.pill.unverifiable{color:#ffd98c;border-color:rgba(243,186,76,.28);background:rgba(243,186,76,.10)}
.tag.low,.pill.supported{color:#9ee5c8;border-color:rgba(53,197,142,.28);background:rgba(53,197,142,.10)}.pill.refuted{color:#ffb7b7;border-color:rgba(255,115,115,.28);background:rgba(255,115,115,.10)}
.conf{color:var(--dim);font-family:'IBM Plex Mono',monospace;padding-top:.38rem}
.verdict{padding:1.8rem;border:1px solid var(--line);border-radius:28px;background:linear-gradient(180deg,rgba(255,255,255,.05),rgba(255,255,255,.025)),linear-gradient(135deg,rgba(255,107,61,.12),transparent 58%);position:relative;overflow:hidden;transition:transform .22s ease,border-color .22s ease}
.verdict::after{content:"";position:absolute;right:-70px;top:-70px;width:220px;height:220px;border-radius:50%;background:radial-gradient(circle,rgba(255,255,255,.12),transparent 68%)}
.v-word{font-size:clamp(2rem,5vw,3.3rem);line-height:.95;font-weight:800;letter-spacing:-.05em}.v-word.real{color:var(--good)}.v-word.fake{color:var(--bad)}.v-word.misleading{color:var(--warn)}
.v-copy{max-width:42rem;color:var(--muted);font-size:.96rem;line-height:1.68;margin-top:1rem}.track{width:min(380px,100%);margin:.95rem 0;height:9px;border-radius:999px;background:rgba(255,255,255,.08);overflow:hidden}.track span{display:block;height:100%;border-radius:999px}
.track .real{background:linear-gradient(90deg,#28a574,#35c58e)}.track .fake{background:linear-gradient(90deg,#f25656,#ff7373)}.track .misleading{background:linear-gradient(90deg,#e7a22d,#f3ba4c)}
.metric-stack{margin-top:1rem;display:grid;gap:.75rem}.metric-bar{height:12px;border-radius:999px;background:rgba(255,255,255,.08);overflow:hidden;display:flex}
.metric-bar span{display:block;height:100%}.metric-bar .real{background:linear-gradient(90deg,#28a574,#35c58e)}.metric-bar .fake{background:linear-gradient(90deg,#f25656,#ff7373)}.metric-bar .misleading{background:linear-gradient(90deg,#e7a22d,#f3ba4c)}
.metric-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:.8rem}.metric-item{padding-top:.7rem;border-top:1px solid rgba(255,255,255,.09)}
.metric-item strong{display:block;font-size:.72rem;letter-spacing:.12em;text-transform:uppercase;color:rgba(255,255,255,.58);margin-bottom:.2rem}.metric-item span{display:block;font-family:'IBM Plex Mono',monospace;font-size:1.05rem}
.metric-item.real span{color:var(--good)}.metric-item.fake span{color:var(--bad)}.metric-item.misleading span{color:var(--warn)}
.callout{margin-top:1rem;padding:1rem 1.05rem;border:1px solid var(--line);border-radius:20px;background:rgba(255,255,255,.035)}.callout.good{border-color:rgba(53,197,142,.22);background:linear-gradient(180deg,rgba(53,197,142,.09),rgba(255,255,255,.025))}.callout.warn{border-color:rgba(243,186,76,.24);background:linear-gradient(180deg,rgba(243,186,76,.10),rgba(255,255,255,.025))}.callout.bad{border-color:rgba(255,115,115,.24);background:linear-gradient(180deg,rgba(255,115,115,.10),rgba(255,255,255,.025))}
.health-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:1rem}.health-card{padding-top:.85rem;border-top:1px solid var(--line)}.health-card strong{display:block;font-size:.74rem;letter-spacing:.14em;text-transform:uppercase;color:rgba(255,255,255,.56);margin-bottom:.35rem}.health-card span{display:block;color:var(--text);font-family:'IBM Plex Mono',monospace;font-size:1.45rem;line-height:1.05}.health-card small{display:block;margin-top:.34rem;color:rgba(255,255,255,.56);font-size:.72rem;letter-spacing:.14em;text-transform:uppercase}
.health-pill{font-size:.7rem;letter-spacing:.12em;text-transform:uppercase;padding:.38rem .65rem;border-radius:999px;border:1px solid var(--line)}.health-pill.good{color:#9ee5c8;border-color:rgba(53,197,142,.28);background:rgba(53,197,142,.10)}.health-pill.warn{color:#ffd98c;border-color:rgba(243,186,76,.28);background:rgba(243,186,76,.10)}.health-pill.bad{color:#ffb7b7;border-color:rgba(255,115,115,.28);background:rgba(255,115,115,.10)}
[data-testid="stExpander"]{border:1px solid var(--line)!important;border-radius:20px!important;background:rgba(255,255,255,.03)!important;transition:transform .22s ease,border-color .22s ease!important}
.debate{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:1rem}.face{padding:1rem;border:1px solid var(--line);border-radius:20px;background:rgba(255,255,255,.03)}.face.ver{background:linear-gradient(180deg,rgba(53,197,142,.10),rgba(255,255,255,.025));border-color:rgba(53,197,142,.22)}.face.fal{background:linear-gradient(180deg,rgba(255,115,115,.10),rgba(255,255,255,.025));border-color:rgba(255,115,115,.22)}
.face-head{display:flex;justify-content:space-between;gap:.75rem;align-items:center;padding-bottom:.65rem;margin-bottom:.75rem;border-bottom:1px solid rgba(255,255,255,.08)}.face-title{font-size:.9rem;font-weight:700;letter-spacing:.08em;text-transform:uppercase}.face-tone{font-size:.75rem;color:var(--muted);letter-spacing:.12em;text-transform:uppercase}
.debate-body{color:var(--muted);font-size:.92rem;line-height:1.7}.evidence{display:grid;gap:.7rem;margin-top:.8rem}.e-item{padding-top:.75rem;border-top:1px solid rgba(255,255,255,.07)}.e-head{display:flex;justify-content:space-between;gap:.7rem;align-items:baseline}.e-title{font-size:.85rem;font-weight:700}.e-meta{font-size:.72rem;color:var(--dim);font-family:'IBM Plex Mono',monospace}.e-copy{margin-top:.25rem;color:var(--muted);font-size:.84rem;line-height:1.58}.e-link{display:inline-block;margin-top:.35rem;font-size:.76rem;color:#ffb39a;text-decoration:none}.e-link:hover{text-decoration:underline}
.stats{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:1rem}.slot{padding-top:.9rem;border-top:1px solid var(--line)}.slot strong{display:block;font-family:'IBM Plex Mono',monospace;font-size:2rem;line-height:1}.slot span{display:block;margin-top:.34rem;color:rgba(255,255,255,.56);font-size:.72rem;letter-spacing:.14em;text-transform:uppercase}
.footer-wrap{margin-top:2.3rem;padding-top:1.2rem;border-top:1px solid rgba(255,255,255,.08);display:flex;justify-content:space-between;gap:1rem;flex-wrap:wrap;color:var(--dim);font-size:.78rem}.footer-wrap a{color:#ffb39a;text-decoration:none}.footer-wrap a:hover{text-decoration:underline}
div[data-testid="stStatusWidget"]{border-radius:20px!important;border:1px solid var(--line)!important;background:rgba(255,255,255,.04)!important}
.panel:hover,.verdict:hover,[data-testid="stExpander"]:hover{transform:translateY(-2px);border-color:rgba(255,255,255,.16)!important}
@keyframes fade-up{from{opacity:0;transform:translateY(18px)}to{opacity:1;transform:translateY(0)}}
.hero-grid>div:first-child{animation:fade-up .55s ease both}.hero-side{animation:fade-up .75s ease both}.section{animation:fade-up .35s ease both}
@media (max-width:980px){.hero-grid,.workspace{grid-template-columns:1fr}.hero{min-height:auto}.hero-side{border-left:none;border-top:1px solid rgba(255,255,255,.08);padding-left:0;padding-top:1.2rem;min-height:auto}.strip,.stats,.hero-rail,.debate,.metric-grid,.health-grid,.flow{grid-template-columns:1fr 1fr}}
@media (max-width:640px){.block-container{padding-left:.9rem!important;padding-right:.9rem!important}.hero-grid{padding:0 1rem}.strip,.stats,.hero-rail,.debate,.metric-grid,.health-grid,.flow{grid-template-columns:1fr}.row{grid-template-columns:36px 1fr}.tag,.pill,.conf{grid-column:2;justify-self:start}}
</style>
""",
    unsafe_allow_html=True,
)


def _escape(value: object) -> str:
    return html.escape(str(value or ""), quote=True)


def _safe_url(url: str) -> str | None:
    raw = str(url or "").strip()
    parsed = urlparse(raw)
    if parsed.scheme in {"http", "https"} and parsed.netloc:
        return html.escape(raw, quote=True)
    return None


def _confidence_metrics(verdict: dict) -> dict[str, float]:
    metrics = verdict.get("confidence_metrics") or {}
    normalized: dict[str, float] = {}
    for label in ("REAL", "FAKE", "MISLEADING"):
        try:
            normalized[label] = max(0.0, min(float(metrics.get(label, 0.0)), 1.0))
        except (TypeError, ValueError):
            normalized[label] = 0.0

    total = sum(normalized.values())
    if total <= 0:
        fallback = max(0.0, min(float(verdict.get("overall_confidence", 0.5)), 1.0))
        overall = verdict.get("overall_verdict", "MISLEADING")
        normalized = {"REAL": 0.0, "FAKE": 0.0, "MISLEADING": 0.0}
        normalized[overall if overall in normalized else "MISLEADING"] = fallback or 1.0
        total = sum(normalized.values())

    return {label: value / total for label, value in normalized.items()}


def _as_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clear_loaded_url() -> None:
    st.session_state.pop("url_article_text", None)
    st.session_state.pop("url_article_title", None)
    st.session_state.pop("url_article_error", None)
    st.session_state["article_url_input_v2"] = ""


def _load_sample(label: str, text: str) -> None:
    st.session_state["sample_text"] = text
    st.session_state["sample_loaded"] = label
    st.session_state["article_input_v2"] = text
    _clear_loaded_url()


def _analysis_health(verifier_report: dict, falsifier_report: dict) -> dict:
    reports = verifier_report.get("claim_reports", []) + falsifier_report.get("claim_reports", [])
    evidence = [item for report in reports for item in report.get("evidence", [])]
    total_evidence = len(evidence)
    scraped_sources = sum(1 for item in evidence if item.get("scraped"))
    fallback_classifications = sum(1 for item in evidence if item.get("used_fallback"))
    usable_signals = sum(
        1
        for item in evidence
        if item.get("stance") in {"SUPPORT", "CONTRADICT"}
        and _as_float(item.get("confidence", 0.0)) >= 0.55
        and not item.get("used_fallback")
    )

    if total_evidence == 0:
        tone = "bad"
        label = "Limited signal"
        summary = "No evidence items survived ranking, so the final call should be treated cautiously."
    elif fallback_classifications >= max(3, int(total_evidence * 0.35)):
        tone = "bad"
        label = "Degraded run"
        summary = (
            f"{fallback_classifications} of {total_evidence} evidence items fell back to neutral after "
            "stance-classification failures."
        )
    elif usable_signals < max(2, len(reports) // 2):
        tone = "warn"
        label = "Thin signal"
        summary = (
            "The run completed, but only a small share of the evidence produced clear support "
            "or contradiction."
        )
    else:
        tone = "good"
        label = "Healthy run"
        summary = (
            f"The pipeline surfaced {usable_signals} usable evidence signals across "
            f"{scraped_sources} scraped sources."
        )

    return {
        "engine": "Groq stance",
        "label": label,
        "tone": tone,
        "summary": summary,
        "total_evidence": total_evidence,
        "scraped_sources": scraped_sources,
        "fallback_classifications": fallback_classifications,
        "usable_signals": usable_signals,
    }


def _ensure_analysis_health(results: dict) -> dict:
    health = results.get("analysis_health")
    if health:
        return health
    health = _analysis_health(results["verifier_report"], results["falsifier_report"])
    results["analysis_health"] = health
    return health


def _render_sidebar() -> None:
    with st.sidebar:
        st.markdown("## Fake News Debater")
        st.caption("Adversarial misinformation review")
        st.markdown("---")
        st.markdown("### System")
        st.markdown("1. Claim extraction\n2. Support retrieval\n3. Rebuttal retrieval\n4. Final judgment")
        st.markdown("---")
        st.markdown("### Samples")
        for label, text in SAMPLE_ARTICLES.items():
            if st.button(label, key=f"sample_{label}", use_container_width=True):
                _load_sample(label, text)
                st.rerun()
        st.markdown("---")
        st.markdown("### Recent")
        for item in reversed(st.session_state.get("history", [])[-5:]):
            confidence = item.get("confidence")
            suffix = f" | {confidence:.0%}" if isinstance(confidence, float) else ""
            st.caption(f"{item['verdict']}{suffix} | {item['preview']}")
        if not st.session_state.get("history"):
            st.caption("No analyses yet.")


def _render_hero() -> None:
    st.markdown(
        """
        <section class="hero">
            <div class="hero-grid">
                <div>
                    <div class="kicker">Editorial intelligence for noisy articles</div>
                    <div class="eyebrow">Fake News Debater</div>
                    <h1>Audit the story, not just the headline.</h1>
                    <p>
                        Feed the system an article or URL and it extracts factual claims, sends two agents
                        to argue opposite sides, and then hands the evidence to a final judge.
                    </p>
                    <div class="hero-rail">
                        <div class="rail"><strong>3-5</strong><span>claims extracted</span></div>
                        <div class="rail"><strong>2</strong><span>debating agents</span></div>
                        <div class="rail"><strong>1</strong><span>final verdict</span></div>
                    </div>
                </div>
                <div class="hero-side">
                    <div class="signal"><div class="mini">claims</div><div class="bar"><span style="width:78%"></span></div></div>
                    <div class="signal"><div class="mini">support</div><div class="bar"><span style="width:62%"></span></div></div>
                    <div class="signal"><div class="mini">rebuttal</div><div class="bar"><span style="width:85%"></span></div></div>
                    <div class="signal"><div class="mini">judge</div><div class="bar"><span style="width:56%"></span></div></div>
                    <div class="quote">"Slow the reader down and make the argument visible before the trust settles in."</div>
                    <div class="quote-sub">A sharper first screen, restrained surfaces, and evidence-first pacing.</div>
                </div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def _render_workspace() -> None:
    st.markdown(
        """
        <section class="section">
            <div class="head">
                <div><div class="label">Primary workspace</div><div class="title">Load an article and run the debate.</div></div>
                <div class="copy">Paste full text for speed or scrape a live URL when you want the article body extracted automatically.</div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    prefill = st.session_state.get("sample_text", "")
    if prefill:
        st.info(f"Loaded sample: {st.session_state.get('sample_loaded', 'Sample')}")

    left, right = st.columns([1.12, 0.78], gap="large")
    article_text = ""

    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        tab_text, tab_url = st.tabs(["Paste text", "From URL"])
        with tab_text:
            article_text = st.text_area(
                "Article text",
                value=prefill,
                height=280,
                placeholder="Paste the article body here.",
                label_visibility="collapsed",
                key="article_input",
            )
            # Fix #15: Article length indicator
            if article_text:
                char_count = len(article_text)
                if char_count < MIN_ARTICLE_LENGTH:
                    st.caption(f"⚠️ {char_count:,} chars — minimum {MIN_ARTICLE_LENGTH} required")
                elif char_count > MAX_ARTICLE_LENGTH:
                    st.caption(f"⚠️ {char_count:,} chars — maximum {MAX_ARTICLE_LENGTH:,} exceeded")
                else:
                    st.caption(f"✓ {char_count:,} chars")
            if prefill:
                st.session_state.pop("sample_text", None)
        with tab_url:
            url = st.text_input("URL", placeholder="https://example.com/news-article", label_visibility="collapsed")
            if url:
                with st.spinner("Scraping article..."):
                    result = scrape_article(url)
                if result["success"]:
                    article_text = result["text"]
                    st.success(f"Loaded: {result['title']} - {len(article_text):,} chars")
                    with st.expander("Preview extracted text", expanded=False):
                        st.text(article_text[:1000] + ("..." if len(article_text) > 1000 else ""))
                else:
                    st.error(result["error"])
        c1, c2, c3 = st.columns([1, 1.2, 1])
        with c2:
            analyze = st.button("Run analysis", use_container_width=True, type="primary")
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown(
            """
            <div class="panel side">
                <div class="info"><strong>What happens</strong><span>The system extracts claims, retrieves supporting and opposing evidence, then asks a judge to weigh both cases.</span></div>
                <div class="info"><strong>Best input</strong><span>Articles with named people, organizations, dates, figures, and checkable events work best.</span></div>
                <div class="info"><strong>Caution</strong><span>This is a research-style review surface, not a final authority. Weak evidence and unverifiable claims should trigger deeper checking.</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        for label, text in SAMPLE_ARTICLES.items():
            if st.button(label, key=f"work_{label}", use_container_width=True):
                st.session_state["sample_text"] = text
                st.session_state["sample_loaded"] = label
                st.rerun()

    st.markdown(
        """
        <section class="section">
            <div class="head">
                <div><div class="label">Reading the output</div><div class="title">One interface, four jobs.</div></div>
                <div class="copy">Claims isolate the article, the verdict compresses it, the debate slows you down, and the breakdown shows exactly where things hold or fail.</div>
            </div>
            <div class="strip">
                <div class="strip-item"><strong>Claims</strong><span>See what the model believes is factual and checkable.</span></div>
                <div class="strip-item"><strong>Verdict</strong><span>Get a fast article-level read on whether the story looks real, fake, or mixed.</span></div>
                <div class="strip-item"><strong>Debate</strong><span>Inspect support and rebuttal before trusting the final call.</span></div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    if analyze and article_text:
        # Fix #10: Input length validation
        if len(article_text) < MIN_ARTICLE_LENGTH:
            st.warning(f"Article is too short ({len(article_text)} chars). Please provide at least {MIN_ARTICLE_LENGTH} characters.")
        elif len(article_text) > MAX_ARTICLE_LENGTH:
            st.warning(f"Article is too long ({len(article_text):,} chars). Please shorten to {MAX_ARTICLE_LENGTH:,} characters.")
        else:
            _run_analysis(article_text)
    elif analyze:
        st.warning("Paste article text or provide a URL first.")


def _run_analysis(article_text: str) -> None:
    # Fix #5: Cache check — skip re-analysis for same article
    article_hash = hashlib.sha256(article_text.encode()).hexdigest()[:16]
    if st.session_state.get("last_hash") == article_hash and st.session_state.get("last_results"):
        st.info("Showing cached results for this article. Change the text to re-analyze.")
        cached = st.session_state["last_results"]
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        _render_claims(cached["claims"])
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        _render_verdict(cached["verdict"])
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        _render_debate(cached["verifier_report"], cached["falsifier_report"], cached["claims"])
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        _render_breakdown(cached["verdict"])
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        _render_stats(cached["verifier_report"], cached["falsifier_report"], cached["verdict"], cached["claims"])
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        _render_export(cached)
        return

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    with st.status("Running multi-agent analysis...", expanded=True) as status:
        # Fix #14: More granular progress
        st.write("⏳ Step 1/4 — Extracting factual claims via spaCy NER + LLM...")
        claims = extract_claims(article_text)
        if not claims:
            st.error("Could not extract any claims from this article.")
            status.update(label="Analysis failed", state="error")
            return
        st.write(f"✅ Extracted **{len(claims)}** verifiable claims.")

        st.write("⏳ Step 2/4 — Verifier & Falsifier searching the web for evidence...")
        with ThreadPoolExecutor(max_workers=2) as executor:
            verifier_future = executor.submit(verify_claims, claims)
            falsifier_future = executor.submit(falsify_claims, claims)
            verifier_report = verifier_future.result()
            falsifier_report = falsifier_future.result()
        ver_ev = sum(len(r["evidence"]) for r in verifier_report["claim_reports"])
        fal_ev = sum(len(r["evidence"]) for r in falsifier_report["claim_reports"])
        st.write(f"✅ Gathered **{ver_ev + fal_ev}** evidence items ({ver_ev} support, {fal_ev} rebuttal).")

        st.write("⏳ Step 3/4 — Judge weighing both cases...")
        verdict = judge_debate(verifier_report, falsifier_report)
        st.write(f"✅ Verdict: **{verdict.get('overall_verdict', '?')}** ({verdict.get('overall_confidence', 0):.0%} confidence)")

        st.write("✅ Step 4/4 — Rendering results.")
        status.update(label="Analysis complete ✓", state="complete", expanded=False)

    # Cache results
    results = {
        "claims": claims,
        "verifier_report": verifier_report,
        "falsifier_report": falsifier_report,
        "verdict": verdict,
        "article_preview": article_text[:200],
    }
    st.session_state["last_hash"] = article_hash
    st.session_state["last_results"] = results

    st.session_state.setdefault("history", []).append({"verdict": verdict.get("overall_verdict", "?"), "preview": article_text[:54] + "..."})
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    _render_claims(claims)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    _render_verdict(verdict)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    _render_debate(verifier_report, falsifier_report, claims)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    _render_breakdown(verdict)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    _render_stats(verifier_report, falsifier_report, verdict, claims)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    _render_export(results)


def _render_claims(claims: list[dict]) -> None:
    rows = []
    for idx, claim in enumerate(claims, start=1):
        importance = str(claim.get("importance", "medium")).lower()
        if importance not in {"low", "medium", "high"}:
            importance = "medium"
        rows.append(f'<div class="row"><div class="idx">{idx}</div><div class="text">{_escape(claim.get("claim",""))}</div><div class="tag {importance}">{_escape(importance)}</div><div></div></div>')
    st.markdown(f'<section class="section"><div class="head"><div><div class="label">Claim extraction</div><div class="title">The article reduced to its checkable spine.</div></div><div class="copy">These are the factual statements the system judged worth verifying independently.</div></div><div class="claim">{"".join(rows)}</div></section>', unsafe_allow_html=True)


def _render_verdict(verdict: dict) -> None:
    overall = verdict.get("overall_verdict", "MISLEADING")
    if overall not in {"REAL", "FAKE", "MISLEADING"}:
        overall = "MISLEADING"
    key = overall.lower()
    confidence = max(0.0, min(float(verdict.get("overall_confidence", 0.5)), 1.0))
    metrics = _confidence_metrics(verdict)
    real_pct = metrics["REAL"] * 100
    fake_pct = metrics["FAKE"] * 100
    misleading_pct = metrics["MISLEADING"] * 100
    st.markdown(
        f'<section class="section"><div class="head"><div><div class="label">Article verdict</div><div class="title">Fast call, visible confidence.</div></div><div class="copy">The final call now comes from claim-level scoring, and the metric bar shows how strongly the article leans real, fake, or misleading.</div></div><div class="verdict"><div class="label">Final call</div><div class="v-word {key}">{overall}</div><div class="copy" style="margin-top:.45rem;color:var(--text);font-weight:700">{confidence:.0%} confidence</div><div class="track"><span class="{key}" style="width:{confidence*100:.0f}%"></span></div><div class="metric-stack"><div class="metric-bar"><span class="real" style="width:{real_pct:.1f}%"></span><span class="fake" style="width:{fake_pct:.1f}%"></span><span class="misleading" style="width:{misleading_pct:.1f}%"></span></div><div class="metric-grid"><div class="metric-item real"><strong>Real</strong><span>{real_pct:.0f}%</span></div><div class="metric-item fake"><strong>Fake</strong><span>{fake_pct:.0f}%</span></div><div class="metric-item misleading"><strong>Misleading</strong><span>{misleading_pct:.0f}%</span></div></div></div><div class="v-copy">{_escape(verdict.get("summary",""))}</div></div></section>',
        unsafe_allow_html=True,
    )
    if verdict.get("reasoning"):
        with st.expander("Judge reasoning", expanded=False):
            st.markdown(verdict["reasoning"])


def _evidence_html(items: list[dict]) -> str:
    blocks = []
    for item in items[:3]:
        link = _safe_url(item.get("url", ""))
        link_html = f'<a class="e-link" href="{link}" target="_blank" rel="noopener noreferrer">Open source</a>' if link else ""
        blocks.append(
            f'<div class="e-item"><div class="e-head"><div class="e-title">{_escape(item.get("title","Untitled source"))}</div><div class="e-meta">{_escape(item.get("stance","NEUTRAL"))} {float(item.get("confidence",0)):.0%}</div></div><div class="e-copy">{_escape((item.get("full_text") or item.get("snippet") or "")[:220])}</div>{link_html}</div>'
        )
    return "".join(blocks) or '<div class="e-copy">No strong evidence surfaced in the top results.</div>'


def _render_debate(ver: dict, fal: dict, claims: list[dict]) -> None:
    st.markdown('<section class="section"><div class="head"><div><div class="label">Debate surface</div><div class="title">Two retrieval paths, one claim at a time.</div></div><div class="copy">Open any claim to inspect the support and rebuttal cases side by side.</div></div></section>', unsafe_allow_html=True)
    for i, claim in enumerate(claims):
        ver_rep = ver["claim_reports"][i] if i < len(ver["claim_reports"]) else {}
        fal_rep = fal["claim_reports"][i] if i < len(fal["claim_reports"]) else {}
        with st.expander(f"Claim {i + 1}: {claim['claim'][:92]}", expanded=(i == 0)):
            st.markdown(
                f'<div class="debate"><div class="face ver"><div class="face-head"><div class="face-title">Verifier</div><div class="face-tone">Support case</div></div><div class="debate-body">{_escape(ver_rep.get("argument",""))}</div><div class="evidence">{_evidence_html(ver_rep.get("supporting_evidence", []))}</div></div><div class="face fal"><div class="face-head"><div class="face-title">Falsifier</div><div class="face-tone">Counter case</div></div><div class="debate-body">{_escape(fal_rep.get("argument",""))}</div><div class="evidence">{_evidence_html(fal_rep.get("contradicting_evidence", []))}</div></div></div>',
                unsafe_allow_html=True,
            )


def _render_breakdown(verdict: dict) -> None:
    rows = []
    for idx, item in enumerate(verdict.get("claim_verdicts", []), start=1):
        label = item.get("verdict", "UNVERIFIABLE")
        css = {"SUPPORTED": "supported", "REFUTED": "refuted", "UNVERIFIABLE": "unverifiable"}.get(label, "unverifiable")
        rows.append(f'<div class="row"><div class="idx">{idx}</div><div><div class="text">{_escape(item.get("claim","")[:140])}</div><div class="reason">{_escape(item.get("reasoning","")[:260])}</div></div><div class="pill {css}">{_escape(label)}</div><div class="conf">{float(item.get("confidence",0)):.0%}</div></div>')
    st.markdown(f'<section class="section"><div class="head"><div><div class="label">Per-claim outcomes</div><div class="title">Where the story holds and where it breaks.</div></div><div class="copy">The claim-by-claim view is usually more useful than the headline verdict when an article mixes fact and fiction.</div></div><div class="break">{"".join(rows)}</div></section>', unsafe_allow_html=True)


def _render_stats(ver: dict, fal: dict, verdict: dict, claims: list[dict]) -> None:
    supported = sum(1 for item in verdict.get("claim_verdicts", []) if item.get("verdict") == "SUPPORTED")
    total_evidence = sum(len(r["evidence"]) for r in ver["claim_reports"]) + sum(len(r["evidence"]) for r in fal["claim_reports"])
    strong_signals = sum(r["evidence_count"] for r in ver["claim_reports"]) + sum(r["evidence_count"] for r in fal["claim_reports"])
    st.markdown(
        f'<section class="section"><div class="head"><div><div class="label">Analysis metrics</div><div class="title">A quick read on scope and evidence density.</div></div><div class="copy">These numbers show how much material the system actually worked with before issuing the final call.</div></div><div class="stats"><div class="slot"><strong>{len(claims)}</strong><span>claims analyzed</span></div><div class="slot"><strong>{supported}</strong><span>supported claims</span></div><div class="slot"><strong>{strong_signals}</strong><span>strong signals</span></div><div class="slot"><strong>{total_evidence}</strong><span>evidence items</span></div></div></section>',
        unsafe_allow_html=True,
    )

def _render_export(results: dict) -> None:
    """Fix #12: Allow users to download the full analysis as JSON."""
    st.markdown(
        '<section class="section"><div class="head"><div>'
        '<div class="label">Export</div>'
        '<div class="title">Save this analysis for later.</div>'
        '</div><div class="copy">Download the full report including claims, evidence, arguments, and verdicts as a JSON file.</div>'
        '</div></section>',
        unsafe_allow_html=True,
    )

    # Build a clean export dict (strip full_text to keep file size reasonable)
    export = {
        "verdict": results["verdict"],
        "claims": results["claims"],
        "article_preview": results.get("article_preview", ""),
        "verifier_summary": results["verifier_report"].get("overall_assessment", ""),
        "falsifier_summary": results["falsifier_report"].get("overall_assessment", ""),
        "claim_details": [],
    }
    for i, claim in enumerate(results["claims"]):
        ver_r = results["verifier_report"]["claim_reports"][i] if i < len(results["verifier_report"]["claim_reports"]) else {}
        fal_r = results["falsifier_report"]["claim_reports"][i] if i < len(results["falsifier_report"]["claim_reports"]) else {}
        export["claim_details"].append({
            "claim": claim.get("claim", ""),
            "importance": claim.get("importance", "medium"),
            "verifier_argument": ver_r.get("argument", ""),
            "verifier_evidence_count": ver_r.get("evidence_count", 0),
            "falsifier_argument": fal_r.get("argument", ""),
            "falsifier_evidence_count": fal_r.get("evidence_count", 0),
        })

    json_str = json_lib.dumps(export, indent=2, ensure_ascii=False)
    st.download_button(
        label="Download report (JSON)",
        data=json_str,
        file_name="fake_news_debater_report.json",
        mime="application/json",
        use_container_width=True,
    )


def _render_footer() -> None:
    st.markdown(
        '<div class="footer-wrap"><div>Fake News Debater - adversarial multi-agent article review.</div><div>Powered by <a href="https://groq.com" target="_blank" rel="noopener noreferrer">Groq</a>, <a href="https://spacy.io" target="_blank" rel="noopener noreferrer">spaCy</a>, and live web retrieval.</div></div>',
        unsafe_allow_html=True,
    )


def _render_claims_v2(claims: list[dict]) -> None:
    rows = []
    for idx, claim in enumerate(claims, start=1):
        importance = str(claim.get("importance", "medium")).lower()
        if importance not in {"low", "medium", "high"}:
            importance = "medium"
        entity_count = len(claim.get("entities", []))
        entity_copy = f"{entity_count} named entities" if entity_count else "No entities retained"
        rows.append(
            f'<div class="row"><div class="idx">{idx}</div><div><div class="text">{_escape(claim.get("claim",""))}</div>'
            f'<div class="reason">{_escape(entity_copy)}</div></div><div class="tag {importance}">{_escape(importance)}</div><div></div></div>'
        )
    st.markdown(
        f'<section class="section"><div class="head"><div><div class="label">Claim extraction</div><div class="title">The article reduced to its checkable spine.</div></div><div class="copy">These are the factual statements the system judged worth verifying independently.</div></div><div class="claim">{"".join(rows)}</div></section>',
        unsafe_allow_html=True,
    )


def _render_verdict_v2(verdict: dict, health: dict) -> None:
    overall = verdict.get("overall_verdict", "MISLEADING")
    if overall not in {"REAL", "FAKE", "MISLEADING"}:
        overall = "MISLEADING"
    key = overall.lower()
    confidence = max(0.0, min(_as_float(verdict.get("overall_confidence", 0.5)), 1.0))
    metrics = _confidence_metrics(verdict)
    real_pct = metrics["REAL"] * 100
    fake_pct = metrics["FAKE"] * 100
    misleading_pct = metrics["MISLEADING"] * 100
    tone = _escape(health.get("tone", "warn"))
    health_summary = _escape(health.get("summary", "No run-health summary available."))
    health_label = _escape(health.get("label", "Unknown"))
    summary = _escape(verdict.get("summary", ""))
    st.markdown(
        f'<section class="section"><div class="head"><div><div class="label">Article verdict</div><div class="title">Fast call, visible confidence.</div></div><div class="copy">The final call is scored from the claim outcomes, and the metric bar shows how strongly the article leans real, fake, or misleading.</div></div><div class="verdict"><div class="label">Final call</div><div class="v-word {key}">{overall}</div><div class="copy" style="margin-top:.45rem;color:var(--text);font-weight:700">{confidence:.0%} confidence</div><div class="track"><span class="{key}" style="width:{confidence*100:.0f}%"></span></div><div class="metric-stack"><div class="metric-bar"><span class="real" style="width:{real_pct:.1f}%"></span><span class="fake" style="width:{fake_pct:.1f}%"></span><span class="misleading" style="width:{misleading_pct:.1f}%"></span></div><div class="metric-grid"><div class="metric-item real"><strong>Real</strong><span>{real_pct:.0f}%</span></div><div class="metric-item fake"><strong>Fake</strong><span>{fake_pct:.0f}%</span></div><div class="metric-item misleading"><strong>Misleading</strong><span>{misleading_pct:.0f}%</span></div></div></div><div class="callout {tone}"><div style="display:flex;justify-content:space-between;gap:1rem;align-items:center;flex-wrap:wrap"><div><div class="label">Run health</div><div class="reason">{health_summary}</div></div><div class="health-pill {tone}">{health_label}</div></div></div><div class="v-copy">{summary}</div></div></section>',
        unsafe_allow_html=True,
    )
    if verdict.get("reasoning"):
        with st.expander("Judge reasoning", expanded=False):
            st.markdown(verdict["reasoning"])


def _evidence_html_v2(items: list[dict]) -> str:
    blocks = []
    for item in items[:3]:
        link = _safe_url(item.get("url", ""))
        link_html = f'<a class="e-link" href="{link}" target="_blank" rel="noopener noreferrer">Open source</a>' if link else ""
        meta_bits = [f'{_escape(item.get("stance","NEUTRAL"))} {_as_float(item.get("confidence",0)):.0%}']
        if item.get("scraped"):
            meta_bits.append("scraped")
        if item.get("provider"):
            meta_bits.append(_escape(item.get("provider", "")))
        if item.get("used_fallback"):
            meta_bits.append("fallback")
        meta = " | ".join(meta_bits)
        blocks.append(
            f'<div class="e-item"><div class="e-head"><div class="e-title">{_escape(item.get("title","Untitled source"))}</div><div class="e-meta">{meta}</div></div><div class="e-copy">{_escape((item.get("full_text") or item.get("snippet") or "")[:220])}</div>{link_html}</div>'
        )
    return "".join(blocks) or '<div class="e-copy">No strong evidence surfaced in the top results.</div>'


def _render_signal_board_v2(health: dict) -> None:
    tone = _escape(health.get("tone", "warn"))
    health_summary = _escape(health.get("summary", "No run-health summary available."))
    health_label = _escape(health.get("label", "Unknown"))
    engine = _escape(health.get("engine", "Unknown"))
    st.markdown(
        f'<section class="section"><div class="head"><div><div class="label">System health</div><div class="title">How clean was this run?</div></div><div class="copy">This shows whether the pipeline surfaced enough usable evidence and whether stance classification had to fall back to neutral.</div></div><div class="callout {tone}"><div style="display:flex;justify-content:space-between;gap:1rem;align-items:center;flex-wrap:wrap"><div><div class="label">Pipeline status</div><div class="reason">{health_summary}</div></div><div class="health-pill {tone}">{health_label}</div></div></div><div class="health-grid"><div class="health-card"><strong>Engine</strong><span>{engine}</span><small>stance classifier</small></div><div class="health-card"><strong>Scraped Sources</strong><span>{health["scraped_sources"]}</span><small>pages extracted</small></div><div class="health-card"><strong>Usable Signals</strong><span>{health["usable_signals"]}</span><small>support or contradiction</small></div><div class="health-card"><strong>Fallback Neutrals</strong><span>{health["fallback_classifications"]}</span><small>degraded classifications</small></div></div></section>',
        unsafe_allow_html=True,
    )


def _render_debate_v2(ver: dict, fal: dict, claims: list[dict]) -> None:
    st.markdown(
        '<section class="section"><div class="head"><div><div class="label">Debate surface</div><div class="title">Two retrieval paths, one claim at a time.</div></div><div class="copy">Open any claim to inspect the support and rebuttal cases side by side, including source-level evidence metadata.</div></div></section>',
        unsafe_allow_html=True,
    )
    for i, claim in enumerate(claims):
        ver_rep = ver["claim_reports"][i] if i < len(ver["claim_reports"]) else {}
        fal_rep = fal["claim_reports"][i] if i < len(fal["claim_reports"]) else {}
        with st.expander(f"Claim {i + 1}: {claim.get('claim', '')[:92]}", expanded=(i == 0)):
            st.markdown(
                f'<div class="debate"><div class="face ver"><div class="face-head"><div class="face-title">Verifier</div><div class="face-tone">Support case</div></div><div class="debate-body">{_escape(ver_rep.get("argument",""))}</div><div class="evidence">{_evidence_html_v2(ver_rep.get("supporting_evidence", []))}</div></div><div class="face fal"><div class="face-head"><div class="face-title">Falsifier</div><div class="face-tone">Counter case</div></div><div class="debate-body">{_escape(fal_rep.get("argument",""))}</div><div class="evidence">{_evidence_html_v2(fal_rep.get("contradicting_evidence", []))}</div></div></div>',
                unsafe_allow_html=True,
            )


def _render_breakdown_v2(verdict: dict) -> None:
    rows = []
    for idx, item in enumerate(verdict.get("claim_verdicts", []), start=1):
        label = item.get("verdict", "UNVERIFIABLE")
        css = {"SUPPORTED": "supported", "REFUTED": "refuted", "UNVERIFIABLE": "unverifiable"}.get(label, "unverifiable")
        rows.append(
            f'<div class="row"><div class="idx">{idx}</div><div><div class="text">{_escape(item.get("claim","")[:140])}</div><div class="reason">{_escape(item.get("reasoning","")[:260])}</div></div><div class="pill {css}">{_escape(label)}</div><div class="conf">{_as_float(item.get("confidence",0)):.0%}</div></div>'
        )
    body = "".join(rows) or '<div class="reason">No claim-level verdicts were produced.</div>'
    st.markdown(
        f'<section class="section"><div class="head"><div><div class="label">Per-claim outcomes</div><div class="title">Where the story holds and where it breaks.</div></div><div class="copy">The claim-by-claim view is usually more useful than the headline verdict when an article mixes fact and fiction.</div></div><div class="break">{body}</div></section>',
        unsafe_allow_html=True,
    )


def _render_stats_v2(ver: dict, fal: dict, verdict: dict, claims: list[dict], health: dict) -> None:
    supported = sum(1 for item in verdict.get("claim_verdicts", []) if item.get("verdict") == "SUPPORTED")
    total_evidence = sum(len(r["evidence"]) for r in ver["claim_reports"]) + sum(len(r["evidence"]) for r in fal["claim_reports"])
    st.markdown(
        f'<section class="section"><div class="head"><div><div class="label">Analysis metrics</div><div class="title">A quick read on scope and evidence density.</div></div><div class="copy">These numbers show how much material the system actually worked with before issuing the final call.</div></div><div class="stats"><div class="slot"><strong>{len(claims)}</strong><span>claims analyzed</span></div><div class="slot"><strong>{supported}</strong><span>supported claims</span></div><div class="slot"><strong>{health["scraped_sources"]}</strong><span>scraped sources</span></div><div class="slot"><strong>{health["usable_signals"]}</strong><span>usable signals</span></div><div class="slot"><strong>{health["fallback_classifications"]}</strong><span>fallback neutrals</span></div><div class="slot"><strong>{total_evidence}</strong><span>evidence items</span></div></div></section>',
        unsafe_allow_html=True,
    )


def _render_export_v2(results: dict) -> None:
    health = _ensure_analysis_health(results)
    st.markdown(
        '<section class="section"><div class="head"><div><div class="label">Export</div><div class="title">Save this analysis for later.</div></div><div class="copy">Download the full report including claims, evidence, verdicts, and run-health metadata as JSON.</div></div></section>',
        unsafe_allow_html=True,
    )
    export = {
        "verdict": results["verdict"],
        "analysis_health": health,
        "claims": results["claims"],
        "article_preview": results.get("article_preview", ""),
        "verifier_summary": results["verifier_report"].get("overall_assessment", ""),
        "falsifier_summary": results["falsifier_report"].get("overall_assessment", ""),
        "claim_details": [],
    }
    for i, claim in enumerate(results["claims"]):
        ver_r = results["verifier_report"]["claim_reports"][i] if i < len(results["verifier_report"]["claim_reports"]) else {}
        fal_r = results["falsifier_report"]["claim_reports"][i] if i < len(results["falsifier_report"]["claim_reports"]) else {}
        export["claim_details"].append({
            "claim": claim.get("claim", ""),
            "importance": claim.get("importance", "medium"),
            "verifier_argument": ver_r.get("argument", ""),
            "verifier_evidence_count": ver_r.get("evidence_count", 0),
            "verifier_classification_failures": ver_r.get("classification_failures", 0),
            "falsifier_argument": fal_r.get("argument", ""),
            "falsifier_evidence_count": fal_r.get("evidence_count", 0),
            "falsifier_classification_failures": fal_r.get("classification_failures", 0),
        })
    json_str = json_lib.dumps(export, indent=2, ensure_ascii=False)
    st.download_button(
        label="Download report (JSON)",
        data=json_str,
        file_name="fake_news_debater_report.json",
        mime="application/json",
        use_container_width=True,
    )


def _render_analysis_results_v2(results: dict) -> None:
    health = _ensure_analysis_health(results)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    _render_claims_v2(results["claims"])
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    _render_verdict_v2(results["verdict"], health)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    _render_signal_board_v2(health)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    _render_debate_v2(results["verifier_report"], results["falsifier_report"], results["claims"])
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    _render_breakdown_v2(results["verdict"])
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    _render_stats_v2(results["verifier_report"], results["falsifier_report"], results["verdict"], results["claims"], health)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    _render_export_v2(results)


def _run_analysis_v2(article_text: str) -> None:
    article_hash = hashlib.sha256(article_text.encode()).hexdigest()[:16]
    if st.session_state.get("last_hash") == article_hash and st.session_state.get("last_results"):
        st.info("Showing cached results for this article. Change the text to re-run the analysis.")
        _render_analysis_results_v2(st.session_state["last_results"])
        return

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    with st.status("Running multi-agent analysis...", expanded=True) as status:
        st.write("Step 1/4 - Extracting factual claims with spaCy and Groq...")
        claims = extract_claims(article_text)
        if not claims:
            st.error("Could not extract any usable claims from this article.")
            status.update(label="Analysis failed", state="error")
            return
        st.write(f"Step 1 complete - Extracted {len(claims)} verifiable claims.")

        st.write("Step 2/4 - Verifier and falsifier are collecting evidence...")
        with ThreadPoolExecutor(max_workers=2) as executor:
            verifier_future = executor.submit(verify_claims, claims)
            falsifier_future = executor.submit(falsify_claims, claims)
            verifier_report = verifier_future.result()
            falsifier_report = falsifier_future.result()

        health = _analysis_health(verifier_report, falsifier_report)
        st.write(
            f"Step 2 complete - Gathered {health['total_evidence']} evidence items from "
            f"{health['scraped_sources']} scraped sources."
        )
        if health["fallback_classifications"]:
            st.write(
                f"Note - {health['fallback_classifications']} evidence items fell back to neutral after "
                "stance-classification failures."
            )

        st.write("Step 3/4 - Judge is scoring the claim outcomes...")
        verdict = judge_debate(verifier_report, falsifier_report)
        st.write(
            f"Step 3 complete - Verdict: {verdict.get('overall_verdict', '?')} "
            f"({verdict.get('overall_confidence', 0):.0%} confidence)."
        )

        st.write("Step 4/4 - Rendering the report.")
        status.update(label="Analysis complete", state="complete", expanded=False)

    results = {
        "claims": claims,
        "verifier_report": verifier_report,
        "falsifier_report": falsifier_report,
        "verdict": verdict,
        "analysis_health": health,
        "article_preview": article_text[:200],
    }
    st.session_state["last_hash"] = article_hash
    st.session_state["last_results"] = results
    st.session_state.setdefault("history", []).append(
        {
            "verdict": verdict.get("overall_verdict", "?"),
            "confidence": _as_float(verdict.get("overall_confidence", 0.0)),
            "preview": article_text[:54] + "...",
        }
    )
    _render_analysis_results_v2(results)


def _render_workspace_v2() -> None:
    st.markdown(
        """
        <section class="section">
            <div class="head">
                <div><div class="label">Primary workspace</div><div class="title">Load an article and run the debate.</div></div>
                <div class="copy">Paste full text for speed or fetch a live URL when you want the article body extracted automatically.</div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    prefill = st.session_state.get("sample_text", "")
    if prefill:
        st.info(f"Loaded sample: {st.session_state.get('sample_loaded', 'Sample')}")

    left, right = st.columns([1.12, 0.78], gap="large")
    article_text = ""
    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        tab_text, tab_url = st.tabs(["Paste text", "From URL"])
        with tab_text:
            article_text = st.text_area(
                "Article text",
                value=prefill,
                height=280,
                placeholder="Paste the article body here.",
                label_visibility="collapsed",
                key="article_input_v2",
            )
            if article_text:
                char_count = len(article_text)
                if char_count < MIN_ARTICLE_LENGTH:
                    st.caption(f"Input length: {char_count:,} characters. Minimum required: {MIN_ARTICLE_LENGTH}.")
                elif char_count > MAX_ARTICLE_LENGTH:
                    st.caption(f"Input length: {char_count:,} characters. Maximum allowed: {MAX_ARTICLE_LENGTH:,}.")
                else:
                    st.caption(f"Input length: {char_count:,} characters.")
            if prefill:
                st.session_state.pop("sample_text", None)
        with tab_url:
            url = st.text_input("URL", key="article_url_input_v2", placeholder="https://example.com/news-article", label_visibility="collapsed")
            fetch_col, clear_col = st.columns([1, 0.7])
            with fetch_col:
                fetch_url = st.button("Fetch article", key="fetch_article_v2", use_container_width=True)
            with clear_col:
                clear_url = st.button("Clear fetched URL", key="clear_url_article_v2", use_container_width=True)

            if clear_url:
                _clear_loaded_url()
                st.rerun()
            if fetch_url:
                if not url.strip():
                    st.warning("Enter a URL before fetching.")
                else:
                    with st.spinner("Scraping article..."):
                        result = scrape_article(url)
                    if result["success"]:
                        st.session_state["url_article_text"] = result["text"]
                        st.session_state["url_article_title"] = result["title"]
                        st.session_state.pop("url_article_error", None)
                    else:
                        st.session_state["url_article_error"] = result["error"]

            if st.session_state.get("url_article_text"):
                fetched_text = st.session_state["url_article_text"]
                article_text = fetched_text
                st.success(f"Loaded: {st.session_state.get('url_article_title', 'Article')} | {len(fetched_text):,} chars")
                with st.expander("Preview extracted text", expanded=False):
                    st.text(fetched_text[:1000] + ("..." if len(fetched_text) > 1000 else ""))
            elif st.session_state.get("url_article_error"):
                st.error(st.session_state["url_article_error"])

        typed_text = st.session_state.get("article_input_v2", "").strip()
        article_text = typed_text or st.session_state.get("url_article_text", "")
        c1, c2, c3 = st.columns([1, 1.2, 1])
        with c2:
            analyze = st.button("Run analysis", use_container_width=True, type="primary", key="run_analysis_v2")
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown(
            """
            <div class="panel side">
                <div class="info"><strong>What happens</strong><span>The system extracts claims, retrieves supporting and opposing evidence, then asks a judge to score both cases.</span></div>
                <div class="info"><strong>Best input</strong><span>Articles with named people, organizations, dates, figures, and checkable events work best.</span></div>
                <div class="info"><strong>What to watch</strong><span>Thin evidence, degraded classifications, and mixed claim outcomes usually matter more than the headline label.</span></div>
                <div class="flow">
                    <div class="flow-item"><strong>Claims</strong><span>Reduce the article to the statements worth checking.</span></div>
                    <div class="flow-item"><strong>Evidence</strong><span>Pull supporting and rebutting sources before the verdict.</span></div>
                    <div class="flow-item"><strong>Health</strong><span>See whether the run had clean signal or degraded classifications.</span></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        for label, text in SAMPLE_ARTICLES.items():
            if st.button(label, key=f"work_v2_{label}", use_container_width=True):
                _load_sample(label, text)
                st.rerun()

    st.markdown(
        """
        <section class="section">
            <div class="head">
                <div><div class="label">Reading the output</div><div class="title">One interface, four jobs.</div></div>
                <div class="copy">Claims isolate the article, the verdict compresses it, system health shows whether the run was clean, and the debate reveals exactly where the story holds or fails.</div>
            </div>
            <div class="strip">
                <div class="strip-item"><strong>Claims</strong><span>See what the system believes is factual and checkable.</span></div>
                <div class="strip-item"><strong>Verdict</strong><span>Get a fast read, plus a score distribution instead of a single opaque label.</span></div>
                <div class="strip-item"><strong>Health</strong><span>Spot degraded evidence classification before you over-trust the final call.</span></div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    if analyze and article_text:
        if len(article_text) < MIN_ARTICLE_LENGTH:
            st.warning(f"Article is too short ({len(article_text)} chars). Please provide at least {MIN_ARTICLE_LENGTH} characters.")
        elif len(article_text) > MAX_ARTICLE_LENGTH:
            st.warning(f"Article is too long ({len(article_text):,} chars). Please shorten it to {MAX_ARTICLE_LENGTH:,} characters.")
        else:
            _run_analysis_v2(article_text)
    elif analyze:
        st.warning("Paste article text or fetch a URL first.")


def _render_footer_v2() -> None:
    st.markdown(
        '<div class="footer-wrap"><div>Fake News Debater | adversarial multi-agent article review.</div><div>Powered by <a href="https://groq.com" target="_blank" rel="noopener noreferrer">Groq</a>, <a href="https://spacy.io" target="_blank" rel="noopener noreferrer">spaCy</a>, and live web retrieval.</div></div>',
        unsafe_allow_html=True,
    )
def main() -> None:
    _render_sidebar()
    _render_hero()
    try:
        validate_config()
    except EnvironmentError as exc:
        st.error(f"Configuration error: {exc}")
        st.stop()
    _render_workspace_v2()
    _render_footer_v2()


if __name__ == "__main__":
    main()
