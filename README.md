# Fake News Debater

**Multi-agent misinformation analysis with adversarial retrieval, Groq-based reasoning, and explainable verdicts**

Fake News Debater is a Streamlit app that breaks an article into factual claims, sends two agents to argue opposite sides with web evidence, and then produces both claim-level and article-level verdicts with visible confidence metrics.

---

## Architecture

```text
User submits article text or URL
         |
         v
   [Claim Extractor]          <- spaCy NER + Groq
         |
    _____|_____
    |         |
    v         v
[Verifier]  [Falsifier]       <- Independent search + scraping
 Agent       Agent               + Groq stance classification
    |         |
 Evidence   Evidence
    |_________|
         |
         v
   [Judge Agent]              <- Claim-level verdicts
         |                       + deterministic article scoring
         v
 Final Verdict + Report       <- JSON export + confidence metric bar
```

### Current pipeline

1. **Claim extraction**
Uses spaCy entity extraction plus Groq to generate 3 to 5 verifiable claims from the article.

2. **Verifier / falsifier debate**
Each claim is searched independently. Top results are scraped, then each evidence passage is classified by Groq as `SUPPORT`, `CONTRADICT`, or `NEUTRAL`.

3. **Claim-level judging**
The judge reviews both sides and returns `SUPPORTED`, `REFUTED`, or `UNVERIFIABLE` for each claim.

4. **Article-level verdict**
The final article label is now scored deterministically from the claim verdicts, producing:
- `REAL`
- `FAKE`
- `MISLEADING`

This avoids the old issue where a final free-form LLM call could overuse `MISLEADING`.

---

## Key Features

- **Adversarial retrieval:** verifier and falsifier search independently for supporting and contradicting evidence.
- **Groq-first pipeline:** claim extraction, stance classification, and judging all run through Groq.
- **Deterministic article scoring:** article verdicts are computed from claim outcomes instead of relying on a final unconstrained model guess.
- **Confidence visualization:** the UI shows both the winning confidence and a three-way metric bar for `REAL`, `FAKE`, and `MISLEADING`.
- **Parallel execution:** verifier and falsifier run concurrently, and evidence scoring is parallelized per claim.
- **Result caching:** repeated analysis of the same article is served from session cache.
- **JSON export:** full analysis can be downloaded for later review.

---

## Tech Stack

| Component | Tool / Library | Notes |
|-----------|----------------|-------|
| Reasoning | Groq API (`llama-3.3-70b-versatile`) | Claims, stance, and judging |
| Web Search | DuckDuckGo + Serper.dev | Serper optional |
| NLP / NER | spaCy (`en_core_web_sm`) | Local entity extraction |
| Frontend | Streamlit | Main UI |
| Scraping | `requests` + BeautifulSoup | Article and evidence extraction |
| Execution | Python 3.13 + `ThreadPoolExecutor` | Parallel agent work |

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/Draco-0704/debator.git
cd debator
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Configure environment

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
SERPER_API_KEY=your_serper_key
HF_API_TOKEN=optional_legacy_token
```

Notes:
- `GROQ_API_KEY` is required.
- `SERPER_API_KEY` is optional. Without it, the app falls back to DuckDuckGo.
- `HF_API_TOKEN` is no longer required for stance detection.

### 3. Run the app

```bash
streamlit run app.py
```

Open `http://localhost:8501`.

---

## Project Structure

```text
debater/
|-- agents/
|   |-- claim_extractor.py
|   |-- verifier_agent.py
|   |-- falsifier_agent.py
|   `-- judge_agent.py
|-- tools/
|   |-- article_scraper.py
|   |-- groq_client.py
|   |-- stance_detector.py
|   `-- web_search.py
|-- tests/
|   |-- test_judge_agent.py
|   |-- test_overall_verdict.py
|   `-- test_stance_detector.py
|-- app.py
|-- config.py
|-- requirements.txt
`-- README.md
```

---

## Testing

Run the unit tests with:

```bash
py -m unittest discover tests -v
```

Current tests cover:
- stance result parsing
- evidence formatting for the judge
- deterministic article-level verdict scoring

---

## Notes on Confidence

- **Evidence confidence** comes from the stance classifier output.
- **Claim confidence** is based on the supporting or contradicting evidence that each side finds.
- **Article confidence** is computed from the scored distribution across `REAL`, `FAKE`, and `MISLEADING`.

Because the article verdict is now tied to claim-level outcomes, the displayed label should better match the visible evidence.

---

## License

MIT License.
