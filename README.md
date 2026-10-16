# Small vs Large LLMs: cost-aware routing for intent classification

> A company must classify user requests. A small model is cheap and fast but errs more; a large one is accurate but costs more. Can easy requests go to the small model, with only the hard ones escalated, keeping most of the quality at a fraction of the cost? This project builds and tests exactly that, against fair baselines, and measures where it helps and where it doesn't.


This README explains how the repository is organised and how to reproduce the results.

## Task and setup

- **Task:** intent classification on [MASSIVE](https://huggingface.co/datasets/AmazonScience/massive), English-US portion: one short request in, one of 60 intents out.
- **Systems** (all scored on the same 150 validation and 150 test requests):
  - **S0:** TF-IDF + logistic regression (non-LLM baseline, trained on 11,514 labelled requests)
  - **A:** `openai/gpt-oss-20b` alone (small model)
  - **B:** `openai/gpt-oss-120b` alone (large model)
  - **C:** router. It runs the small model on everything and escalates to the large model when the small model's output is invalid or its label disagrees with S0's.
  - **D:** random escalation at the same number of escalated requests as C (control)
- **Models** run zero-shot through the Groq API (free plan, `reasoning_effort = "low"`). Cost is **estimated** from published per-token prices.
- **Protocol:** the router's configuration was chosen on validation by a rule written in advance, frozen in git, and scored **once** on test. Uncertainty comes from paired bootstrap confidence intervals.

## Main results (test, 150 requests, frozen configuration, scored once)

| System | Accuracy | Macro-F1 | Cost per 1,000 requests (USD, estimated) | Escalated |
|---|---|---|---|---|
| S0 (TF-IDF + LR) | 0.800 | 0.791 | n/a | n/a |
| A: small only | 0.700 | 0.769 | 0.042 | n/a |
| B: large only | 0.780 | 0.829 | 0.081 | n/a |
| **C: router** | 0.773 | 0.815 | 0.072 | 34% (51 of 150) |
| D: random escalation | 0.727 | 0.785 | 0.070 | 34% |

Paired bootstrap, 95% intervals (5,000 resamples):

| Quantity | Estimate | 95% interval | Reading |
|---|---|---|---|
| C minus D (accuracy) | +0.046 | [+0.011, +0.080] | clear |
| C minus A (accuracy) | +0.073 | [+0.020, +0.127] | clear |
| C minus B (accuracy) | -0.007 | [-0.033, +0.013] | cannot distinguish |
| C minus S0 (accuracy) | -0.027 | [-0.087, +0.040] | cannot distinguish |
| Cost saved vs B | 11.6% | [3.3%, 20.2%] | modest, wide |
| Share of B's accuracy kept | 99.1% | [96.0%, 101.8%] | consistent with all of it |

**In short:** the router keeps about 99% of the large model's accuracy at about 88% of its estimated cost, and it clearly beats both the small model and random escalation. It cannot be told apart from the large model or from S0 on this sample. The saving is modest, and the router's 95th-percentile latency is worse than the large model's (3.83 s vs 2.31 s) because escalated requests wait for two calls. S0 needs labelled data and the LLMs do not, which is the cold-start argument for the LLM-based systems.

## Limits (the notebook has the full list)

- Only 150 test requests, so intervals are wide and small differences are not claimed.
- Best of 60 configurations on 150 validation requests: the validation advantage over B did not carry over to test.
- Cost is estimated from published prices, and latency comes from a hosted API (network and queueing included).

## Repository layout

```
analysis.ipynb      data, method, results, error analysis, interpretation
config.py           constants: seed, sample sizes, model ids, prices
data.py             loads MASSIVE (en-US) and the frozen samples
s0.py               trains the S0 baseline, writes cache/s0_preds.csv
prompt.py           the zero-shot prompt (locked, see below)
llm.py              Groq wrapper: on-disk cache, request pacing, retry on rate limits
run_llm.py          runs A and B on validation and test, writes cache/llm_results.csv
samples.json        ids of the 150 + 150 sampled requests (tracked, never redrawn)
cache/              API answers and S0 predictions (NOT tracked by git)
.env                Groq API key 
```

## Setup (Windows, PowerShell)

```powershell
python -m venv llm
.\llm\Scripts\Activate.ps1          
pip install -r requirements.txt
```

For anything that calls the API, create a `.env` file in the project folder containing your Groq key (variable name: `GROQ_API_KEY`).


## References

- Chen, Zaharia, Zou (2024). *FrugalGPT.* TMLR. arXiv:2305.05176
- Ding et al. (2024). *Hybrid LLM: Cost-Efficient and Quality-Aware Query Routing.* ICLR. arXiv:2404.14618
- Ong et al. (2025). *RouteLLM: Learning to Route LLMs with Preference Data.* ICLR. arXiv:2406.18665
- FitzGerald et al. (2023). *MASSIVE.* ACL. · Bastianelli et al. (2020). *SLURP.* EMNLP.
- OpenAI (2025). *gpt-oss-120b and gpt-oss-20b Model Card.* arXiv:2508.10925

The full reference list is at the end of the notebook.

## Use of AI tools

An AI assistant (Claude, by Anthropic) was used substantially for design discussion, code, drafting text and finding references. I ran everything, checked the outputs, and made the final decisions. The full disclosure is in the notebook.