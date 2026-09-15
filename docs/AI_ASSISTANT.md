# AI Assistant — architecture & extension guide

How the in-dashboard assistant answers questions and how to configure an LLM
provider.

## Answer path

A question takes one of two routes:

```
POST /api/assistant/chat
        |
        +-- preset chip?  -->  local deterministic engine  (no API call)
        |
        +-- free text?    -->  LLM provider, if one is configured
                                    |  grounded with the Story Card
                                    |  (live session metrics)
                                    |
                                    +-- success -> LLM answer
                                    +-- failure -> local deterministic engine
```

The local engine is the floor: it needs no key, no network, and always
answers. Everything else is enrichment on top.

**Preset chips never call an LLM.** They map to fixed explanations, so an API
call would add latency and rate-limit pressure for no benefit.

## The no-fabrication rule

The assistant must never invent a performance figure. A plausible-looking
fabricated number is indistinguishable from a measured one to whoever is
reading it, which would misrepresent the benchmark results this project is
judged on.

This is enforced in three places, and anything added here must preserve it:

1. `_has_comparison_results()` gates every numeric claim in the local engine.
   With no run executed it returns `NO_RESULTS_NOTICE` rather than defaults.
2. The Story Card sent to the LLM states explicitly when no run has happened
   and instructs the model not to estimate figures.
3. The system prompt tells the model to quote context figures exactly, to say
   when a value was not measured, and to say so when a question isn't
   answerable from the supplied context.

Verified: asked "What percentage of time did we save? Give me the exact
numbers." with no run executed, the live LLM path answers that no metrics are
available rather than producing a number.

## Configuring a provider

All optional — see [`.env.example`](../.env.example). Providers are tried in
order: **OpenRouter → Gemini → OpenAI**, first configured one wins.

| Variable | Notes |
| :--- | :--- |
| `OPENROUTER_API_KEY` | Free models, no billing account. Get one at [openrouter.ai/keys](https://openrouter.ai/keys). |
| `OPENROUTER_MODEL` | Optional, defaults to `openrouter/free`. |
| `GEMINI_API_KEY` / `OPENAI_API_KEY` | Alternatives; not free. |

For the deployed backend, set these in the **Render dashboard → Environment**,
never in a committed file.

### Free models only

`OpenRouterProvider` rejects any model id that isn't `openrouter/free` or
suffixed `:free`, raising at construction. A paid model therefore can't be
enabled by a typo in an env var — the provider is skipped and the assistant
falls back to the local engine.

`openrouter/free` is an **auto-router** over whichever free models are
currently available, which is why it's the default: the assistant keeps working
when any single free model is retired or rate-limited.

### Why the token budget is 800, not 150

Free models on OpenRouter are frequently *reasoning* models that spend most of
their completion budget on hidden reasoning tokens before emitting visible
text. Measured against `openrouter/free`:

| `max_tokens` | reasoning tokens | visible output | result |
| ---: | ---: | ---: | :--- |
| 150 | 179 | 10 chars | truncated mid-word, `finish_reason=length` |
| 800 | 77 | 418 chars | complete answer, `finish_reason=stop` |

`DEFAULT_MAX_TOKENS = 800` in `llm_providers.py`. A reply that comes back both
truncated and shorter than `MIN_USABLE_REPLY_CHARS` is discarded in favour of
the local engine, so a starved reasoning model produces a good local answer
rather than a broken stub.

## Document grounding / RAG (issue #33)

Implemented in `app/core/rag.py`.

The chatbot grounds answers in the project's actual documentation rather than
allowing the model to free-associate on stage in front of judges.

### Indexed Sources
1. `README.md` — project overview, API workflow, key design notes, design tradeoffs.
2. `PROJECT_STATUS.md` — PS deliverable mapping, bottlenecks, algorithm status.
3. `docs/FORMULATION.md` — formal CVRPTW mathematical model, objective function, penalty formulation, QPSO delta-potential equations, jump-cap, and quantum hardware roadmap.
4. `docs/AI_ASSISTANT.md` — assistant architecture, provider priority, no-fabrication rule.
5. `DEPLOYMENT.md` — Render backend / Vercel frontend split deployment.
6. `data/impact_report.md` — fuel, CO2, and driver hour impact metrics.

### Retrieval & Generation Architecture
- **Chunking**: Hierarchical markdown parser splitting on headings (`#`, `##`, `###`), preserving section breadcrumbs and clean paragraph text.
- **Retriever**: Pure-Python Okapi BM25 index with heading boosting and multi-word phrase matching. Sub-millisecond execution, deterministic, zero external vector-DB dependencies (safe for Render's 512MB RAM tier).
- **Generation**:
  - If an LLM provider is configured (`OpenRouter`, `Gemini`, `OpenAI`), retrieved passages are injected into the prompt with strict anti-hallucination instructions.
  - If offline or unconfigured, the local engine synthesizes structured answers directly from the top passages, citing document and section titles.
  - Out-of-scope domain guardrails decline queries unrelated to the platform.
- **Endpoints**:
  - `POST /api/chat` — takes `ChatRequest(message, context, top_k)`, returns `ChatResponse(reply, sources, suggested_chips)`.
  - `GET /api/chat/status` — returns total chunk count and indexed document list.
  - `POST /api/assistant/chat` — existing in-dashboard assistant also queries the RAG index for free-text conceptual questions.

## Testing

```bash
python test_assistant_providers.py     # offline: providers + no-fabrication
python test_rag_chatbot.py             # offline: RAG indexing + FAQ retrieval + /api/chat
```

No API key or network required. Live provider behaviour is best
checked by setting `OPENROUTER_API_KEY` and asking free-text questions.
