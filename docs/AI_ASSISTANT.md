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

Not implemented, and deliberately not designed here.

[Issue #33](https://github.com/Dev-saxena11/SIH26137/issues/33) covers grounding
the assistant in the project's own documentation (README, PROJECT_STATUS.md, the
mathematical formulation) so it answers judge questions from project sources
rather than the model's general knowledge. That issue is assigned and its design
— retrieval strategy, chunking, ranking, storage, and whether it gets its own
`/api/chat` endpoint or reuses the existing one — is entirely open.

The only thing to know from this side: `_call_external_llm()` in
`app/core/assistant.py` builds a `system_parts` list (session context, then
instructions) and joins it into the system prompt. Retrieved passages would go
in that list. Nothing in this module constrains how they get there.

## Testing

```bash
python test_assistant_providers.py     # offline: providers + no-fabrication
```

No API key or network required. Live provider behaviour is best
checked by setting `OPENROUTER_API_KEY` and asking the dashboard a free-text
question.
