# Acme Telecom Customer Support Agent (OpenAI Agents SDK)

A Streamlit app that diagnoses and resolves customer telecom issues (service outages, device problems, billing disputes, and account questions) using an **OpenAI Agents SDK** agent that calls tools. It runs on OpenAI (`gpt-4.1-mini` by default) or on an Azure OpenAI / Foundry deployment such as `gpt-5.1`.

## How it works

1. The customer describes their issue in the chat UI.
2. The agent decides which tools to call:
   - `search_knowledge_base`: TF-IDF retrieval over troubleshooting runbooks, device manuals, and account-service policies.
   - `lookup_customer_account`: account summary from a mock CRM (try `ACME-7782`, `ACME-3310`, `ACME-9921`).
   - `get_current_date`: today's date.
3. The agent grounds its answer in the tool results and replies to the customer.

## Quick start

> Prerequisites: Python 3.9+ and an OpenAI API key (or an Azure OpenAI deployment).

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in your model settings (see below)
streamlit run streamlit_app.py
```

Open the URL shown in your terminal (usually `http://localhost:8501`), pick a simulated customer in the sidebar, and describe your issue. The browser doesn't open on its own because `.streamlit/config.toml` runs Streamlit headless (this also skips its first-run email prompt and turns off its usage statistics).

## Model configuration

Set one of these in `.env`:

- **OpenAI:** `OPENAI_API_KEY`. The agent uses `gpt-4.1-mini`.
- **Azure OpenAI / Foundry:** `AZURE_OPENAI_ENDPOINT` (e.g. `https://<resource>.services.ai.azure.com/`), `AZURE_OPENAI_API_KEY`, and `AZURE_OPENAI_DEPLOYMENT` (your deployment name, e.g. `gpt-5.1`). When `AZURE_OPENAI_ENDPOINT` is set, it takes precedence over `OPENAI_API_KEY`, and the Agents SDK's built-in trace upload to OpenAI is turned off.

## Try it

- *My calls keep dropping at home*: the agent searches the troubleshooting runbooks.
- *What's the balance on ACME-3310?*: the agent looks up the mock CRM account.

## Project structure

```
.
├── agent/
│   └── telecom_support_agent.py    # Agent definition, tools, knowledge-base index
├── knowledge-base/
│   ├── troubleshooting_runbooks.txt
│   ├── device_manuals.txt
│   └── account_services.txt
├── prompts/
│   └── system_prompt.txt           # Agent system prompt
├── .streamlit/
│   └── config.toml                 # Headless mode, usage stats off
├── streamlit_app.py                # Streamlit chat UI
├── requirements.txt
└── .env.example                    # Environment variable template
```

All entry points call `TelecomSupportAgent.answer_question(question, history)` in `agent/telecom_support_agent.py`, which is the place to add any tracing or monitoring.
