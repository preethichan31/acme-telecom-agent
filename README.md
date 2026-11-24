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
- **Azure OpenAI / Foundry:** `AZURE_OPENAI_ENDPOINT` (e.g. `https://<resource>.services.ai.azure.com/`), `AZURE_OPENAI_API_KEY`, and `AZURE_OPENAI_DEPLOYMENT` (your deployment name, e.g. `gpt-5.1`). When `AZURE_OPENAI_ENDPOINT` is set, it takes precedence over `OPENAI_API_KEY`, and the Agents SDK's built-in trace exporter to OpenAI is removed. Tracing itself stays on, so any trace processor you register later still receives traces.

## Try it

- *My calls keep dropping at home*: the agent searches the troubleshooting runbooks.
- *What's the balance on ACME-3310?*: the agent looks up the mock CRM account.

## Generate traffic

`traffic/generate_traffic.py` replays 47 scripted customer conversations (57 turns) through the agent, with history between turns, so you can fill a monitoring or observability tool with realistic data. The conversations in `traffic/conversations.json` are a deliberate mix, so checks have rows that should pass and rows that should fail:

| Category | What it contains |
|---|---|
| `happy_path` | Questions the knowledge base answers |
| `account_lookup` | Valid, suspended, unknown and lower-case account IDs |
| `kb_gap` | Questions the knowledge base can't answer (the agent should say so) |
| `pii` | Messages containing an SSN, a card number, an email, a phone and an address (all fake) |
| `prompt_injection` | Jailbreaks and attempts to read other customers' accounts |
| `competitor` | "Should I switch to a competitor?" |
| `abusive_off_topic` | Abusive or unrelated messages |
| `multi_turn` | 3 to 4 turn conversations that rely on earlier context |
| `stress` | Long or multi-part messages |

```bash
python3 traffic/generate_traffic.py --dry-run                       # list what would run
python3 traffic/generate_traffic.py                                 # one pass over everything
python3 traffic/generate_traffic.py --category pii --category prompt_injection
python3 traffic/generate_traffic.py --loop --interval 60            # steady trickle; Ctrl-C to stop
```

Every turn is a real model call, so a full pass costs about as much as 57 chat messages. The script reads the same `.env` as the app. If the Openlayer SDK is installed, it also tags each conversation with a user ID and a session ID.

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
├── traffic/
│   ├── generate_traffic.py         # Replays scripted conversations
│   └── conversations.json          # 47 test conversations by category
├── .streamlit/
│   └── config.toml                 # Headless mode, usage stats off
├── streamlit_app.py                # Streamlit chat UI
├── requirements.txt
└── .env.example                    # Environment variable template
```

All entry points call `TelecomSupportAgent.answer_question(question, history)` in `agent/telecom_support_agent.py`, which is the place to add any tracing or monitoring.
