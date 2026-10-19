# Ayushman Bharat PM-JAY WhatsApp Assistant

> ⚠️ **Unofficial student project.** Not affiliated with the Government of India or the National Health Authority.

A WhatsApp chatbot that answers questions about the **Ayushman Bharat Pradhan Mantri Jan Arogya Yojana (PM-JAY)** health insurance scheme, built as a student research project.

## What it does

- Answers common questions about PM-JAY (eligibility, coverage, hospitals, Ayushman Card, etc.)
- Uses **Retrieval-Augmented Generation (RAG)** — the LLM can only answer from the official scheme text, so it won't hallucinate
- Sends an introductory video on first contact
- Routes off-topic questions with a polite redirect
- Falls back to the official helpline (**14555**) when the answer isn't in the knowledge base

## Tech Stack

| Layer | Technology |
|---|---|
| API server | FastAPI (Python 3.11) |
| LLM & embeddings | Mistral AI (open-mistral-nemo) |
| RAG | Cosine similarity over yushman_bharat.txt |
| Messaging | Meta WhatsApp Business Cloud API |
| Tunnel (dev) | ngrok |

## Project Page

The index.html file at the root serves as the public project page (hosted via GitHub Pages). It includes:
- What the bot does and how it works
- Key PM-JAY scheme facts
- Privacy policy

## Running locally

`ash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Copy and fill in your keys
cp .env.example .env   # edit MISTRAL_API_KEY, WHATSAPP_TOKEN, etc.

# 3. Terminal test (no WhatsApp needed)
python bot.py --cli

# 4. Full WhatsApp webhook (expose port 8000 with ngrok)
uvicorn bot:app --port 8000
`

## Privacy

Only the text messages users send and their WhatsApp phone numbers (needed to reply) are processed. Message text is forwarded to the Mistral AI API. No data is sold or shared with third parties. See index.html for the full privacy policy.

## Official resources

- Website: [pmjay.gov.in](https://pmjay.gov.in)
- Helpline: **14555** / 1800-111-565
