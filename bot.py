"""
Ayushman Bharat (PM-JAY) WhatsApp bot - text in / text out.

Flow
 1. User's first message -> Mistral checks: is this about Ayushman Bharat?
      yes -> bot confirms, sends the saved video, then asks "any doubts?"
      no  -> bot politely says it only handles Ayushman Bharat
 2. Every later message  -> RAG over ayushman_bharat.txt.
      The LLM may ONLY use the retrieved text. If the answer is not in the
      file, it says so and points to the official helpline (no guessing).

Run (WhatsApp):  uvicorn bot:app --port 8000      (then expose with ngrok)
Run (no WhatsApp, terminal test):  python bot.py --cli
"""
import argparse
import json
import math
import os
import re
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

import httpx
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import PlainTextResponse

load_dotenv()

BASE_DIR = Path(__file__).parent
KNOWLEDGE_FILE = BASE_DIR / "ayushman_bharat.txt"
VIDEO_FILE = BASE_DIR / "videos" / "ayushman_intro.mp4"


# --------------------------------------------------------------------------
# LLM layer (swap providers by subclassing LLMClient)
# --------------------------------------------------------------------------
class LLMClient(ABC):
    @abstractmethod
    def chat(self, messages: list[dict], json_mode: bool = False,
             temperature: float = 0.0) -> str: ...

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class MistralClient(LLMClient):
    def __init__(self, api_key: str, chat_model: str = "open-mistral-nemo",
                 embed_model: str = "mistral-embed"):
        self.chat_model = chat_model
        self.embed_model = embed_model
        self.http = httpx.Client(
            base_url="https://api.mistral.ai/v1",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=60,
        )

    def chat(self, messages, json_mode=False, temperature=0.0) -> str:
        body = {"model": self.chat_model, "messages": messages,
                "temperature": temperature}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        r = self.http.post("/chat/completions", json=body)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()

    def embed(self, texts) -> list[list[float]]:
        r = self.http.post("/embeddings",
                           json={"model": self.embed_model, "input": texts})
        r.raise_for_status()
        return [d["embedding"] for d in r.json()["data"]]


# --------------------------------------------------------------------------
# Retrieval layer (RAG over the txt file)
# --------------------------------------------------------------------------
class Retriever(ABC):
    @abstractmethod
    def search(self, query: str, k: int = 3) -> list[str]: ...


class EmbeddingRetriever(Retriever):
    """Each paragraph (blocks separated by a blank line) is one chunk."""

    def __init__(self, llm: LLMClient, path: Path):
        self.llm = llm
        self.chunks = self._load_chunks(path)
        if not self.chunks:
            raise ValueError(f"No content found in {path}")
        self.vectors = self._embed_all(self.chunks)
        print(f"[RAG] loaded {len(self.chunks)} chunks from {path.name}")

    @staticmethod
    def _load_chunks(path: Path) -> list[str]:
        text = Path(path).read_text(encoding="utf-8")
        chunks = []
        for block in re.split(r"\n\s*\n", text):
            lines = [l.strip() for l in block.splitlines()
                     if l.strip() and not l.lstrip().startswith("#")]
            if lines:
                chunks.append(" ".join(lines))
        return chunks

    def _embed_all(self, chunks: list[str], batch: int = 16):
        vectors = []
        for i in range(0, len(chunks), batch):
            vectors.extend(self.llm.embed(chunks[i:i + batch]))
        return vectors

    @staticmethod
    def _cosine(a, b) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        return dot / (na * nb) if na and nb else 0.0

    def search(self, query: str, k: int = 3) -> list[str]:
        qv = self.llm.embed([query])[0]
        scored = sorted(
            ((self._cosine(qv, v), c) for v, c in zip(self.vectors, self.chunks)),
            reverse=True,
        )
        return [c for _, c in scored[:k]]


# --------------------------------------------------------------------------
# AI helpers
# --------------------------------------------------------------------------
class IntentClassifier:
    PROMPT = (
        "You check whether a WhatsApp user's message is about the Indian "
        "government health scheme Ayushman Bharat (PM-JAY, Ayushman card, "
        "Rs 5 lakh health cover, eligibility, empanelled hospitals, etc.). "
        "The message may be in English, Hindi or Hinglish. "
        'Reply with JSON only: {"about_scheme": true} or {"about_scheme": false}'
    )

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def is_about_scheme(self, text: str) -> bool:
        raw = self.llm.chat(
            [{"role": "system", "content": self.PROMPT},
             {"role": "user", "content": text}],
            json_mode=True,
        )
        try:
            return bool(json.loads(raw).get("about_scheme"))
        except (json.JSONDecodeError, AttributeError):
            return False


class Answerer:
    NOT_FOUND = "I_DONT_KNOW"
    SYSTEM = (
        "You are a helpful assistant for the Ayushman Bharat PM-JAY scheme. "
        "Answer ONLY using the CONTEXT. Never use outside knowledge, never "
        "guess, and never add examples, categories, groups or numbers that "
        "are not written in the CONTEXT. Stay close to the CONTEXT wording. "
        f"If the CONTEXT does not fully answer the question, reply with exactly: {NOT_FOUND}. "
        "LANGUAGE RULE: reply in the language of the QUESTION, not the "
        "language of the CONTEXT. English question -> English answer. "
        "Hindi (Devanagari) question -> Hindi. Hinglish question -> simple Hinglish. "
        "Keep it under 120 words, WhatsApp style, with short bullet points "
        "and *bold* labels."
    )

    def __init__(self, llm: LLMClient, retriever: Retriever):
        self.llm = llm
        self.retriever = retriever

    def answer(self, question: str, previous_question: str = "") -> str | None:
        # include the previous question so follow-ups ("and for seniors?") work
        query = f"{previous_question} {question}".strip()
        context = "\n\n".join(self.retriever.search(query, k=3))
        reply = self.llm.chat(
            [{"role": "system", "content": self.SYSTEM},
             {"role": "user",
              "content": (f"CONTEXT:\n{context}\n\nQUESTION: {question}\n\n"
                          "Answer in the same language as the QUESTION above.")}],
            temperature=0.0,
        )
        return None if self.NOT_FOUND in reply else reply


# --------------------------------------------------------------------------
# Channels (WhatsApp or terminal)
# --------------------------------------------------------------------------
class Channel(ABC):
    @abstractmethod
    def send_text(self, to: str, body: str) -> None: ...

    @abstractmethod
    def send_video(self, to: str, path: Path, caption: str = "") -> None: ...

    @abstractmethod
    def send_cta_button(self, to: str, body: str,
                        btn_label: str, url: str) -> None: ...


class ConsoleChannel(Channel):
    def send_text(self, to, body):
        print(f"\nBot: {body}")

    def send_video(self, to, path, caption=""):
        print(f"\nBot: [VIDEO would be sent: {path.name}] {caption}")

    def send_cta_button(self, to, body, btn_label, url):
        print(f"\nBot: {body}")
        print(f"      [ {btn_label} → {url} ]")


class WhatsAppChannel(Channel):
    GRAPH = "https://graph.facebook.com/v25.0"

    def __init__(self, token: str, phone_number_id: str):
        self.phone_number_id = phone_number_id
        self.http = httpx.Client(
            headers={"Authorization": f"Bearer {token}"}, timeout=120)
        self._media_cache: dict[str, str] = {}

    def _post_message(self, payload: dict) -> None:
        payload["messaging_product"] = "whatsapp"
        r = self.http.post(f"{self.GRAPH}/{self.phone_number_id}/messages",
                           json=payload)
        if r.status_code >= 400:
            print(f"[WhatsApp ERROR] {r.status_code}: {r.text}")

    def _upload(self, path: Path) -> str:
        key = str(path)
        if key not in self._media_cache:  # upload once, reuse the media id
            with open(path, "rb") as f:
                r = self.http.post(
                    f"{self.GRAPH}/{self.phone_number_id}/media",
                    data={"messaging_product": "whatsapp", "type": "video/mp4"},
                    files={"file": (path.name, f, "video/mp4")},
                )
            r.raise_for_status()
            self._media_cache[key] = r.json()["id"]
        return self._media_cache[key]

    def send_text(self, to, body):
        self._post_message({"to": to, "type": "text", "text": {"body": body}})

    def send_video(self, to, path, caption=""):
        if not path.exists():
            self.send_text(to, f"(Intro video not found on server: {path.name})")
            return
        media_id = self._upload(path)
        self._post_message({"to": to, "type": "video",
                            "video": {"id": media_id, "caption": caption}})

    def send_cta_button(self, to, body, btn_label, url):
        """Send an interactive message with a CTA URL button below the text."""
        self._post_message({
            "to": to,
            "type": "interactive",
            "interactive": {
                "type": "cta_url",
                "body": {"text": body},
                "action": {
                    "name": "cta_url",
                    "parameters": {
                        "display_text": btn_label,
                        "url": url,
                    },
                },
            },
        })


# --------------------------------------------------------------------------
# Bot logic
# --------------------------------------------------------------------------
@dataclass
class Session:
    video_sent: bool = False
    last_question: str = ""


class AyushmanBot:
    GREETING = ("Hi! 👋 I can help with *Ayushman Bharat PM-JAY*. "
                "Ask me about it, for example: \"What is Ayushman Bharat?\"")
    CONFIRM = ("✅ It looks like you're asking about *Ayushman Bharat PM-JAY*. "
               "Here is a short video that explains the scheme 👇")
    ASK_DOUBTS = ("Do you have any doubts? Ask me anything - eligibility, "
                  "coverage, how to get the card, hospitals, etc.")
    FALLBACK = ("Sorry, I don't have this detail. Please check pmjay.gov.in "
                "or call the official helpline (14555).")

    def __init__(self, channel: Channel, intent: IntentClassifier,
                 answerer: Answerer, video_path: Path = VIDEO_FILE):
        self.channel = channel
        self.intent = intent
        self.answerer = answerer
        self.video_path = video_path
        self.sessions: dict[str, Session] = {}

    def handle(self, user_id: str, text: str) -> None:
        s = self.sessions.setdefault(user_id, Session())

        if not s.video_sent:
            if not self.intent.is_about_scheme(text):
                self.channel.send_text(user_id, self.GREETING)
                return
            self.channel.send_text(user_id, self.CONFIRM)
            self.channel.send_video(user_id, self.video_path,
                                    "Ayushman Bharat PM-JAY - quick intro")
            self.channel.send_text(user_id, self.ASK_DOUBTS)
            s.video_sent = True
            return

        reply = self.answerer.answer(text, s.last_question)
        msg = reply or self.FALLBACK
        self.channel.send_cta_button(
            user_id, msg,
            btn_label="Visit pmjay.gov.in",
            url="https://pmjay.gov.in",
        )
        s.last_question = text


def build_bot(channel: Channel) -> AyushmanBot:
    llm = MistralClient(os.environ["MISTRAL_API_KEY"])
    retriever = EmbeddingRetriever(llm, KNOWLEDGE_FILE)
    return AyushmanBot(channel, IntentClassifier(llm), Answerer(llm, retriever))


# --------------------------------------------------------------------------
# FastAPI webhook (WhatsApp Cloud API)
# --------------------------------------------------------------------------
_bot: AyushmanBot | None = None
_seen_ids: set[str] = set()  # Meta retries webhooks; avoid double replies


def get_bot() -> AyushmanBot:
    global _bot
    if _bot is None:
        channel = WhatsAppChannel(os.environ["WHATSAPP_TOKEN"],
                                  os.environ["PHONE_NUMBER_ID"])
        _bot = build_bot(channel)
    return _bot


@asynccontextmanager
async def lifespan(_app: FastAPI):
    get_bot()  # build embeddings once at startup
    yield


app = FastAPI(lifespan=lifespan)


def process_message(msg: dict) -> None:
    bot = get_bot()
    user = msg["from"]
    try:
        if msg.get("type") == "text":
            bot.handle(user, msg["text"]["body"])
        else:
            bot.channel.send_text(user, "Please send your question as text 🙂")
    except Exception as e:  # keep the demo alive on any error
        print(f"[ERROR] {e!r}")
        bot.channel.send_text(user, "Sorry, something went wrong. Please try again.")


@app.get("/webhook")
async def verify(request: Request):
    q = request.query_params
    if (q.get("hub.mode") == "subscribe"
            and q.get("hub.verify_token") == os.environ.get("VERIFY_TOKEN")):
        return PlainTextResponse(q.get("hub.challenge", ""))
    return PlainTextResponse("Forbidden", status_code=403)


@app.post("/webhook")
async def incoming(request: Request, background: BackgroundTasks):
    data = await request.json()
    for entry in data.get("entry", []):
        for change in entry.get("changes", []):
            for msg in change.get("value", {}).get("messages", []):
                if msg["id"] in _seen_ids:
                    continue
                _seen_ids.add(msg["id"])
                background.add_task(process_message, msg)
    return {"ok": True}  # reply 200 fast; real work happens in background


# --------------------------------------------------------------------------
# Terminal test mode (no WhatsApp needed)
# --------------------------------------------------------------------------
def run_cli() -> None:
    bot = build_bot(ConsoleChannel())
    print("CLI test mode. Type your message (Ctrl+C to quit).")
    while True:
        text = input("\nYou: ").strip()
        if text:
            bot.handle("cli-user", text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli", action="store_true", help="test in terminal")
    if parser.parse_args().cli:
        run_cli()
    else:
        print("Run the WhatsApp server with: uvicorn bot:app --port 8000")
