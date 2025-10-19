import os
import random
import requests
from flask import Flask, request, Response
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from twilio.twiml.messaging_response import MessagingResponse

app = Flask(__name__)

# ----------------------------------------------------
# 🤖 Model laden met veilige fallback
# ----------------------------------------------------
MODEL_PATH = os.getenv("MODEL_PATH", "microsoft/DialoGPT-medium")

try:
    # eerst proberen lokaal pad of HF repo te laden
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=-1)
    print(f"[INFO] Model geladen: {MODEL_PATH}")
except Exception as e:
    # fallback naar DialoGPT als het lokale pad faalt
    print(f"[WARN] Kon model niet laden ({e}), gebruik fallback DialoGPT.")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=-1)

# ----------------------------------------------------
# ⚙️ Config
# ----------------------------------------------------
TWILIO_NUMBER = "whatsapp:+15558665761"
ACCOUNT_SID = "AC431b5be05867e4dc6b7298d8b886a07b"
AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN")  # in Railway secrets
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")

# ----------------------------------------------------
# 📱 WhatsApp route
# ----------------------------------------------------
@app.route("/whatsapp", methods=["POST"])
def whatsapp_reply():
    incoming_msg = request.values.get("Body", "").strip()
    from_number = request.values.get("From", "")

    print(f"[WhatsApp] Van {from_number}: {incoming_msg}")

    if not incoming_msg:
        return "OK"

    # korte, natuurlijke stijl
    response = generator(
        incoming_msg,
        max_length=random.randint(30, 60),
        do_sample=True,
        top_k=50,
        top_p=0.9,
        temperature=0.8,
        num_return_sequences=1
    )[0]["generated_text"]

    reply = response.replace(incoming_msg, "").strip()
    print(f"[WhatsApp] Antwoord: {reply}")

    twilio_response = MessagingResponse()
    twilio_response.message(reply)
    return str(twilio_response)

# ----------------------------------------------------
# 💬 Telegram route
# ----------------------------------------------------
@app.route("/telegram", methods=["POST"])
def telegram_webhook():
    data = request.json
    message = data.get("message", {}).get("text", "")
    chat_id = data.get("message", {}).get("chat", {}).get("id", "")

    print(f"[Telegram] Gebruiker zei: {message}")

    if not message:
        return "OK"

    response = generator(
        message,
        max_length=60,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.9,
        num_return_sequences=1
    )[0]["generated_text"]

    reply = response.replace(message, "").strip()
    print(f"[Telegram] Bot antwoordt: {reply}")

    if TELEGRAM_TOKEN:
        telegram_url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": chat_id, "text": reply}
        requests.post(telegram_url, json=payload)

    return "OK"

# ----------------------------------------------------
# 🌍 Home route
# ----------------------------------------------------
@app.route("/", methods=["GET"])
def home():
    return "🚂 SolarBot is live and waiting at the station."

# ----------------------------------------------------
# 🚀 Start server
# ----------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
