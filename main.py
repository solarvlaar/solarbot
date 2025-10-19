import os
import requests
from flask import Flask, request
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from twilio.twiml.messaging_response import MessagingResponse

app = Flask(__name__)

# 🧠 MODEL CONFIG ----------------------------------------------------
MODEL_PATH = os.getenv("MODEL_PATH", "microsoft/DialoGPT-medium")
tokenizer = None
model = None
generator = None


def load_model():
    """Laadt het taalmodel pas bij eerste gebruik (lazy load)."""
    global tokenizer, model, generator
    if generator is None:
        try:
            print(f"[INFO] 📦 Loading model: {MODEL_PATH} ...")
            tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
            model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
            generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=-1)
            print("[INFO] ✅ Model loaded successfully.")
        except Exception as e:
            print(f"[WARN] Kon model niet laden ({e}), gebruik fallback DialoGPT-small.")
            tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
            model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
            generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=-1)
            print("[INFO] ✅ Fallback model loaded.")


# 💬 WHATSAPP ROUTE ---------------------------------------------------
@app.route("/whatsapp", methods=["POST"])
def whatsapp_reply():
    if generator is None:
        load_model()

    incoming_msg = request.values.get("Body", "")
    from_number = request.values.get("From", "")
    print(f"[WhatsApp] Van {from_number}: {incoming_msg}")

    response = generator(
        incoming_msg,
        max_length=60,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.9,
        num_return_sequences=1
    )[0]['generated_text']

    print(f"[WhatsApp] Antwoord: {response}")
    twilio_response = MessagingResponse()
    twilio_response.message(response)
    return str(twilio_response)


# 💬 TELEGRAM ROUTE ---------------------------------------------------
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")


@app.route("/telegram", methods=["POST"])
def telegram_webhook():
    if generator is None:
        load_model()

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
    )[0]['generated_text']

    print(f"[Telegram] Bot antwoordt: {response}")
    telegram_url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": chat_id, "text": response}
    requests.post(telegram_url, json=payload)
    return "OK"


# 🌍 HEALTHCHECK ------------------------------------------------------
@app.route("/", methods=["GET"])
def home():
    return "🚂 SolarBot is live and waiting at the station."


# 🚀 MAIN ENTRY -------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"[INFO] Starting Flask server on port {port} ...")
    app.run(host="0.0.0.0", port=port)
