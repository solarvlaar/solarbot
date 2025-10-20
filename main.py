import os
import sys
import time
import requests
from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from twilio.twiml.messaging_response import MessagingResponse

# ------------------------------------------------------------
# 🚂 Initialize Flask before loading any heavy modules
# ------------------------------------------------------------
app = Flask(__name__)
print("[BOOT] Flask app initialized, waiting for requests...")
print("[BOOT] Python version:", sys.version)
time.sleep(3)  # ⏳ give Railway healthcheck a moment

# ------------------------------------------------------------
# 🧠 MODEL CONFIG (lazy load)
# ------------------------------------------------------------
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


# ------------------------------------------------------------
# 💬 WhatsApp route
# ------------------------------------------------------------
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


# ------------------------------------------------------------
# 💬 Telegram route (optional)
# ------------------------------------------------------------
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")


@app.route("/telegram", methods=["POST"])
def telegram_webhook():
    if generator is None:
        load_model()

    data = request.json
    message = data.get("message", {}).get("text", "")
    chat_id = data.get("message", {}).get("chat", {}).get("id", "")

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

    print(f"[Telegram] Gebruiker zei: {message}")
    print(f"[Telegram] Bot antwoordt: {response}")
    telegram_url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": chat_id, "text": response}
    requests.post(telegram_url, json=payload)
    return "OK"


# ------------------------------------------------------------
# 🌍 Health check — Railway checks this endpoint
# ------------------------------------------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "ok", "message": "🚂 SolarBot is alive!"}), 200


# ------------------------------------------------------------
# 🚀 Local entry point (for manual testing)
# ------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"[INFO] Starting Flask server locally on port {port} ...")
    app.run(host="0.0.0.0", port=port)
