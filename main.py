import os
import sys
import requests
import time
from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from twilio.rest import Client
from twilio.twiml.messaging_response import MessagingResponse

# ------------------------------------------------------------
# 🚂 Initialize Flask before heavy modules
# ------------------------------------------------------------
app = Flask(__name__)
print("[BOOT] Flask app initialized, waiting for requests...")
print("[BOOT] Python version:", sys.version)

# ------------------------------------------------------------
# 🧠 MODEL CONFIG (lazy load)
# ------------------------------------------------------------
MODEL_PATH = os.getenv("MODEL_PATH", "microsoft/DialoGPT-medium")
tokenizer = None
model = None
generator = None

def load_model():
    """Laadt het model pas bij eerste gebruik (lazy load)."""
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
# 📱 Twilio config via environment variables
# ------------------------------------------------------------
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")  # e.g. whatsapp:+15558665761
RECIPIENT_NUMBERS = os.getenv("RECIPIENT_NUMBERS", "").split(",")

client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

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

    # Twilio webhook reply
    twilio_response = MessagingResponse()
    twilio_response.message(response)

    # Eventueel pushbericht terugsturen via Twilio REST API
    for recipient in RECIPIENT_NUMBERS:
        if recipient.strip():
            client.messages.create(
                from_=TWILIO_PHONE_NUMBER,
                to=recipient.strip(),
                body=response
            )

    return str(twilio_response)

# ------------------------------------------------------------
# 💬 Telegram route (async reply)
# ------------------------------------------------------------
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")

@app.route("/telegram", methods=["POST"])
def telegram_webhook():
    if generator is None:
        load_model()

    data = request.get_json(force=True)
    message = data.get("message", {}).get("text")
    chat_id = data.get("message", {}).get("chat", {}).get("id")

    if not message or not chat_id:
        return jsonify({"status": "ignored"}), 200

    print(f"[Telegram] Gebruiker zei: {message}")

    def send_reply():
        try:
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

            requests.post(
                f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                json={"chat_id": chat_id, "text": response}
            )
        except Exception as e:
            print(f"[ERROR] Telegram-fout: {e}")

    threading.Thread(target=send_reply, daemon=True).start()
    return jsonify({"status": "ok"}), 200

# ------------------------------------------------------------
# 🌍 Health check
# ------------------------------------------------------------
@app.route("/", methods=["GET"])
def home():
    print("[HEALTHCHECK] Received health check ping ✅")
    return "ok", 200
# ------------------------------------------------------------
# 🕒 Prevent Railway from killing the app too early
# ------------------------------------------------------------
import atexit
import threading

def keepalive():
    """Houd de app actief zodat Railway hem niet afsluit."""
    def ping_loop():
        while True:
            try:
                requests.get("http://0.0.0.0:" + os.environ.get("PORT", "5000"))
            except Exception:
                pass
            time.sleep(25)  # elke 25 seconden ping
    thread = threading.Thread(target=ping_loop, daemon=True)
    thread.start()

# Start keepalive zodra Flask geladen is
atexit.register(lambda: print("[KEEPALIVE] Flask shutting down gracefully."))
keepalive()
# ------------------------------------------------------------
# 🚀 Entry point (for local or gunicorn)
# ------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"[INFO] Starting Flask server locally on port {port} ...")
    app.run(host="0.0.0.0", port=port)
