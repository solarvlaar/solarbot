import os
import sys
import time
import threading
import requests
from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from twilio.twiml.messaging_response import MessagingResponse

# ------------------------------------------------------------
# 🚂 Initialize Flask before heavy modules
# ------------------------------------------------------------
app = Flask(__name__)

def startup_ping():
    """Ping de healthcheck zodat Railway weet dat de server leeft."""
    time.sleep(3)
    try:
        print("[BOOT] Sending self-ping to health endpoint...")
        requests.get("http://0.0.0.0:" + os.environ.get("PORT", "5000"))
    except Exception as e:
        print(f"[BOOT] Ping failed (but it's fine): {e}")

# Start de ping in een aparte thread
threading.Thread(target=startup_ping, daemon=True).start()

print("[BOOT] Initializing Flask app ...")
time.sleep(5)
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
    twilio_response = MessagingResponse()
    twilio_response.message(response)
    return str(twilio_response)


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


@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "ok", "message": "🚂 SolarBot is alive!"}), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"[INFO] Starting Flask server locally on port {port} ...")
    app.run(host="0.0.0.0", port=port)
