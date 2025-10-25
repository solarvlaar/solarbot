import os
import sys
import time
import threading
import requests
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from flask import Flask, request, jsonify
from twilio.rest import Client
from twilio.twiml.messaging_response import MessagingResponse

# ------------------------------------------------------------
# 🧠 MODEL LAAD VOOR FLASK
# ------------------------------------------------------------
MODEL_PATH = os.getenv("MODEL_PATH", "microsoft/DialoGPT-medium")
print("[BOOT] Starting model load *before* Flask...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=-1)
    READY = True
    print("[INFO] ✅ Model preloaded and ready.")
except Exception as e:
    print(f"[WARN] Kon model niet laden ({e}), gebruik fallback DialoGPT-small.")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=-1)
    READY = True
    print("[INFO] ✅ Fallback model loaded.")

# ------------------------------------------------------------
# 🚂 Flask setup
# ------------------------------------------------------------
app = Flask(__name__)
print("[BOOT] Flask app initialized.")
print("[BOOT] Python version:", sys.version)

# ------------------------------------------------------------
# 📱 Twilio config
# ------------------------------------------------------------
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")
RECIPIENT_NUMBERS = os.getenv("RECIPIENT_NUMBERS", "").split(",")

client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# ------------------------------------------------------------
# 💬 WhatsApp route
# ------------------------------------------------------------
@app.route("/whatsapp", methods=["POST"])
def whatsapp_reply():
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

    for recipient in RECIPIENT_NUMBERS:
        if recipient.strip():
            client.messages.create(
                from_=TWILIO_PHONE_NUMBER,
                to=recipient.strip(),
                body=response
            )
    return str(twilio_response)

# ------------------------------------------------------------
# 💬 Telegram route
# ------------------------------------------------------------
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
RAILWAY_URL = os.environ.get("RAILWAY_URL", "https://solarbot.up.railway.app")

@app.route("/telegram", methods=["POST"])
def telegram_webhook():
    data = request.get_json(force=True)
    message = data.get("message", {}).get("text")
    chat_id = data.get("message", {}).get("chat", {}).get("id")

    if not message or not chat_id:
        return jsonify({"status": "ignored"}), 200

    print(f"[Telegram] Gebruiker zei: {message}")

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

    return jsonify({"status": "ok"}), 200

# ------------------------------------------------------------
# 🌍 Health check
# ------------------------------------------------------------
@app.route("/", methods=["GET"])
def home():
    if not READY:
        print("[HEALTHCHECK] App not ready yet ❌")
        return "initializing...", 503
    print("[HEALTHCHECK] Received health check ping ✅")
    return "ok", 200

# ------------------------------------------------------------
# 🕒 Persistent Keepalive + webhook check
# ------------------------------------------------------------
def keepalive_forever():
    while True:
        try:
            port = os.environ.get("PORT", "5000")
            url = f"http://127.0.0.1:{port}/"
            requests.get(url, timeout=5)
            print("[KEEPALIVE] Self-ping sent ✅")
        except Exception as e:
            print(f"[KEEPALIVE] Ping failed: {e}")
        time.sleep(20)

def check_webhook_periodically():
    while True:
        try:
            info = requests.get(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getWebhookInfo").json()
            if not info.get("ok") or info["result"].get("url") != f"{RAILWAY_URL}/telegram":
                print("[TELEGRAM] ⚠️ Webhook was niet actief, opnieuw instellen...")
                resp = requests.post(
                    f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/setWebhook",
                    data={"url": f"{RAILWAY_URL}/telegram"}
                )
                print(f"[TELEGRAM] Webhook reset: {resp.text}")
            else:
                print("[TELEGRAM] ✅ Webhook is nog actief.")
        except Exception as e:
            print(f"[TELEGRAM] ❌ Webhook check failed: {e}")
        time.sleep(3600)

threading.Thread(target=keepalive_forever, daemon=True).start()
threading.Thread(target=check_webhook_periodically, daemon=True).start()

# ------------------------------------------------------------
# 🚀 Entry point
# ------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"[INFO] Starting Flask server on port {port} ...")
    app.run(host="0.0.0.0", port=port)
