import os
import sys
import re
import random
import time
import threading
import requests
from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
from twilio.rest import Client
from twilio.twiml.messaging_response import MessagingResponse

# ------------------------------------------------------------
# 🚂 Flask setup
# ------------------------------------------------------------
app = Flask(__name__)
print("[BOOT] Flask app initialized, waiting for requests...")
print("[BOOT] Python version:", sys.version)

# ------------------------------------------------------------
# 🧠 MODEL CONFIG
# ------------------------------------------------------------
MODEL_PATH = os.getenv("MODEL_PATH", "microsoft/DialoGPT-medium")
device = "cpu"
tokenizer = None
model = None

def load_model():
    global tokenizer, model
    if model is None or tokenizer is None:
        try:
            print(f"[INFO] 📦 Loading model: {MODEL_PATH} ...")
            tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
            model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
            print("[INFO] ✅ Model loaded successfully.")
        except Exception as e:
            print(f"[WARN] Kon model niet laden ({e}), gebruik fallback DialoGPT-small.")
            tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
            model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
            print("[INFO] ✅ Fallback model loaded.")

# ------------------------------------------------------------
# 🚫 Naamfilter + responslogica
# ------------------------------------------------------------
def bevat_voornaam(tekst):
    """Detecteer hoofdletter-namen zonder leestekens ervoor."""
    return bool(re.search(r"(?:^|[\s,;])([A-Z][a-z]{2,})", tekst))

def generate_response(prompt):
    """Genereer korte, natuurlijke antwoorden zonder namen."""
    if model is None or tokenizer is None:
        load_model()

    input_text = f"<|prompter|>\n{prompt}\n<|responder|>\n"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    stijl = random.choices(["kort", "middel", "lang"], weights=[0.85, 0.12, 0.03])[0]
    if stijl == "kort":
        max_len = input_ids.shape[1] + random.randint(5, 20)
        temp = 0.5
    elif stijl == "middel":
        max_len = input_ids.shape[1] + random.randint(25, 40)
        temp = 0.7
    else:
        max_len = input_ids.shape[1] + random.randint(50, 80)
        temp = 0.85

    for _ in range(5):  # probeer tot 5x zonder naam
        output_ids = model.generate(
            input_ids,
            max_length=max_len,
            do_sample=True,
            temperature=temp,
            top_k=50,
            top_p=0.9,
            repetition_penalty=1.3,
            pad_token_id=tokenizer.eos_token_id,
        )

        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        response = generated_text.split("<|responder|>\n")[-1].strip()

        if not bevat_voornaam(response):
            return response[:500]

    return "..."

# ------------------------------------------------------------
# 💬 Telegram webhook
# ------------------------------------------------------------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

@app.route("/telegram", methods=["POST"])
def telegram_webhook():
    data = request.get_json(force=True)
    message = data.get("message", {}).get("text")
    chat_id = data.get("message", {}).get("chat", {}).get("id")

    if not message or not chat_id:
        return jsonify({"status": "ignored"}), 200

    print(f"[Telegram] Gebruiker zei: {message}")

    def send_reply():
        try:
            reply = generate_response(message)
            print(f"[Telegram] Bot antwoordt: {reply}")
            requests.post(
                f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                json={"chat_id": chat_id, "text": reply},
            )
        except Exception as e:
            print(f"[ERROR] Telegram-fout: {e}")

    threading.Thread(target=send_reply, daemon=True).start()
    return jsonify({"status": "ok"}), 200

# ------------------------------------------------------------
# 💬 WhatsApp (Twilio)
# ------------------------------------------------------------
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")
RECIPIENT_NUMBERS = os.getenv("RECIPIENT_NUMBERS", "").split(",")

client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

@app.route("/whatsapp", methods=["POST"])
def whatsapp_reply():
    incoming_msg = request.values.get("Body", "")
    from_number = request.values.get("From", "")
    print(f"[WhatsApp] Van {from_number}: {incoming_msg}")
    reply = generate_response(incoming_msg)

    twilio_response = MessagingResponse()
    twilio_response.message(reply)

    for recipient in RECIPIENT_NUMBERS:
        if recipient.strip():
            client.messages.create(
                from_=TWILIO_PHONE_NUMBER,
                to=recipient.strip(),
                body=reply,
            )

    return str(twilio_response)

# ------------------------------------------------------------
# 🌍 Healthcheck
# ------------------------------------------------------------
@app.route("/", methods=["GET"])
def home():
    print("[HEALTHCHECK] Received health check ping ✅")
    return "ok", 200

# ------------------------------------------------------------
# 🔁 Keepalive (Railway persistent fix)
# ------------------------------------------------------------
import atexit
def keepalive_forever():
    while True:
        try:
            port = os.environ.get("PORT", "5000")
            requests.get(f"http://127.0.0.1:{port}/", timeout=5)
            print("[KEEPALIVE] Self-ping sent ✅")
        except Exception as e:
            print(f"[KEEPALIVE] Ping failed: {e}")
        time.sleep(20)

threading.Thread(target=keepalive_forever, daemon=True).start()
atexit.register(lambda: print("[KEEPALIVE] Flask shutting down gracefully."))

# ------------------------------------------------------------
# 🚀 Start server
# ------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
