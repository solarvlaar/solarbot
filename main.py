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
    """Genereer korte, natuurlijke antwoorden zonder namen of herhaling."""
    global model, tokenizer
    if model is None or tokenizer is None:
        load_model()

    input_text = f"<|prompter|>\n{prompt}\n<|responder|>\n"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    stijl = random.choices(["kort", "middel", "lang"], weights=[0.85, 0.12, 0.03])[0]
    if stijl == "kort":
        max_len = input_ids.shape[1] + random.randint(5, 20)
        temp = 0.6
    elif stijl == "middel":
        max_len = input_ids.shape[1] + random.randint(25, 40)
        temp = 0.75
    else:
        max_len = input_ids.shape[1] + random.randint(50, 80)
        temp = 0.9

    for _ in range(5):
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

        # filter leeg, echo, of namen
        if not response or response.lower() == prompt.lower() or bevat_voornaam(response):
            continue
        return response[:500]

    return random.choice(["hmm...", "geen idee", "misschien", "🤔"])


def send_reply():
    try:
        reply = generate_response(message)
        print(f"[Telegram] Bot antwoordt: {reply}")

        # afhandelen van tijdelijke SSL-fouten met retry
        for _ in range(3):
            try:
                r = requests.post(
                    f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                    json={"chat_id": chat_id, "text": reply},
                    timeout=5
                )
                if r.status_code == 200:
                    break
            except requests.exceptions.SSLError:
                print("[WARN] SSL error, retrying...")
                time.sleep(1)
    except Exception as e:
        print(f"[ERROR] Telegram-fout: {e}")

# ------------------------------------------------------------
# 💬 Telegram webhook
# ------------------------------------------------------------
# ------------------------------------------------------------
# 💬 Telegram config + auto-webhook herstel
# ------------------------------------------------------------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
WEBHOOK_URL = "https://solarbot.up.railway.app/telegram"

def ensure_webhook():
    """Controleert periodiek of de Telegram webhook nog goed staat."""
    if not TELEGRAM_TOKEN:
        print("[TELEGRAM] ⚠️ Geen token gevonden, webhook wordt niet gezet.")
        return
    while True:
        try:
            info_url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getWebhookInfo"
            set_url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/setWebhook?url={WEBHOOK_URL}"
            info = requests.get(info_url, timeout=10).json()

            if not info.get("ok"):
                print(f"[TELEGRAM] ❌ getWebhookInfo mislukt: {info}")
            else:
                current = info.get("result", {}).get("url", "")
                if current != WEBHOOK_URL:
                    r = requests.get(set_url, timeout=10).json()
                    print(f"[TELEGRAM] 🔄 Webhook opnieuw gezet: {r}")
                else:
                    print("[TELEGRAM] ✅ Webhook is nog actief.")
        except Exception as e:
            print(f"[TELEGRAM] ⚠️ Fout bij webhook-check: {e}")

        time.sleep(300)  # check elke 5 minuten

threading.Thread(target=ensure_webhook, daemon=True).start()

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

    try:
        response = generator(
            message,
            max_length=60,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.9,
            truncation=True,
            num_return_sequences=1
        )[0]['generated_text'].strip()

        if not response:
            response = "..."

        print(f"[Telegram] Bot antwoordt: {response}")

        send_url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(send_url, json={"chat_id": chat_id, "text": response})
    except Exception as e:
        print(f"[ERROR] Telegram-fout: {e}")

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
