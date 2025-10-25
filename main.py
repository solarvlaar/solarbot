import os
import sys
import time
import random
import re
import threading
import requests
from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# ------------------------------------------------------------
# 🚂 Initialize Flask
# ------------------------------------------------------------
app = Flask(__name__)
print("[BOOT] Flask app initialized, waiting for requests...")
print("[BOOT] Python version:", sys.version)

# ------------------------------------------------------------
# ⚙️ MODEL CONFIG
# ------------------------------------------------------------
MODEL_PATH = os.getenv("MODEL_PATH", "microsoft/DialoGPT-medium")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = None
model = None

def load_model():
    """Laadt het model slechts één keer (lazy load)."""
    global tokenizer, model
    if model is None or tokenizer is None:
        try:
            print(f"[INFO] 📦 Loading model: {MODEL_PATH} ...")
            tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
            model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to(device)
            print("[INFO] ✅ Model loaded and ready.")
        except Exception as e:
            print(f"[WARN] Kon model niet laden ({e}), gebruik fallback small model.")
            tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
            model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small").to(device)

# ------------------------------------------------------------
# 🧩 Helperfuncties
# ------------------------------------------------------------
def bevat_voornaam(tekst):
    """Detecteer voornamen of hoofdletterwoorden."""
    return bool(re.search(r"(?:^|[\s,;])([A-Z][a-z]{2,})", tekst))

def generate_response(prompt):
    """Genereer korte, natuurlijke antwoorden zonder namen."""
    if model is None or tokenizer is None:
        load_model()

    input_text = f"<|prompter|>\n{prompt}\n<|responder|>\n"
    input_ids = tokenizer.encode(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(device)

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

    for _ in range(5):
        output_ids = model.generate(
            input_ids,
            max_length=max_len,
            do_sample=True,
            temperature=temp,
            top_k=50,
            top_p=0.9,
            repetition_penalty=1.3,
            pad_token_id=tokenizer.eos_token_id
        )
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        response = generated_text.split("<|responder|>\n")[-1].strip()

        if not bevat_voornaam(response) and response:
            return response[:500]
    return "..."

# ------------------------------------------------------------
# 💬 TELEGRAM WEBHOOK
# ------------------------------------------------------------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
RAILWAY_URL = os.getenv("RAILWAY_PUBLIC_URL", "https://solarbot.up.railway.app")

@app.route("/telegram", methods=["POST"])
def telegram_webhook():
    load_model()
    data = request.get_json(force=True)
    message = data.get("message", {}).get("text")
    chat_id = data.get("message", {}).get("chat", {}).get("id")

    if not message or not chat_id:
        return jsonify({"status": "ignored"}), 200

    print(f"[Telegram] Gebruiker zei: {message}")
    response = generate_response(message)
    print(f"[Telegram] Bot antwoordt: {response}")

    try:
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
    print("[HEALTHCHECK] Received health check ping ✅")
    return "ok", 200

# ------------------------------------------------------------
# 🔁 Periodieke Telegram-webhook check
# ------------------------------------------------------------
def ensure_webhook():
    """Controleert of de webhook actief is en herstelt hem indien nodig."""
    try:
        info = requests.get(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getWebhookInfo").json()
        url = info.get("result", {}).get("url", "")
        if RAILWAY_URL not in url:
            print("[TELEGRAM] ⚠️ Webhook incorrect, opnieuw instellen...")
            resp = requests.get(
                f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/setWebhook",
                params={"url": f"{RAILWAY_URL}/telegram"}
            )
            print(f"[TELEGRAM] 🔁 Webhook reset result: {resp.text}")
        else:
            print("[TELEGRAM] ✅ Webhook is nog actief.")
    except Exception as e:
        print(f"[TELEGRAM] ❌ Fout bij webhook-check: {e}")

def webhook_loop():
    """Controleert elke 10 minuten of de webhook nog actief is."""
    while True:
        ensure_webhook()
        time.sleep(600)

threading.Thread(target=webhook_loop, daemon=True).start()

# ------------------------------------------------------------
# 🕒 Keepalive tegen Railway timeouts
# ------------------------------------------------------------
def keepalive_forever():
    """Ping zichzelf om wakker te blijven."""
    while True:
        try:
            requests.get(f"http://127.0.0.1:{os.environ.get('PORT', '5000')}/", timeout=5)
            print("[KEEPALIVE] Self-ping sent ✅")
        except Exception:
            pass
        time.sleep(30)

threading.Thread(target=keepalive_forever, daemon=True).start()

# ------------------------------------------------------------
# 🚀 Start
# ------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"[INFO] Starting Flask server locally on port {port} ...")
    ensure_webhook()
    app.run(host="0.0.0.0", port=port)
