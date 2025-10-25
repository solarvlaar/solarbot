import os
import sys
import re
import time
import random
import threading
import requests
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# ------------------------------------------------------------
# 🚂 Initialize Flask
# ------------------------------------------------------------
app = Flask(__name__)
print("[BOOT] Flask app initialized, waiting for requests...")
print("[BOOT] Python version:", sys.version)

# ------------------------------------------------------------
# 🧠 Model config (uses your own fine-tuned model)
# ------------------------------------------------------------
MODEL_PATH = os.getenv("MODEL_PATH", "microsoft/DialoGPT-medium")
tokenizer = None
model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def bevat_voornaam(tekst):
    return bool(re.search(r"(?:^|[\s,;])([A-Z][a-z]{2,})", tekst))

def load_model():
    """Lazy load van model"""
    global tokenizer, model
    if model is None or tokenizer is None:
        print(f"[INFO] 📦 Loading model: {MODEL_PATH}")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to(device)
        print("[INFO] ✅ Model loaded and ready.")

def generate_response(prompt):
    """Genereer korte, natuurlijke antwoorden zonder namen."""
    if model is None or tokenizer is None:
        load_model()

    input_text = f"<|prompter|>\n{prompt}\n<|responder|>\n"
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

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
            pad_token_id=tokenizer.eos_token_id,
            truncation=True
        )
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        response = generated_text.split("<|responder|>\n")[-1].strip()

        if not bevat_voornaam(response) and response:
            return response[:500]
    return "..."

# ------------------------------------------------------------
# 💬 Telegram webhook
# ------------------------------------------------------------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
BOT_URL = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"

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
            f"{BOT_URL}/sendMessage",
            json={"chat_id": chat_id, "text": response},
            timeout=10
        )
    except Exception as e:
        print(f"[ERROR] Telegram-send failed: {e}")

    return jsonify({"status": "ok"}), 200

# ------------------------------------------------------------
# 🌍 Health check
# ------------------------------------------------------------
@app.route("/", methods=["GET"])
def home():
    print("[HEALTHCHECK] Received health check ping ✅")
    return jsonify({"status": "ok"}), 200

# ------------------------------------------------------------
# 🕒 Keepalive & Auto-Webhook Fix
# ------------------------------------------------------------
def keepalive():
    port = os.environ.get("PORT", "5000")
    url = f"http://127.0.0.1:{port}/"
    while True:
        try:
            requests.get(url, timeout=5)
            print("[KEEPALIVE] Self-ping sent ✅")
        except:
            pass
        time.sleep(25)

def ensure_webhook():
    """Controleert of de webhook actief is en herstelt indien nodig."""
    try:
        info = requests.get(f"{BOT_URL}/getWebhookInfo", timeout=10).json()
        if not info.get("result", {}).get("url"):
            webhook_url = "https://solarbot.up.railway.app/telegram"
            set_hook = requests.get(f"{BOT_URL}/setWebhook?url={webhook_url}").json()
            if set_hook.get("ok"):
                print(f"[TELEGRAM] ✅ Webhook hersteld: {webhook_url}")
            else:
                print("[TELEGRAM] ⚠️ Webhook kon niet gezet worden.")
        else:
            print("[TELEGRAM] ✅ Webhook is nog actief.")
    except Exception as e:
        print(f"[TELEGRAM] ❌ Webhook check failed: {e}")

threading.Thread(target=keepalive, daemon=True).start()
threading.Thread(target=ensure_webhook, daemon=True).start()

# ------------------------------------------------------------
# 🚀 Entry point
# ------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
