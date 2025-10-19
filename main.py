import os
from flask import Flask, request
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from twilio.twiml.messaging_response import MessagingResponse
import requests

app = Flask(__name__)

# 🤖 Model laden
MODEL_PATH = os.getenv("MODEL_PATH", "microsoft/DialoGPT-medium")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)
generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=-1)

# 📱 Twilio instellingen
TWILIO_NUMBER = "whatsapp:+15558665761"
RECIPIENT_NUMBERS = [
    "whatsapp:+31687437563",  # ontvanger 1
    "whatsapp:+31683050411",  # ontvanger 2
]
ACCOUNT_SID = "AC431b5be05867e4dc6b7298d8b886a07b"
AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN")  # Zet deze in Railway secrets

# 🌐 Telegram (optioneel)
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")

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

    print(f"[WhatsApp] Antwoord: {response}")
    twilio_response = MessagingResponse()
    twilio_response.message(response)
    return str(twilio_response)

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
    )[0]['generated_text']

    print(f"[Telegram] Bot antwoordt: {response}")
    telegram_url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": response,
    }
    requests.post(telegram_url, json=payload)
    return "OK"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)

from flask import Flask, request
app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return "🚂 SolarBot is live and waiting at the station."
