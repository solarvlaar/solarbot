from flask import Flask, request
import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

app = Flask(__name__)

# 📍 Laden van model
model_path = "./model"  # Of geef je pad aan als je dat elders mount
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)
generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=-1)

# 🔐 Zet je API keys hier
TELEGRAM_TOKEN = "VUL_HIER_JE_TELEGRAM_BOT_TOKEN_IN"
TWILIO_NUMBER = "whatsapp:+15558665761"
RECIPIENT_NUMBER = "whatsapp:+316XXXXXXX"  # Alleen nodig als jij Twilio direct test

# 🎯 Genereren van antwoord
def generate_response(prompt):
    response = generator(
        prompt,
        max_new_tokens=64,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7,
        num_return_sequences=1,
    )[0]["generated_text"]
    
    return response.replace(prompt, "").strip()

# ✅ WhatsApp endpoint
@app.route("/whatsapp", methods=["POST"])
def whatsapp_webhook():
    from_number = request.form.get("From")
    body = request.form.get("Body")

    response = generate_response(body)

    payload = {
        "From": TWILIO_NUMBER,
        "To": from_number,
        "Body": response,
    }
    requests.post("https://api.twilio.com/2010-04-01/Accounts/YOUR_SID_HERE/Messages.json", data=payload, auth=("YOUR_SID_HERE", "YOUR_AUTH_TOKEN_HERE"))
    return "ok", 200

# ✅ Telegram endpoint
@app.route("/telegram", methods=["POST"])
def telegram_webhook():
    data = request.get_json()
    chat_id = data["message"]["chat"]["id"]
    user_message = data["message"]["text"]

    response = generate_response(user_message)

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": response,
    }
    requests.post(url, json=payload)

    return "ok", 200

# 👋 Root check
@app.route("/", methods=["GET"])
def home():
    return "🤖 Solar chatbot active!", 200

