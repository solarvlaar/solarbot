
import os
import requests
import zipfile
from flask import Flask, request
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

app = Flask(__name__)

MODEL_DIR = "model"

# ⬇️ Download model van Google Drive bij eerste run
def download_model():
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR, exist_ok=True)
        zip_url = "https://drive.google.com/uc?id=1NyK4jGVdnMJh7MbAYgFVkgWe9M3i7Cg3"  # een ZIP-bestand moet daar staan
        zip_path = "model.zip"
        print("📦 Downloading model...")
        r = requests.get(zip_url)
        with open(zip_path, "wb") as f:
            f.write(r.content)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(MODEL_DIR)
        os.remove(zip_path)
        print("✅ Model downloaded and extracted.")

download_model()

# 🔁 Model laden
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR)
generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=-1)

@app.route("/whatsapp", methods=["POST"])
def whatsapp_webhook():
    from_number = request.form.get("From", "")
    body = request.form.get("Body", "").strip()

    print(f"📩 Bericht ontvangen van {from_number}: {body}")

    if not body:
        return "No content", 200

    # 🔮 Genereer antwoord
    response = generator(
        body,
        max_length=100,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.3,
        num_return_sequences=1,
    )[0]["generated_text"].strip()

    # Alleen het nieuwe stuk tekst teruggeven
    response = response[len(body):].strip().split("\n")[0]

    # 📤 Stuur terug via Twilio API
    requests.post(
        "https://api.twilio.com/2010-04-01/Accounts/{}/Messages.json".format(os.environ["TWILIO_ACCOUNT_SID"]),
        auth=(os.environ["TWILIO_ACCOUNT_SID"], os.environ["TWILIO_AUTH_TOKEN"]),
        data={
            "From": f"whatsapp:{os.environ['TWILIO_WHATSAPP_NUMBER']}",
            "To": f"whatsapp:{os.environ['YOUR_PHONE_NUMBER']}",
            "Body": response
        }
    )

    print(f"🤖 Antwoord verzonden: {response}")
    return "OK", 200

@app.route("/", methods=["GET"])
def index():
    return "🤖 WhatsApp bot is running.", 200
