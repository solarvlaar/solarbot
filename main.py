import os
import re
import random
import time
import requests
import torch

from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
from twilio.twiml.messaging_response import MessagingResponse


app = Flask(__name__)

MODEL_PATH = os.getenv("MODEL_PATH", "solarvlaar/solarbot")
HF_TOKEN = os.getenv("HF_TOKEN")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
RAILWAY_URL = os.getenv(
    "RAILWAY_URL",
    "https://solarbot.up.railway.app"
)

print("Loading model:", MODEL_PATH)

try:
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        token=HF_TOKEN
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        token=HF_TOKEN
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model.to(device)
    model.eval()

    READY = True

    print("Model loaded.")
    print("Device:", device)

except Exception as e:
    READY = False
    tokenizer = None
    model = None

    print("Model loading failed:", repr(e))


# Common capitalized words that should not be treated as names.
EXCLUDED_WORDS = {
    "Ik",
    "Je",
    "Jij",
    "Jullie",
    "We",
    "Wij",
    "Hij",
    "Zij",
    "Ze",
    "Het",
    "Maar",
    "Dus",
    "En",
    "Dan",
    "Nee",
    "Ja",
    "Oke",
    "Oké",
    "Lol",
    "Hey",
    "Hoi",
    "Hé",
    "Hee",
    "Goeiemiddag",
    "Goeiemorgen",
    "Goedenavond",
    "Lief",
    "Lieve",
    "Nou",
    "Hm",
    "Hmm",
    "Ach",
    "Oh",
    "Ooh",
    "Aah",
    "Ah",
    "Ok"
}


def bevat_voornaam(text):
    for match in re.finditer(
        r"\b([A-Z][a-zäöüïëéèáíóúû]{3,12})\b",
        text
    ):
        word = match.group(1)

        if word not in EXCLUDED_WORDS:
            return True

    return False


def cleanup_response(text):
    text = text.strip()
    text = text.split("\n\n")[0].strip()

    for ending in [".", "!", "?", "…", "🤍", "❤️", "💖", "💘"]:
        parts = text.split(ending)

        if len(parts) > 1:
            first = parts[0].strip()

            if len(first) >= 3:
                return (first + ending).strip()

    last_space = text.rfind(" ")

    if last_space != -1:
        text = text[:last_space].strip()

    return text


def generate_response(prompt):
    if not READY:
        return "❤️"

    input_text = (
        f"<|prompter|>\n"
        f"{prompt}\n"
        f"<|responder|>\n"
    )

    input_ids = tokenizer.encode(
        input_text,
        return_tensors="pt"
    ).to(device)

    style = random.choices(
        ["short", "medium", "long"],
        weights=[0.90, 0.09, 0.01]
    )[0]

    if style == "short":
        max_new_tokens = random.randint(12, 22)
        temperature = 0.55

    elif style == "medium":
        max_new_tokens = random.randint(22, 35)
        temperature = 0.65

    else:
        max_new_tokens = random.randint(35, 50)
        temperature = 0.75

    eos_token_id = (
        tokenizer.eos_token_id
        if tokenizer.eos_token_id is not None
        else tokenizer.pad_token_id
    )

    for _ in range(2):
        try:
            with torch.no_grad():
                output_ids = model.generate(
                    input_ids,
                    do_sample=True,
                    temperature=temperature,
                    top_k=50,
                    top_p=0.85,
                    repetition_penalty=1.05,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=eos_token_id
                )

            generated = tokenizer.decode(
                output_ids[0],
                skip_special_tokens=True
            )

            if "<|responder|>" in generated:
                response = generated.split(
                    "<|responder|>\n"
                )[-1].strip()
            else:
                response = generated.replace(
                    input_text,
                    ""
                ).strip()

            response = cleanup_response(response)

            if len(response) < 2:
                continue

            if bevat_voornaam(response):
                continue

            return response[:500]

        except Exception as e:
            print("Generation failed:", repr(e))

    return "❤️"


@app.route("/telegram", methods=["POST"])
def telegram_webhook():
    data = request.get_json(force=True)

    message_data = data.get("message", {})
    message = message_data.get("text", "")
    chat_id = message_data.get("chat", {}).get("id")

    if not message or not chat_id:
        return jsonify({"status": "ignored"}), 200

    print("Telegram:", message)

    try:
        response = generate_response(message)

        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={
                "chat_id": chat_id,
                "text": response
            },
            timeout=30
        )

        print("Telegram response:", response)

    except Exception as e:
        print("Telegram error:", repr(e))

    return jsonify({"status": "ok"}), 200


@app.route("/whatsapp", methods=["POST"])
def whatsapp_webhook():
    incoming_message = request.values.get("Body", "")
    sender = request.values.get("From", "")

    print(
        "WhatsApp:",
        sender,
        incoming_message
    )

    if not incoming_message:
        return str(MessagingResponse())

    try:
        response = generate_response(incoming_message)

        print(
            "WhatsApp response:",
            response
        )

    except Exception as e:
        print("WhatsApp error:", repr(e))
        response = "❤️"

    twilio_response = MessagingResponse()
    twilio_response.message(response)

    return str(twilio_response)


@app.route("/", methods=["GET"])
def health_check():
    if not READY:
        return "Model unavailable", 503

    return "Solarbot is running", 200


def setup_telegram_webhook():
    if not TELEGRAM_TOKEN:
        print("Telegram token not configured.")
        return

    webhook_url = f"{RAILWAY_URL}/telegram"

    try:
        response = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/setWebhook",
            data={"url": webhook_url},
            timeout=30
        )

        print(
            "Telegram webhook:",
            response.text
        )

    except Exception as e:
        print(
            "Webhook setup failed:",
            repr(e)
        )

if __name__ == "__main__":
    time.sleep(3)

    setup_telegram_webhook()

    port = int(
        os.environ.get("PORT", 5000)
    )

    print("Starting server on port", port)

    app.run(
        host="0.0.0.0",
        port=port
    )
