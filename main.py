import os
import random
import re
import threading
import requests
import torch

from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
from twilio.twiml.messaging_response import MessagingResponse


app = Flask(__name__)

MODEL_PATH = os.getenv("MODEL_PATH", "solarvlaar/solarbot")

HF_TOKEN = (
    os.getenv("HF_TOKEN")
    or os.getenv("HUGGINGFACE_HUB_TOKEN")
)

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

RAILWAY_URL = os.getenv(
    "RAILWAY_URL",
    "https://solarbot.up.railway.app"
)

BLOCKED_NAMES = {
    "erasmus",
    "douwe",
    "jelle",
    "remy",
    "wieg",
    "wieger",
    "jan",
    "thijs",
    "matthijs",
    "friso",
}

generation_lock = threading.Lock()


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

    print(
        "Model loading failed:",
        repr(e)
    )


def contains_blocked_name(text):

    words = re.findall(
        r"\b[\wÀ-ÿ'-]+\b",
        text.lower()
    )

    return any(
        word in BLOCKED_NAMES
        for word in words
    )


def limit_repeated_lines(text, max_repetitions=3):

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    result = []
    last_line = None
    repetition_count = 0

    for line in lines:

        if line == last_line:
            repetition_count += 1
        else:
            last_line = line
            repetition_count = 1

        if repetition_count <= max_repetitions:
            result.append(line)
        else:
            break

    return "\n".join(result)


def remove_truncated_last_line(text, was_truncated):

    if not was_truncated:
        return text

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    if len(lines) <= 1:
        return text

    lines.pop()

    return "\n".join(lines)


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

    for attempt in range(3):

        style = random.choices(
            ["short", "normal", "long"],
            weights=[0.25, 0.70, 0.05]
        )[0]

        if style == "short":
            max_new_tokens = 25
            temperature = 0.60

        elif style == "normal":
            max_new_tokens = 60
            temperature = 0.65

        else:
            max_new_tokens = 80
            temperature = 0.70

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
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )

            generated_tokens = (
                output_ids.shape[1]
                - input_ids.shape[1]
            )

            was_truncated = (
                generated_tokens >= max_new_tokens
            )

            generated = tokenizer.decode(
                output_ids[0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
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

            response = limit_repeated_lines(
                response,
                max_repetitions=3
            )

            response = remove_truncated_last_line(
                response,
                was_truncated
            )

            if not response:
                continue

            if contains_blocked_name(response):

                print(
                    f"[Model] Blocked name detected "
                    f"(attempt {attempt + 1}):",
                    response
                )

                continue

            return response[:500]

        except Exception as e:

            print(
                "[Model] Generation failed:",
                repr(e)
            )

            break

    return "❤️"


def process_telegram_message(message, chat_id, update_id):

    print(
        f"[Telegram] Processing update {update_id}:",
        message
    )

    with generation_lock:

        try:

            response = generate_response(
                message
            )

            print(
                f"[Telegram] Generated update {update_id}:",
                response
            )

            result = requests.post(
                f"https://api.telegram.org/"
                f"bot{TELEGRAM_TOKEN}/sendMessage",
                json={
                    "chat_id": chat_id,
                    "text": response
                },
                timeout=30
            )

            print(
                f"[Telegram] Sent update {update_id}:",
                result.status_code
            )

            if not result.ok:

                print(
                    f"[Telegram] Send error {update_id}:",
                    result.text
                )

        except Exception as e:

            print(
                f"[Telegram] Processing error {update_id}:",
                repr(e)
            )


@app.route("/telegram", methods=["POST"])
def telegram_webhook():

    data = request.get_json(
        force=True
    )

    update_id = data.get(
        "update_id"
    )

    message_data = data.get(
        "message",
        {}
    )

    message = message_data.get(
        "text",
        ""
    )

    chat_id = message_data.get(
        "chat",
        {}
    ).get(
        "id"
    )

    print(
        f"[Telegram] Received update {update_id}"
    )

    if not message or not chat_id:

        print(
            f"[Telegram] Ignored update {update_id}"
        )

        return jsonify({
            "status": "ignored"
        }), 200

    print(
        f"[Telegram] Accepted update {update_id}:",
        message
    )

    threading.Thread(
        target=process_telegram_message,
        args=(
            message,
            chat_id,
            update_id
        ),
        daemon=True
    ).start()

    return jsonify({
        "status": "accepted"
    }), 200


@app.route("/whatsapp", methods=["POST"])
def whatsapp_webhook():

    incoming_message = request.values.get(
        "Body",
        ""
    )

    sender = request.values.get(
        "From",
        ""
    )

    print(
        "[WhatsApp]:",
        sender,
        incoming_message
    )

    if not incoming_message:

        return str(
            MessagingResponse()
        )

    try:

        response = generate_response(
            incoming_message
        )

        print(
            "[WhatsApp] Response:",
            response
        )

    except Exception as e:

        print(
            "[WhatsApp] Error:",
            repr(e)
        )

        response = "❤️"

    twilio_response = MessagingResponse()

    twilio_response.message(
        response
    )

    return str(
        twilio_response
    )


@app.route("/", methods=["GET"])
def health_check():

    if not READY:

        return (
            "Model unavailable",
            503
        )

    return (
        "Solarbot is running",
        200
    )


def setup_telegram_webhook():

    if not TELEGRAM_TOKEN:

        print(
            "[Telegram] Token not configured."
        )

        return

    webhook_url = (
        f"{RAILWAY_URL}/telegram"
    )

    try:

        response = requests.post(
            f"https://api.telegram.org/"
            f"bot{TELEGRAM_TOKEN}/setWebhook",
            data={
                "url": webhook_url
            },
            timeout=30
        )

        print(
            "[Telegram] Webhook:",
            response.text
        )

    except Exception as e:

        print(
            "[Telegram] Webhook setup failed:",
            repr(e)
        )


if TELEGRAM_TOKEN:

    setup_telegram_webhook()


if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    print(
        "Starting server on port",
        port
    )

    app.run(
        host="0.0.0.0",
        port=port
    )
