import os
import random
import re
import threading
import time

import requests
from flask import Flask, request, jsonify

app = Flask(__name__)

RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
RUNPOD_ENDPOINT_ID = os.getenv(\n    "SOLARBOT_V2_ENDPOINT_ID",\n    "d0u37r2jjwo9h3"\n)

RUNPOD_CHAT_COMPLETIONS_URL = (
    f"https://api.runpod.ai/v2/"
    f"{RUNPOD_ENDPOINT_ID}/openai/v1/chat/completions"
)

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_WHATSAPP_NUMBER = os.getenv("TWILIO_WHATSAPP_NUMBER")

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
    "solar",
    "lucas",
    "luc",
    "coen",
    "daan",
    "daantje",
    "olivier",
    "steven",
}

generation_lock = threading.Lock()

print(
    "[RunPod] Endpoint configured:",
    bool(RUNPOD_ENDPOINT_ID)
)

print(
    "[RunPod] API key configured:",
    bool(RUNPOD_API_KEY)
)

print(
    "[RunPod] Completion URL:",
    RUNPOD_CHAT_COMPLETIONS_URL
)

print(
    "[Twilio] Account SID configured:",
    bool(TWILIO_ACCOUNT_SID)
)

print(
    "[Twilio] Auth token configured:",
    bool(TWILIO_AUTH_TOKEN)
)

print(
    "[Twilio] WhatsApp number configured:",
    bool(TWILIO_WHATSAPP_NUMBER)
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

    last_line = lines[-1]

    if last_line.endswith(
        (
            ".",
            "!",
            "?",
            "…",
            "❤️",
            "♥️"
        )
    ):
        return text

    lines.pop()

    return "\n".join(lines)


def generate_response(prompt):
    if not RUNPOD_API_KEY:
        print("[RunPod] Missing RUNPOD_API_KEY.")
        return "❤️"

    if not RUNPOD_ENDPOINT_ID:
        print("[RunPod] Missing RUNPOD_ENDPOINT_ID.")
        return "❤️"

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

    payload = {
        "model": "solarvlaar/solarbot",
        "messages": [\n            {\n                "role": "system",\n                "content": (\n                    "Je bent Solarbot. Antwoord in natuurlijk, informeel " \\n                    "Nederlands alsof je appt. Houd het kort en persoonlijk. " \\n                    "Denk niet hardop."\n                )\n            },\n            {\n                "role": "user",\n                "content": prompt\n            }\n        ],
        "max_tokens": max_new_tokens,
        "temperature": temperature,
        "top_k": 50,
        "top_p": 0.85,
        "repetition_penalty": 1.05,
        "chat_template_kwargs": {
            "enable_thinking": False
        }
    }

    headers = {
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
        "Content-Type": "application/json"
    }

    started_at = time.time()

    try:
        print(
            "[RunPod] Sending OpenAI completion request..."
        )

        response = requests.post(
            RUNPOD_CHAT_COMPLETIONS_URL,
            json=payload,
            headers=headers,
            timeout=300
        )

        elapsed = time.time() - started_at

        print(
            f"[RunPod] Request completed in "
            f"{elapsed:.2f}s"
        )

        print(
            "[RunPod] HTTP status:",
            response.status_code
        )

        if not response.ok:
            print(
                "[RunPod] Error response:",
                response.text
            )

            return "❤️"

        result = response.json()

        print(
            "[RunPod] Response received."
        )

        choices = result.get(
            "choices",
            []
        )

        if not choices:
            print(
                "[RunPod] No choices in response:",
                result
            )

            return "❤️"

        choice = choices[0]

        text = choice.get(
            "message",
            {}
        ).get(
            "content",
            ""
        ).strip()

        finish_reason = choice.get(
            "finish_reason"
        )

        text = limit_repeated_lines(
            text,
            max_repetitions=3
        )

        text = remove_truncated_last_line(
            text,
            finish_reason == "length"
        )

        if contains_blocked_name(text):
            print(
                "[RunPod] Blocked name detected:",
                text
            )

            return "❤️"

        if not text:
            print(
                "[RunPod] Empty response."
            )

            return "❤️"

        print(
            "[RunPod] Response:",
            text
        )

        return text[:500]

    except requests.exceptions.Timeout:
        print(
            "[RunPod] Request timed out."
        )

        return "❤️"

    except requests.exceptions.RequestException as e:
        print(
            "[RunPod] Request failed:",
            repr(e)
        )

        return "❤️"

    except Exception as e:
        print(
            "[RunPod] Unexpected error:",
            repr(e)
        )

        return "❤️"


def process_telegram_message(
    message,
    chat_id,
    update_id
):
    print(
        f"[Telegram] Processing update "
        f"{update_id}: {message}"
    )

    with generation_lock:
        started_at = time.time()

        try:
            response = generate_response(
                message
            )

            elapsed = time.time() - started_at

            print(
                f"[Generation] Finished update "
                f"{update_id} in {elapsed:.2f}s"
            )

            print(
                f"[Generation] Output update "
                f"{update_id}: {response}"
            )

            telegram_url = (
                "https://api.telegram.org/"
                f"bot{TELEGRAM_TOKEN}/sendMessage"
            )

            result = requests.post(
                telegram_url,
                json={
                    "chat_id": chat_id,
                    "text": response
                },
                timeout=30
            )

            print(
                f"[Telegram] Sent update "
                f"{update_id}: "
                f"{result.status_code}"
            )

            if not result.ok:
                print(
                    f"[Telegram] Send error "
                    f"{update_id}: "
                    f"{result.text}"
                )

        except Exception as e:
            print(
                f"[Telegram] Processing error "
                f"{update_id}:",
                repr(e)
            )


def send_whatsapp_message(
    to_number,
    message
):
    if not TWILIO_ACCOUNT_SID:
        print(
            "[WhatsApp] Missing TWILIO_ACCOUNT_SID."
        )

        return False

    if not TWILIO_AUTH_TOKEN:
        print(
            "[WhatsApp] Missing TWILIO_AUTH_TOKEN."
        )

        return False

    if not TWILIO_WHATSAPP_NUMBER:
        print(
            "[WhatsApp] Missing TWILIO_WHATSAPP_NUMBER."
        )

        return False

    try:
        twilio_url = (
            "https://api.twilio.com/2010-04-01/"
            f"Accounts/{TWILIO_ACCOUNT_SID}/"
            "Messages.json"
        )

        data = {
            "From": TWILIO_WHATSAPP_NUMBER,
            "To": to_number,
            "Body": message
        }

        response = requests.post(
            twilio_url,
            data=data,
            auth=(
                TWILIO_ACCOUNT_SID,
                TWILIO_AUTH_TOKEN
            ),
            timeout=30
        )

        print(
            "[WhatsApp] Twilio send status:",
            response.status_code
        )

        if not response.ok:
            print(
                "[WhatsApp] Twilio send error:",
                response.text
            )

            return False

        print(
            "[WhatsApp] Message sent successfully."
        )

        return True

    except Exception as e:
        print(
            "[WhatsApp] Send error:",
            repr(e)
        )

        return False


def process_whatsapp_message(
    message,
    sender,
):
    print(
        "[WhatsApp] Processing message:",
        message
    )

    with generation_lock:
        started_at = time.time()

        try:
            response = generate_response(
                message
            )

            elapsed = time.time() - started_at

            print(
                "[WhatsApp] Generation finished in "
                f"{elapsed:.2f}s"
            )

            print(
                "[WhatsApp] Output:",
                response
            )

            send_whatsapp_message(
                sender,
                response
            )

        except Exception as e:
            print(
                "[WhatsApp] Processing error:",
                repr(e)
            )


@app.route(
    "/telegram",
    methods=["POST"]
)
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
    ).get("id")

    print(
        f"[Telegram] Received update "
        f"{update_id}: {message}"
    )

    if not message or not chat_id:
        print(
            f"[Telegram] Ignored update "
            f"{update_id}"
        )

        return jsonify({
            "status": "ignored"
        }), 200

    print(
        f"[Telegram] Accepted update "
        f"{update_id}: {message}"
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


@app.route(
    "/whatsapp",
    methods=["POST"]
)
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

    if not incoming_message or not sender:
        return (
            "",
            200
        )

    threading.Thread(
        target=process_whatsapp_message,
        args=(
            incoming_message,
            sender
        ),
        daemon=True
    ).start()

    return (
        "",
        200
    )


@app.route(
    "/",
    methods=["GET"]
)
def health_check():
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
