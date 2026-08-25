import os
import time
import threading
import requests
import torch

from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer


app = Flask(__name__)

MODEL_PATH = os.getenv(
    "MODEL_PATH",
    "solarvlaar/solarbot"
)

HF_TOKEN = (
    os.getenv("HF_TOKEN")
    or os.getenv("HUGGINGFACE_HUB_TOKEN")
)

TELEGRAM_TOKEN = os.getenv(
    "TELEGRAM_BOT_TOKEN"
)

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
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
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


generation_lock = threading.Lock()


def test_generation(
    prompt,
    chat_id,
    update_id
):

    print(
        f"[Generation] Starting update {update_id}"
    )

    start_time = time.time()

    try:

        input_text = (
            f"<|prompter|>\n"
            f"{prompt}\n"
            f"<|responder|>\n"
        )

        input_ids = tokenizer.encode(
            input_text,
            return_tensors="pt"
        ).to(device)

        print(
            f"[Generation] Input prepared "
            f"for update {update_id}"
        )

        with generation_lock:

            output_ids = model.generate(
                input_ids,
                do_sample=True,
                temperature=0.65,
                top_k=50,
                top_p=0.85,
                max_new_tokens=10,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        elapsed = (
            time.time()
            - start_time
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

        print(
            f"[Generation] Finished update "
            f"{update_id} in {elapsed:.2f}s"
        )

        print(
            f"[Generation] Output update "
            f"{update_id}: {response}"
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
            f"[Telegram] Sent update "
            f"{update_id}: "
            f"{result.status_code}"
        )

    except Exception as e:

        elapsed = (
            time.time()
            - start_time
        )

        print(
            f"[Generation] Failed update "
            f"{update_id} after "
            f"{elapsed:.2f}s:",
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
    ).get(
        "id"
    )

    print(
        f"[Telegram] Received update "
        f"{update_id}: {message}"
    )

    if not message or not chat_id:

        return jsonify({
            "status": "ignored"
        }), 200

    threading.Thread(
        target=test_generation,
        args=(
            message,
            chat_id,
            update_id
        ),
        daemon=True
    ).start()

    print(
        f"[Telegram] Accepted update "
        f"{update_id}"
    )

    return jsonify({
        "status": "accepted"
    }), 200


@app.route("/", methods=["GET"])
def health_check():

    if not READY:

        return (
            "Model unavailable",
            503
        )

    return (
        "Solarbot CPU test running",
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
