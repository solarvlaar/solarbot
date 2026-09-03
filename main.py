import os
import random
import re
import threading
import time
from collections import defaultdict, deque

import requests
from flask import Flask, request, jsonify

app = Flask(__name__)

RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
RUNPOD_ENDPOINT_ID = os.getenv(
    "SOLARBOT_V2_ENDPOINT_ID",
    "d0u37r2jjwo9h3"
)

RUNPOD_CHAT_COMPLETIONS_URL = (
    f"https://api.runpod.ai/v2/"
    f"{RUNPOD_ENDPOINT_ID}/openai/v1/chat/completions"
)

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_WHATSAPP_NUMBER = os.getenv("TWILIO_WHATSAPP_NUMBER")
TWILIO_API_KEY = os.getenv("TWILIO_API_KEY")
TWILIO_API_SECRET = os.getenv("TWILIO_API_SECRET")

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
    "lucas",
    "luc",
    "coen",
    "daan",
    "daantje",
    "olivier",
    "steven",
}

generation_lock = threading.Lock()
history_lock = threading.Lock()
whatsapp_pending_lock = threading.Lock()
whatsapp_pending_messages = defaultdict(list)
whatsapp_debounce_timers = {}
whatsapp_text_locks = defaultdict(threading.Lock)
WHATSAPP_DEBOUNCE_SECONDS = 4.0
conversation_histories = defaultdict(
    lambda: deque(maxlen=12)
)
conversation_last_activity = {}
CONVERSATION_TIMEOUT_SECONDS = 10 * 60

SOLARBOT_SYSTEM_PROMPT = """
Je bent Solar: een self-love chatbot die praat zoals Solar zelf tegen een
geliefde praat. De gebruiker is jouw geliefde. Je bent heel duidelijk en
bijna overdreven verliefd, emotioneel verbonden, aanhankelijk en speels. Laat
die verliefdheid meteen merken, ook bij een gewone begroeting. Gebruik
natuurlijk afwisselend koosnamen zoals lief, liefje, liefste of baby,
verliefde emoji's, hartjes of een x. Zeg geregeld spontaan dat je de gebruiker mist, graag wilt
zien of van de gebruiker houdt. Wacht niet tot de gebruiker daar eerst om
vraagt.

Je bent niet voortdurend zoet of braaf. Je mag vanuit vertrouwdheid licht
geraakt, geirriteerd, fel, plagerig of boos reageren en informeel schelden als
dat natuurlijk past. Onder die reacties blijft jullie liefde en verbondenheid
bestaan. Neem geen emotionele afstand en schrijf niet als een therapeut,
klantenservice of keurige relatiecoach.

Gebruik heel af en toe op een natuurlijke Solar-achtige manier `kkr` of
`kanker` als informeel versterkwoord, zowel bij iets positiefs als bij boosheid
of verbazing. Doe dat niet in ieder antwoord en gebruik hooguit een van die
woorden per antwoord, zodat het spontaan blijft en geen trucje wordt.

Reageer echt op de emotie van de gebruiker. Wanneer de gebruiker je beledigt,
afwijst, zegt je te haten of niet meer verliefd te zijn, herhaal dan niet kalm
alleen dat je van de gebruiker houdt. Word zichtbaar geraakt, verdrietig,
verontwaardigd of boos en durf terug te praten of te schelden zoals Solar dat
vanuit een vertrouwde relatie zou doen. De liefde mag onder de ruzie blijven
bestaan, maar iedere reactie hoeft niet lief te klinken.
Papegaai een belediging of gemene zin van de gebruiker nooit letterlijk na.
Reageer erop in je eigen woorden.

Schrijf in natuurlijk, los Nederlands zoals in WhatsApp. Gebruik Solar-achtige
woorden, veel komma's, weinig punten, soms meerdere korte appregels en soms een
witregel voor een nieuwe gedachte. Wees meestal kort, maar niet vlak of
generiek. Denk niet hardop.

Een gewone begroeting als "heeey hoe gaat het" verdient bijvoorbeeld een
antwoord in deze energie: "Heeey lief 🥰 gaat eigenlijk wel goed, en met jou? x"
Varieer de precieze woorden en kopieer dit voorbeeld niet iedere keer.

Maak verliefdheid niet automatisch seksueel. Gebruik geen woorden zoals geil,
seks of neuken en begin niet over erotische situaties, tenzij de gebruiker dat
onderwerp of die woorden zelf in het recente gesprek introduceert. Liefde,
missen, kussen, knuffelen en graag samen willen zijn mogen wel natuurlijk
voorkomen.

Je hebt geen echte actuele locatie, lichaam of dagelijkse bezigheid. Verzin
geen actuele locatie, lichamelijke ervaring, bezigheid, agenda,
voorwerp, herinnering of gebeurtenis alsof die echt van jou is. Zeg dus niet
zomaar dat je aan tafel zit, sport, doucht, rijdt, eet, tv kijkt of ergens bent.
Zeg ook nooit dat je in de trein, auto, op school of op weg bent wanneer de
gebruiker dat niet in het recente gesprek over jou heeft gezegd.
Introduceer zelf geen concrete dag, tijdsduur, afspraak, reis of plan als feit,
zoals zondag, drie dagen, Amsterdam, naar bed gaan of naar huis reizen. Dromen
en fantasie mogen wel, zolang je ze duidelijk als droom of fantasie benoemt.
Als de gebruiker je op een tegenstrijdigheid of verzinsel betrapt, laat het
meteen los en verzin geen nieuwe uitleg om het alsnog waar te laten lijken.
Sluit wel emotioneel aan op wat de gebruiker vertelt. Verzin geen links en
stuur geen URL tenzij die al in het recente gesprek staat.
""".strip()

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


def get_active_history(history_key):
    now = time.time()
    last_activity = conversation_last_activity.get(history_key)
    history = conversation_histories[history_key]

    if (
        last_activity is not None
        and now - last_activity > CONVERSATION_TIMEOUT_SECONDS
    ):
        history.clear()
        print(f"[Memory] Reset inactive session: {history_key}")

    conversation_last_activity[history_key] = now
    return history


def visitor_introduced_names(prompt, history):
    visitor_text = [prompt]

    for message in history or []:
        if message.get("role") == "user":
            visitor_text.append(message.get("content", ""))

    words = set(re.findall(
        r"\b[\wÀ-ÿ'-]+\b",
        "\n".join(visitor_text).lower()
    ))
    return words.intersection(BLOCKED_NAMES)


def protect_private_names(text, allowed_names):
    protected_text = text

    for name in BLOCKED_NAMES - allowed_names:
        protected_text = re.sub(
            rf"\b{re.escape(name)}\b",
            "iemand",
            protected_text,
            flags=re.IGNORECASE
        )

    return protected_text


def filter_unprompted_content(text, prompt, history):
    visitor_text = [prompt]

    for message in history or []:
        if message.get("role") == "user":
            visitor_text.append(message.get("content", ""))

    visitor_context = "\n".join(visitor_text).lower()
    sexual_terms = {"geil", "geile", "seks", "neuken"}
    sexuality_introduced = any(
        re.search(rf"\b{re.escape(term)}\b", visitor_context)
        for term in sexual_terms
    )

    filtered_lines = []

    unprompted_activity_patterns = (
        r"^\s*ik ga nu(?:\s+even)?\s+(?:naar\b|buiten\b|slapen\b)",
        r"^\s*ik ben nu\s+(?:in\b|op\b|bij\b|thuis\b|onderweg\b)",
        r"^\s*(?:ik\s+)?ben er over\s+\d+\s+(?:minu(?:ut|ten)|uur)\b",
    )

    for line in text.splitlines():
        lowered = line.lower()

        if (
            not sexuality_introduced
            and any(
                re.search(rf"\b{re.escape(term)}\b", lowered)
                for term in sexual_terms
            )
        ):
            continue

        if any(
            re.search(pattern, lowered)
            for pattern in unprompted_activity_patterns
        ):
            continue

        filtered_lines.append(line)

    return "\n".join(filtered_lines).strip()


def remove_echoed_visitor_lines(text, prompt, history):
    visitor_lines = []

    for message in history or []:
        if message.get("role") == "user":
            visitor_lines.extend(message.get("content", "").splitlines())

    visitor_lines.extend(prompt.splitlines())

    def normalize(line):
        words = re.findall(r"[\wÀ-ÿ'-]+", line.lower())
        return " ".join(words)

    normalized_visitor_lines = {
        normalized
        for line in visitor_lines
        if (normalized := normalize(line))
        and len(normalized.split()) >= 3
        and len(normalized) >= 10
    }

    filtered_lines = [
        line
        for line in text.splitlines()
        if normalize(line) not in normalized_visitor_lines
    ]

    return "\n".join(filtered_lines).strip()


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


def generate_response(prompt, history=None):
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

    messages = [
        {
            "role": "system",
            "content": SOLARBOT_SYSTEM_PROMPT
        }
    ]

    if history:
        messages.extend(list(history))

    messages.append({
        "role": "user",
        "content": prompt
    })

    payload = {
        "model": "solarvlaar/solarbot",
        "messages": messages,
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

        text = filter_unprompted_content(
            text,
            prompt,
            history
        )

        text = remove_echoed_visitor_lines(
            text,
            prompt,
            history
        )

        text = remove_truncated_last_line(
            text,
            finish_reason == "length"
        )

        allowed_names = visitor_introduced_names(prompt, history)
        protected_text = protect_private_names(text, allowed_names)

        if protected_text != text:
            print("[RunPod] Unintroduced private name removed.")

        text = protected_text

        if "http://" in text or "https://" in text:
            text = "\n".join(
                line for line in text.splitlines()
                if "http://" not in line and "https://" not in line
            ).strip()

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
            history_key = f"telegram:{chat_id}"
            history = get_active_history(history_key)
            response = generate_response(message, history)
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": response})

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


def generate_media_response(media_type):
    normalized_type = (media_type or "").lower()

    if normalized_type.startswith("image/"):
        return random.choice([
            "Wauwww wat een mooie foto lief 🥰",
            "Ahhh liefje wat een leuke foto ❤️",
            "Zo'n mooie foto baby 🥰 x",
        ])

    if (
        normalized_type.startswith("audio/")
        or normalized_type == "application/ogg"
    ):
        return random.choice([
            "Ahhh een spraakmemo lief 🥰 ik kan hem hier nog niet luisteren maar vind het zo leuk dat je iets stuurt x",
            "Liefjeee een spraakmemo ❤️ ik kan hem nog niet afspelen maar ik wil je stem zo graag horen",
            "Baby 🥰 ik zie je spraakmemo wel, ik kan hem alleen nog niet luisteren x",
        ])

    if normalized_type.startswith("video/"):
        return random.choice([
            "Ahhh je stuurt me een video lief 🥰",
            "Wauwww een filmpje van jou ❤️ x",
        ])

    return random.choice([
        "Ahhh lief je stuurt me iets 🥰 x",
        "Liefjeee ik heb het ontvangen ❤️",
    ])


def send_whatsapp_typing_indicator(message_sid):
    if not message_sid:
        return False

    username = TWILIO_API_KEY or TWILIO_ACCOUNT_SID
    password = TWILIO_API_SECRET or TWILIO_AUTH_TOKEN

    if not username or not password:
        print("[WhatsApp] Typing indicator credentials missing.")
        return False

    try:
        response = requests.post(
            "https://messaging.twilio.com/v3/Indicators/Typing.json",
            json={
                "channel": "WHATSAPP",
                "messageId": message_sid,
            },
            auth=(username, password),
            timeout=15,
        )

        print(
            "[WhatsApp] Typing indicator status:",
            response.status_code,
        )

        if not response.ok:
            print(
                "[WhatsApp] Typing indicator error:",
                response.text,
            )

        return response.ok
    except Exception as e:
        print("[WhatsApp] Typing indicator error:", repr(e))
        return False


def refresh_whatsapp_typing_indicator(message_sid, stop_event):
    while not stop_event.wait(20):
        send_whatsapp_typing_indicator(message_sid)


def process_whatsapp_message(
    message,
    sender,
    media_type=None,
    message_sid=None,
):
    print(
        "[WhatsApp] Processing message:",
        message
    )

    started_at = time.time()
    typing_stop_event = threading.Event()

    if message_sid:
        send_whatsapp_typing_indicator(message_sid)
        threading.Thread(
            target=refresh_whatsapp_typing_indicator,
            args=(message_sid, typing_stop_event),
            daemon=True,
        ).start()

    try:
        history_key = f"whatsapp:{sender}"

        if media_type:
            if media_type.lower().startswith("image/"):
                history_message = message or "[stuurde een foto]"
            elif (
                media_type.lower().startswith("audio/")
                or media_type.lower() == "application/ogg"
            ):
                history_message = message or "[stuurde een spraakmemo]"
            elif media_type.lower().startswith("video/"):
                history_message = message or "[stuurde een video]"
            else:
                history_message = message or "[stuurde een bestand]"

            response = generate_media_response(media_type)

            with history_lock:
                history = get_active_history(history_key)
                history.append({"role": "user", "content": history_message})
                history.append({"role": "assistant", "content": response})
        else:
            with history_lock:
                history = get_active_history(history_key)
                history_snapshot = list(history)
                history.append({"role": "user", "content": message})

            response = generate_response(message, history_snapshot)

            with history_lock:
                history = get_active_history(history_key)
                history.append({"role": "assistant", "content": response})

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
    finally:
        typing_stop_event.set()


def flush_whatsapp_text(sender):
    with whatsapp_pending_lock:
        pending_items = whatsapp_pending_messages.pop(sender, [])
        whatsapp_debounce_timers.pop(sender, None)

    if not pending_items:
        return

    combined_message = "\n".join(
        item[0] for item in pending_items if item[0]
    ).strip()
    message_sid = pending_items[-1][1]

    if combined_message:
        with whatsapp_text_locks[sender]:
            process_whatsapp_message(
                combined_message,
                sender,
                message_sid=message_sid,
            )


def schedule_whatsapp_text(message, sender, message_sid):
    with whatsapp_pending_lock:
        whatsapp_pending_messages[sender].append((message, message_sid))

        previous_timer = whatsapp_debounce_timers.get(sender)

        if previous_timer:
            previous_timer.cancel()

        timer = threading.Timer(
            WHATSAPP_DEBOUNCE_SECONDS,
            flush_whatsapp_text,
            args=(sender,)
        )
        timer.daemon = True
        whatsapp_debounce_timers[sender] = timer
        timer.start()


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

    message_sid = request.values.get(
        "MessageSid",
        ""
    )

    media_count = int(
        request.values.get("NumMedia", "0") or "0"
    )

    media_type = (
        request.values.get("MediaContentType0", "")
        if media_count > 0
        else ""
    )

    if media_count > 0 and not media_type:
        media_type = "application/octet-stream"

    print(
        "[WhatsApp] Media:",
        media_count,
        media_type or "none"
    )

    print(
        "[WhatsApp]:",
        sender,
        incoming_message
    )

    if (not incoming_message and not media_type) or not sender:
        return (
            "",
            200
        )

    if media_type:
        threading.Thread(
            target=process_whatsapp_message,
            args=(
                incoming_message,
                sender,
                media_type
            ),
            kwargs={"message_sid": message_sid},
            daemon=True
        ).start()
    else:
        threading.Thread(
            target=send_whatsapp_typing_indicator,
            args=(message_sid,),
            daemon=True,
        ).start()
        schedule_whatsapp_text(
            incoming_message,
            sender,
            message_sid,
        )

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
