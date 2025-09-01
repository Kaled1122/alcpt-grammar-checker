import os
import json
import csv
import tempfile
from datetime import datetime
from typing import Optional
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
from dotenv import load_dotenv

# ---------------------------
# Config & setup
# ---------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY in environment")

client = OpenAI(api_key=OPENAI_API_KEY)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # ✅ Allow all origins

# ---------------------------
# Load grammar points
# ---------------------------
GRAMMAR_JSON_PATH = "grammar_points.json"

def load_grammar_points(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find grammar file at: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

GRAMMAR_POINTS = load_grammar_points(GRAMMAR_JSON_PATH)

# ---------------------------
# Prompt builder
# ---------------------------
def grammar_points_block():
    return "\n".join([
        f"{p['id']}. {p['title']} - {p['rule']} Example: {p['example']}"
        for p in GRAMMAR_POINTS
    ])

def make_prompt(user_text: str, selected_id: Optional[int]) -> str:
    selected_block = ""
    if selected_id:
        selected_point = next((p for p in GRAMMAR_POINTS if p["id"] == int(selected_id)), None)
        if selected_point:
            selected_block = f"\nFocus on grammar point #{selected_point['id']}: {selected_point['title']}"

    return f"""
You are an ESL grammar coach. Analyze the learner’s sentence (always assume it should be English).
Reply ONLY in valid JSON with:
- corrected: string
- explanation: string (simple explanation for learner)
- grammar_ok: boolean
- score: integer 0-100
- matched_grammar_id: integer 1-70
- matched_grammar_label: string

Grammar Points:
{grammar_points_block()}
{selected_block}

Learner sentence (English only expected):
\"\"\"{user_text}\"\"\"
"""

# ---------------------------
# Save learner logs (CSV only)
# ---------------------------
LOG_FILE = "learner_logs.csv"

def save_log(learner_id, data):
    row = {
        "learner_id": learner_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "transcript": data.get("transcript"),
        "correction": data.get("corrected"),
        "explanation": data.get("explanation"),
        "score": data.get("score"),
        "grammar_point": data.get("matched_grammar_label"),
        "selected_point": data.get("selected_grammar_label"),
    }
    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

# ---------------------------
# Whisper transcription (English only)
# ---------------------------
def transcribe_audio_to_text(file_storage) -> str:
    if file_storage is None:
        return ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
            file_storage.save(tmp.name)
            tmp_path = tmp.name
        with open(tmp_path, "rb") as audio_file:
            tr = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language="en"   # 🔒 Force English transcription
            )
        return tr.text.strip() if hasattr(tr, "text") else ""
    except Exception as e:
        print("Transcription error:", e)
        return ""

# ---------------------------
# GPT grammar check
# ---------------------------
def run_grammar_llm(user_text: str, grammar_id: Optional[int]) -> dict:
    prompt = make_prompt(user_text, grammar_id)

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {"role": "system", "content": "You are a JSON-only ESL grammar evaluator. Always output ONLY valid JSON. Assume learner input is English only."},
            {"role": "user", "content": prompt}
        ]
    )

    raw = completion.choices[0].message.content.strip()
    try:
        result = json.loads(raw)
    except Exception:
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1:
            try:
                result = json.loads(raw[start:end+1])
            except Exception:
                result = {"error": "Could not parse GPT response", "raw": raw}
        else:
            result = {"error": "Invalid response", "raw": raw}
    return result

# ---------------------------
# Routes
# ---------------------------
@app.route("/")
def home():
    return "Grammar checker backend running"

@app.route("/api/grammar-points", methods=["GET"])
def api_grammar_points():
    return jsonify(GRAMMAR_POINTS)

@app.route("/api/text", methods=["POST"])
def api_text():
    learner_id = request.form.get("learner_id", "").strip()
    grammar_id = request.form.get("grammar_id", "").strip()
    typed = request.form.get("typed", "").strip()

    if not learner_id.isdigit():
        return jsonify({"error": "Learner ID (numbers only) is required."}), 400
    if not grammar_id.isdigit():
        return jsonify({"error": "Grammar point selection is required."}), 400
    if not typed:
        return jsonify({"error": "No text provided."}), 400

    sel_id = int(grammar_id)
    out = run_grammar_llm(typed, sel_id)

    out["transcript"] = typed
    out["selected_grammar_label"] = next(
        (p["title"] for p in GRAMMAR_POINTS if p["id"] == sel_id),
        None
    )
    save_log(learner_id, out)
    return jsonify(out)

@app.route("/api/grammar", methods=["POST"])
def api_grammar():
    learner_id = request.form.get("learner_id", "").strip()
    grammar_id = request.form.get("grammar_id", "").strip()

    if not learner_id.isdigit():
        return jsonify({"error": "Learner ID (numbers only) is required."}), 400
    if not grammar_id.isdigit():
        return jsonify({"error": "Grammar point selection is required."}), 400

    typed = request.form.get("typed", "").strip()
    transcript = typed

    if not typed:
        file = request.files.get("audio")
        transcript = transcribe_audio_to_text(file)

    if not transcript:
        return jsonify({"error": "No speech or text found."}), 400

    sel_id = int(grammar_id)
    out = run_grammar_llm(transcript, sel_id)

    out["transcript"] = transcript
    out["selected_grammar_label"] = next(
        (p["title"] for p in GRAMMAR_POINTS if p["id"] == sel_id),
        None
    )
    save_log(learner_id, out)
    return jsonify(out)

# ---------------------------
# Render-compatible start
# ---------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
