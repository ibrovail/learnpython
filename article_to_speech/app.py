import os
from uuid import uuid4

import boto3
import requests
from bs4 import BeautifulSoup
from flask import Flask, render_template, request, send_file
from pypdf import PdfReader

AWS_REGION = "us-west-2"
VOICE_ID = "Joanna"
OUTPUT_FORMAT = "mp3"
POLLY_CHUNK_SIZE = 3000
OUTPUT_DIR = "static"
MAX_UPLOAD_MB = 20
REQUEST_TIMEOUT = 15
USER_AGENT = "Mozilla/5.0 (compatible; ArticleToSpeech/1.0)"

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

polly_client = boto3.Session().client("polly", region_name=AWS_REGION)


def get_article_content(url):
    response = requests.get(
        url, timeout=REQUEST_TIMEOUT, headers={"User-Agent": USER_AGENT}
    )
    response.raise_for_status()
    soup = BeautifulSoup(response.content, "html.parser")
    return " ".join(p.get_text(strip=True) for p in soup.find_all("p"))


def extract_pdf_text(file_storage):
    reader = PdfReader(file_storage)
    return "\n".join((page.extract_text() or "") for page in reader.pages).strip()


def split_text(text, max_length=POLLY_CHUNK_SIZE):
    return [text[i : i + max_length] for i in range(0, len(text), max_length)]


def text_to_speech(text):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, f"{uuid4().hex}.mp3")

    with open(output_path, "wb") as output_file:
        for chunk in split_text(text):
            response = polly_client.synthesize_speech(
                Text=chunk, OutputFormat=OUTPUT_FORMAT, VoiceId=VOICE_ID
            )
            output_file.write(response["AudioStream"].read())

    return output_path


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method != "POST":
        return render_template("index.html")

    try:
        pdf_file = request.files.get("pdf")
        url = (request.form.get("url") or "").strip()

        if pdf_file and pdf_file.filename:
            if not pdf_file.filename.lower().endswith(".pdf"):
                return render_template("index.html", error="File must be a PDF."), 400
            text = extract_pdf_text(pdf_file)
        elif url:
            text = get_article_content(url)
        else:
            return render_template("index.html", error="Provide a URL or a PDF."), 400

        if not text:
            return render_template("index.html", error="No text could be extracted."), 400

        audio_file = text_to_speech(text)
        return send_file(audio_file, as_attachment=True)
    except requests.RequestException as e:
        return render_template("index.html", error=f"Could not fetch URL: {e}"), 502
    except Exception as e:
        return render_template("index.html", error=f"Conversion failed: {e}"), 500


if __name__ == "__main__":
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1")
