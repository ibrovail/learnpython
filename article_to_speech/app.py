from flask import Flask, request, render_template, send_file
import boto3
import requests
from bs4 import BeautifulSoup
import os

app = Flask(__name__)

# Configure AWS Polly with region
polly_client = boto3.Session().client('polly', region_name='us-west-2')  # Specify your region

# Function to scrape article content from a URL
def get_article_content(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    article_text = ' '.join([p.text for p in soup.find_all('p')])
    return article_text

# Function to split the text into chunks of 3000 characters or less
def split_text(text, max_length=3000):
    return [text[i:i+max_length] for i in range(0, len(text), max_length)]

# Convert article content to speech using Polly
def text_to_speech(text, voice_id='Joanna', output_format='mp3'):
    # Split the text into smaller chunks
    text_chunks = split_text(text)
    
    # Create the static/ directory if it doesn't exist
    output_dir = 'static'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Prepare the output file path
    output_path = os.path.join(output_dir, 'output.mp3')

    # Remove the output file if it already exists
    if os.path.exists(output_path):
        os.remove(output_path)
    
    # Write the audio chunks to the output file
    with open(output_path, 'ab') as output_file:
        for chunk in text_chunks:
            response = polly_client.synthesize_speech(
                Text=chunk,
                OutputFormat=output_format,
                VoiceId=voice_id
            )
            output_file.write(response['AudioStream'].read())
    
    return output_path

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        article_url = request.form['url']
        # Scrape article content
        article_text = get_article_content(article_url)
        # Convert to speech using Polly
        audio_file = text_to_speech(article_text)
        return send_file(audio_file, as_attachment=True)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)