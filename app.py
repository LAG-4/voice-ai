"""
IMPORTANT:
If you see an APIRemovedInV1 error when calling openai.ChatCompletion.create,
you have one of two options:
1. Run 'openai migrate' to update your code for openai>=1.0.0, OR
2. Pin your OpenAI library version to a pre-1.0.0 release (e.g., pip install openai==0.28.0)
(This example, however, uses an open-source model for text generation via Groq.)
"""

import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from gtts import gTTS
import tempfile
from groq import Groq

load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

app = Flask(__name__)
CORS(app)

# In-memory conversation history (starting with an initial prompt)
history_messages = [
    {"role": "assistant", "content": "You are a helpful assistant. Ask me anything."}
]

# Initialize Groq client for transcription and text generation
groq_client = Groq()

def generate_response_with_stream(messages):
    """
    Generates a response using Groq's chat completions with streaming.
    Expects `messages` to be a list of dictionaries with keys "role" and "content".
    """
    completion = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=1,
        max_completion_tokens=100,
        top_p=1,
        stream=True,
        stop=None,
    )
    result = ""
    for chunk in completion:
        result += chunk.choices[0].delta.content or ""
    return result.strip()

@app.route('/process-speech', methods=['POST'])
def process_speech():
    """
    This endpoint expects a JSON payload with a "text" field.
    It appends the text to the conversation history,
    generates a reply using Groq's chat completions with streaming,
    and returns the AI response.
    """
    data = request.json
    user_text = data.get("text", "")
    history_messages.append({"role": "user", "content": user_text})
    
    prompt = history_messages  # Already in the expected format
    ai_text = generate_response_with_stream(prompt)
    history_messages.append({"role": "assistant", "content": ai_text})
    
    return jsonify({"response": ai_text})

@app.route('/process-audio', methods=['POST'])
def process_audio():
    """
    This endpoint accepts an audio file (multipart/form-data with key 'file'),
    transcribes it using Groq's transcription (with model "whisper-large-v3"),
    then uses the transcription as a query for text generation.
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    
    # Save the uploaded file to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".m4a", delete=False) as tmp:
        temp_filename = tmp.name
        file.save(temp_filename)
    
    # Transcribe the audio file using Groq
    with open(temp_filename, "rb") as f:
        transcription_result = groq_client.audio.transcriptions.create(
            file=(temp_filename, f.read()),
            model="whisper-large-v3",
            response_format="verbose_json",
        )
    os.remove(temp_filename)
    
    user_text = transcription_result.text.strip()
    history_messages.append({"role": "user", "content": user_text})
    
    prompt = history_messages
    ai_text = generate_response_with_stream(prompt)
    history_messages.append({"role": "assistant", "content": ai_text})
    
    return jsonify({
        "transcription": user_text,
        "response": ai_text
    })

@app.route('/synthesize-speech', methods=['POST'])
def synthesize_speech():
    """
    This endpoint converts provided text to speech using gTTS and returns
    the audio file inline so the browser can play it.
    """
    data = request.json
    text = data.get("text", "")
    
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        temp_filename = tmp.name

    tts = gTTS(text=text, lang='en')
    tts.save(temp_filename)
    
    response = send_file(temp_filename, mimetype="audio/mpeg", as_attachment=False)
    response.headers["Content-Disposition"] = "inline; filename=response.mp3"
    
    @response.call_on_close
    def cleanup():
        try:
            os.remove(temp_filename)
        except Exception as e:
            print("Error removing temp file:", e)
    
    return response

if __name__ == "__main__":
    # Run the Flask app on port 5000
    app.run(host="0.0.0.0", port=5000, debug=True)
