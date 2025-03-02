"""
IMPORTANT:
If you see an APIRemovedInV1 error when calling openai.ChatCompletion.create,
you have one of two options:
1. Run 'openai migrate' to update your code for openai>=1.0.0, OR
2. Pin your OpenAI library version to a pre-1.0.0 release (e.g., pip install openai==0.28.0)
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import openai
from gtts import gTTS
import os
from dotenv import load_dotenv
import tempfile

load_dotenv()

app = Flask(__name__)
CORS(app)

# Set your OpenAI API key from your environment
openai.api_key = os.getenv("OPENAI_API_KEY")

# In-memory conversation history (starting with an initial prompt)
history_messages = [
    {"role": "assistant", "content": "You are a helpful assistant. Ask me anything."}
]

@app.route('/process-speech', methods=['POST'])
def process_speech():
    data = request.json
    user_text = data.get("text", "")
    history_messages.append({"role": "user", "content": user_text})
    
    try:
        # This call uses the ChatCompletion endpoint.
        # If you get an APIRemovedInV1 error, please run 'openai migrate' or pin your OpenAI version.
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",  # Adjust model name as needed
            messages=history_messages
        )
    except Exception as e:
        return jsonify({"response": f"Error calling ChatCompletion: {str(e)}"}), 500

    ai_text = response.choices[0].message.content.strip()
    history_messages.append({"role": "assistant", "content": ai_text})
    
    return jsonify({"response": ai_text})

@app.route('/synthesize-speech', methods=['POST'])
def synthesize_speech():
    data = request.json
    text = data.get("text", "")
    
    # Use a temporary file to store the audio output
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        temp_filename = tmp.name

    tts = gTTS(text=text, lang='en')
    tts.save(temp_filename)
    
    # Return the audio file inline so the browser can play it
    response = send_file(temp_filename, mimetype="audio/mpeg", as_attachment=False)
    response.headers["Content-Disposition"] = "inline; filename=response.mp3"
    
    # Clean up the temporary file after response is sent
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
