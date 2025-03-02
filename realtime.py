from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from flask_cors import CORS
from dotenv import load_dotenv
import openai
import os
load_dotenv()
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'

CORS(app)  # enable CORS for Flask routes
socketio = SocketIO(app, cors_allowed_origins="*")  # Keep this initialization
openai.api_key = os.getenv("OPENAI_API_KEY")
@app.route('/')
def index():
    return render_template('index.html')


def generate_response(message):
    """
    Example of calling an OpenAI GPT model. 
    Replace with your actual GPT-4o-mini-realtime-preview integration if different.
    """
    # For ChatCompletion:
    try:
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # or whichever model you want to use
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": message}
            ],
            max_tokens=100,
            temperature=0.7
        )
        # Extract the assistant’s message
        bot_reply = completion.choices[0].message.content
        return bot_reply
    except Exception as e:
        print("Error calling OpenAI API:", e)
        return "Sorry, I’m having trouble generating a response."
@socketio.on('user_message')
def handle_user_message(data):
    user_message = data.get('message', '')
    print("Received message:", user_message)
    bot_response = generate_response(user_message)
    emit('bot_response', {'message': bot_response})

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
