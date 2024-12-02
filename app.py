from flask import Flask, render_template, Response
import cv2
from tensorflow.keras.models import load_model
import numpy as np

app = Flask(__name__)

# Load your trained model
model = load_model('emotion.h5')

# Define emotion labels (adjust based on your model's output classes)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Initialize the webcam
camera = cv2.VideoCapture(0)

def gen_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Convert to grayscale for emotion detection
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            resized_frame = cv2.resize(gray_frame, (48, 48))  # Resize to match input size of the model
            reshaped_frame = np.expand_dims(np.expand_dims(resized_frame, -1), 0) / 255.0
            
            # Predict emotion
            predictions = model.predict(reshaped_frame)
            emotion = emotion_labels[np.argmax(predictions)]

            # Display the emotion on the frame
            cv2.putText(frame, emotion, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Encode the frame for streaming
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # Yield the frame to the client
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
