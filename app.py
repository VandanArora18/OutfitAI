from flask import Flask, render_template, Response, request, redirect, url_for, jsonify
import cv2
import numpy as np
import tensorflow as tf
import os
import random
from PIL import Image
import google.generativeai as genai
import datetime
import base64

app = Flask(__name__)
fit_model = tf.keras.models.load_model("fit_tightness_best_model.h5")
complexity_model = tf.keras.models.load_model("complexity_best_model.h5")
fit_class_names = ['Baggy', 'Tight']
complexity_class_names = ['Multilayer', 'Single Layer']
input_height, input_width = fit_model.input_shape[1:3]

genai.configure(api_key="AIzaSyDPmEp36vj--CN116xyiTxHsfYHIAHA6y4") 
ai_model = genai.GenerativeModel("gemini-1.5-flash")
def preprocess_for_model(image):
    img_resized = cv2.resize(image, (input_width, input_height))
    img_norm = img_resized / 255.0
    return np.expand_dims(img_norm, axis=0)

def predict_outfit(image):
    input_data = preprocess_for_model(image)
    fit_pred = fit_model.predict(input_data, verbose=0)
    complexity_pred = complexity_model.predict(input_data, verbose=0)
    fit_label = fit_class_names[np.argmax(fit_pred)]
    complexity_label = complexity_class_names[np.argmax(complexity_pred)]
    return fit_label, complexity_label

def generate_ai_comment(img_path):
    image = Image.open(img_path)
    prompt = (
        "Look at this outfit and give a funny, playful roast in 3-4 words "
        "(slight, sassy, cute, based on clothing style, color, or accessories)."
    )
    response = ai_model.generate_content([prompt, image])
    return response.text

def get_random_rating():
    return f"{random.randint(59,100)}/100"
cap = cv2.VideoCapture(0)
captured_image_path = None
latest_frame = None

def gen_frames():
    global latest_frame
    while True:
        success, frame = cap.read()
        if not success:
            break
        latest_frame = frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/webcam')
def webcam():
    return render_template("webcam.html")

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture', methods=['POST'])
def capture():
    global captured_image_path, latest_frame
    if latest_frame is not None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs("static/images/captured", exist_ok=True)
        img_path = f"static/images/captured/{timestamp}.jpg"
        cv2.imwrite(img_path, latest_frame)
        captured_image_path = img_path
    return redirect(url_for('result'))

@app.route('/result')
def result():
    if not captured_image_path:
        return redirect(url_for('webcam'))
    fit_label, complexity_label = predict_outfit(cv2.imread(captured_image_path))
    rating = get_random_rating()  # generate rating
    return render_template("result.html", img_path=captured_image_path, comment=None,
                           fit_label=fit_label, complexity_label=complexity_label,
                           rating=rating)  # pass rating to template

@app.route('/get_comment')
def get_comment():
    if not captured_image_path:
        return jsonify({"comment": ""})
    comment = generate_ai_comment(captured_image_path)
    return jsonify({"comment": comment})

@app.route('/save_ss', methods=['POST'])
def save_ss():
    data = request.get_json()
    image_data = data['image'].split(",")[1]
    img_bytes = base64.b64decode(image_data)

    save_dir = os.path.join('static', 'images', 'screenshots')
    os.makedirs(save_dir, exist_ok=True)

    filename = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".png"
    filepath = os.path.join(save_dir, filename)
    with open(filepath, 'wb') as f:
        f.write(img_bytes)

    return jsonify({"filename": filename})

@app.route('/saved_images')
def saved_images():
    ss_dir = "static/images/screenshots"
    os.makedirs(ss_dir, exist_ok=True)
    images = os.listdir(ss_dir)
    images = [f"images/screenshots/{img}" for img in images]
    return render_template("saved.html", images=images)

if __name__ == "__main__":
    app.run(debug=True)
