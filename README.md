# 🧥 Outfit Rater: The AI-Powered Fashion Judge  

Outfit Rater is a fun Flask-based web app that rates your outfits in real time.  
It combines computer vision with AI-generated playful roasts to give you a score, style labels (like **Baggy/Tight**, **Single/Multilayer**), and even comments about your fashion choices.  

---

## ✨ Features
✅ **Webcam Outfit Capture**: Take live pictures of your outfit using your webcam.  
✅ **AI-Powered Rating**: Get a randomized yet realistic **style score out of 100**.  
✅ **Style Predictions**: Classifies your outfit’s **fit** (Baggy/Tight) and **complexity** (Single/Multilayer).  
✅ **Funny AI Comments**: Uses Google Gemini AI to roast or compliment your look in a witty way.  
✅ **Image Saving**: Automatically saves screenshots and past outfit captures to view later.  
✅ **Interactive Web UI**: Simple, responsive, and easy-to-use interface built with Flask templates.  

---

## 🛠️ Technology Stack
- **Python 3 🐍**  
- **Flask**: Backend framework for web app  
- **TensorFlow / Keras**: Pretrained deep learning models for fit & complexity classification  
- **OpenCV (cv2)**: Webcam capture and image processing  
- **Google Gemini API**: For generating playful outfit roasts and comments  
- **NumPy & Pillow (PIL)**: For numerical processing and image handling  

---

## 🚀 Getting Started
Follow these steps to run the Outfit Rater locally.

### 1️⃣ Clone the repository:
```bash
git clone https://github.com/VandanArora18/OutfitAI.git
```
2️⃣ Create and activate a virtual environment:
A virtual environment or Conda is highly recommended to keep dependencies clean.

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
3️⃣ Install dependencies:
```bash
pip install -r requirements.txt
```
4️⃣ Add your Google Gemini API Key:
```bash
genai.configure(api_key="YOUR_API_KEY_HERE")
```
5️⃣ Run the app:
```bash
python app.py
```
App will be available at:
👉 http://127.0.0.1:5000/
---

## 🎥 Usage
- Go to the Webcam page to capture your outfit.
- Hit Capture to take a snapshot.  
- Get your fit label, complexity label, random rating, and a funny AI-generated comment.  
- Visit Saved Outfits to see your past captures.






