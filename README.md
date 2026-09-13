# The Ingredient Recognizer

This repository contains the standalone AI engine (YOLOv8) trained to recognize 120 different food ingredients. 

## How to Test the Recognizer on Your Laptop (Webcam)

If you want to test the AI model live using your laptop's webcam, follow these steps:

### 1. Prerequisites
You must have **Python** installed on your computer. 

### 2. Extract the AI Model
Extract the `ingredient_recognizer.zip` file. Inside, you will find a file named `best.pt`.
**Move `best.pt` into this main folder** (the same folder as `test_webcam.py`).

### 3. Install Dependencies
Open your terminal (Command Prompt or PowerShell) in this folder and install the required AI libraries by running:
```bash
pip install -r requirements.txt
```

### 4. Run the Webcam Test
Once the installation is complete, run the webcam script:
```bash
python test_webcam.py
```

A window will pop up showing your webcam feed. Hold up an ingredient (like an egg, garlic, or onion) to the camera, and the AI will draw a bounding box around it if it recognizes it!

Press the **'q'** key on your keyboard while clicked into the camera window to close it.
