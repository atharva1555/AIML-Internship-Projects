# ✍️ MNIST Handwritten Digit Recogniser

A deep learning project that recognises handwritten digits (0–9) using a **Dense Neural Network** trained on the MNIST dataset, with a beautiful **Streamlit web interface**.

---

## 📸 Features

- 🧠 Train a neural network directly from the browser
- ✍️ Draw a digit on canvas and get instant prediction
- 📁 Upload your own handwritten digit image for prediction
- 📊 View confusion matrix, per-class accuracy and sample predictions
- 💾 Auto-saves and auto-loads trained model — no retraining needed every time

---

## 🗂️ Project Structure

```
mnist_digit_classification_using_nn/
│
├── mnist_dc.py           ← Original training script
├── app.py                ← Streamlit frontend
├── mnist_model.keras     ← Saved model (generated after training)
└── README.md             ← This file
```

---

## 🧠 Model Architecture

```
Input (28x28 image)
        ↓
   Flatten → 784 neurons
        ↓
   Dense (50, ReLU)
        ↓
   Dense (50, ReLU)
        ↓
   Dense (10, Softmax)  ← Output: probability for each digit 0-9
```

| Layer | Type | Neurons | Activation |
|-------|------|---------|------------|
| 1 | Flatten | 784 | — |
| 2 | Dense | 50 | ReLU |
| 3 | Dense | 50 | ReLU |
| 4 | Dense (Output) | 10 | Softmax |

---

## 📦 Requirements

Install all dependencies with:

```bash
pip install tensorflow keras numpy matplotlib seaborn opencv-python pillow scikit-learn streamlit streamlit-drawable-canvas
```

---

## 🚀 How to Run

### Step 1 — Train and save the model
Run the original training script first:
```bash
python mnist_dc.py
```
This will train the model for 10 epochs and save it as `mnist_model.keras`.

### Step 2 — Launch the Streamlit app
```bash
streamlit run mnist_app.py
```

> 💡 If `mnist_model.keras` already exists, the app loads it automatically — no retraining needed!

---

## 🖥️ App Tabs

### 🧠 Train Model
- Adjust **Epochs**, **Hidden Units**, and **Learning Rate** using sliders
- Click **Train Neural Network** to train with live progress bar
- View **Accuracy** and **Loss** training curves after training

### ✍️ Predict Digit
- **Draw mode** — Draw a digit on the canvas and click Predict
- **Upload mode** — Upload any handwritten digit image (PNG/JPG)
- See the predicted digit, confidence score, and probability bar chart for all 10 classes

### 📊 Analytics
- **Confusion Matrix** — See where the model makes mistakes
- **Per-Class Accuracy** — Accuracy breakdown for each digit 0–9
- **Sample Test Images** — Grid of 30 test predictions (green = correct, red = wrong)

---

## ⚙️ Hyperparameters Explained

| Parameter | What it does | Default | Recommended Range |
|-----------|-------------|---------|-------------------|
| **Epochs** | How many times model trains over all 60,000 images | 10 | 5–15 |
| **Hidden Units** | Number of neurons in each hidden layer | 50 | 50–128 |
| **Learning Rate** | How fast the model updates its weights | 0.001 | 0.001 (Adam default) |

---

## 📊 Dataset — MNIST

| Property | Value |
|----------|-------|
| Total Images | 70,000 |
| Training Images | 60,000 |
| Test Images | 10,000 |
| Image Size | 28 × 28 pixels |
| Classes | 10 (digits 0–9) |
| Pixel Range | 0–255 (normalized to 0–1) |

---

## 📈 Expected Results

| Metric | Value |
|--------|-------|
| Training Accuracy | ~98% |
| Test Accuracy | ~97% |
| Epochs | 10 |

---

## 🛠️ Technologies Used

- **Python 3.x**
- **TensorFlow / Keras** — Neural network
- **NumPy** — Numerical operations
- **OpenCV** — Image preprocessing
- **Matplotlib / Seaborn** — Visualizations
- **Scikit-learn** — Confusion matrix
- **Streamlit** — Web interface
- **streamlit-drawable-canvas** — Draw digit on canvas

---

## 👨‍💻 Author

Made as part of **AIML Internship Project**
