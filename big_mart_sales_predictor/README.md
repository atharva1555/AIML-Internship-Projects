# 🛒 Big Mart Sales Prediction

This project predicts the sales of products at Big Mart outlets using Machine Learning.  
The model is trained on historical sales data and deployed using Streamlit for an interactive user interface.

---

## 📌 Project Overview
Retail businesses need accurate sales predictions to manage inventory, pricing, and demand.  
This project uses regression techniques to estimate the Item Outlet Sales based on product and outlet features.

---

## ⚙️ Technologies Used
- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- Streamlit

---

## 📊 Features Used for Prediction
- Item Identifier
- Item Weight
- Item Fat Content
- Item Visibility
- Item Type
- Item MRP
- Outlet Identifier
- Outlet Establishment Year
- Outlet Size
- Outlet Location Type
- Outlet Type

Target Variable:
- Item Outlet Sales

---

## 🧠 Machine Learning Model
The project uses:
- XGBoost Regressor for prediction
- Label Encoding for categorical variables
- Train-test split for evaluation
- R² score for performance measurement

---

## 🚀 How to Run the Project

### 1️⃣ Install dependencies
pip install -r requirements.txt

### 2️⃣ Train model
python mart_sales_prediction.py

### 3️⃣ Run Streamlit App
streamlit run app.py

---

## 📁 Project Structure
Big_Mart_Sales_Prediction/
│
├── app.py
├── mart_sales_prediction.py
├── bigmart_model.pkl
├── requirements.txt
└── README.md

---

## 📈 Output
The application takes product and outlet details as input and predicts expected sales.

---

## 👨‍💻 Author
Atharva Sontakke  
Computer Engineering Student  
SPPU University
