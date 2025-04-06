## 📩 SMS Spam Classifier

An intelligent machine learning-based SMS Spam Classifier that detects whether a given message is **Spam** or **Ham (Not Spam)** using natural language processing and classification algorithms.

### 🚀 Features
- Classifies SMS as Spam or Ham.
- Text preprocessing and vectorization.
- Trained using machine learning models.
- Clean and interactive user interface (Streamlit-based).

### 🧠 Technologies Used
- Python
- Pandas
- Scikit-learn
- NLTK
- Streamlit

### 📁 Dataset
The model is trained on the [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset), which contains 5,574 SMS messages labeled as ham or spam.

### ⚙️ How to Run

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/sms-spam-classifier.git
   cd sms-spam-classifier
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

4. **Use the Web UI** to enter your SMS and get predictions.

### 📝 Model Training
- Data cleaned and preprocessed (punctuation removal, lowercasing, stopword removal).
- TF-IDF Vectorizer used to convert text to numerical features.
- Trained using a **Multinomial Naive Bayes** classifier.
- Accuracy: ~98%

### 📌 Future Enhancements
- Add support for multilingual messages.
- Deploy as a browser extension or mobile app.
- Improve accuracy with deep learning models.