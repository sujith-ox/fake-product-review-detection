### Fake Product Review Detection

A machine learning project that detects fake product reviews by classifying them as "authentic" (OR) or "deceptive" (CG). The project utilizes a natural language processing (NLP) pipeline to transform text data into numerical features, which are then used to train and evaluate multiple classification models.

-----

### ✨ Features

  * **Data Preprocessing:** Cleans and prepares text data by removing punctuation and stopwords.
  * **Feature Engineering:** Converts text into numerical features using **Count Vectorization** and **TF-IDF Transformation**.
  * **Model Comparison:** Trains and evaluates three different classification models:
      * **Random Forest Classifier**
      * **Support Vector Classifier (SVC)**
      * **Logistic Regression**
  * **Performance Evaluation:** Reports the accuracy of each model, with Logistic Regression achieving the highest score in the provided notebook.

-----

### 💻 Tech Stack

  * **Jupyter Notebook:** The primary environment for development and demonstration.
  * **Python:** The core programming language.
  * **Pandas & NumPy:** Used for data manipulation and numerical operations.
  * **NLTK:** A powerful library for text preprocessing, specifically for handling stopwords.
  * **Scikit-learn:** Provides tools for model training, evaluation, and building the machine learning pipeline.

-----

### ⚙️ Getting Started

Follow these steps to set up the project and run the notebook.

#### **1. Clone the repository**

```bash
git clone https://github.com/sujith-ox/fake-product-review-detection.git
cd fake-product-review-detection
```

#### **2. Install Dependencies**

It is recommended to use a virtual environment.

```bash
pip install -r requirements.txt
```

If you don't have a `requirements.txt` file, you can install the required libraries manually:

```bash
pip install pandas numpy scikit-learn nltk
```

You will also need to download the NLTK stopwords corpus:

```python
import nltk
nltk.download('stopwords')
```

#### **3. Run the Notebook**

Start the Jupyter Notebook server in your terminal.

```bash
jupyter notebook
```

A browser window will open. Navigate to `fake-product-review-detection.ipynb` to view and run the code.

-----

### 📄 License

This project is licensed under the **MIT License**.
