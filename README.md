🧠 Artificial Neural Network Project: Customer Churn Prediction
🚀 Overview
This project uses an Artificial Neural Network (ANN) built with TensorFlow and Keras to predict customer churn based on structured input features. It demonstrates a full pipeline—from data preprocessing to model deployment.

📂 Project Structure
.
├── data/
│   └── _data.csv
├── models/
│   └── ann_model.h5
├── notebooks/
│   └── exploration.ipynb
├── src/
│ 
├── logs/

└── README.md
📊 Features Used
Geography

Gender

Age

Tenure

Balance

Number of Products

Has Credit Card

Is Active Member

Estimated Salary

🛠️ Tech Stack
Python 3.x

TensorFlow / Keras

Pandas / NumPy / scikit-learn

Streamlit (optional UI)

TensorBoard for visualization

📈 Model Highlights
Architecture: Input → Dense(6, relu) → Dense(6, relu) → Dense(1, sigmoid)

Loss: Binary Crossentropy

Optimizer: Adam

Evaluation: Accuracy, Confusion Matrix

🔍 TensorBoard Logs
TensorBoard was configured to visualize training metrics, including loss curves and accuracy over epochs. Scalar data was properly logged using:

python
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
🧪 How to Run
bash
git clone https://github.com/sparshi15/DEEP-LEARNING-PROJECT-USING-ANN
cd DEEP-LEARNING-PROJECT-USING-ANN
pip install -r requirements.txt
python src/ann_pipeline.py
📌 Notes
Ensure Python version compatibility (e.g., Python 3.8 recommended for TensorFlow 2.x).

Conda environment used for reproducibility (environment.yml included).

Streamlit dashboard available [if applicable].

🌟 Results
Achieved 86.20% accuracy on test set


🧠 Future Enhancements
Hyperparameter tunings with GridSearchCV

Model versioning with MLflow

Deployment as REST API

