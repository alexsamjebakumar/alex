import sys
import pandas as pd
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout, QMessageBox
)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

# ----------------- Load Data & Train Model -----------------
df = pd.read_csv(r"C:\Users\alexs\Downloads\lung cancer.csv")
X = df.drop("RESULT", axis=1)
y = df["RESULT"]
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)

# Evaluate model
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# ----------------- PySide6 GUI -----------------
class LungCancerApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Lung Cancer Prediction")
        self.layout = QVBoxLayout()
        self.inputs = {}
        for col in X.columns:
            h_layout = QHBoxLayout()
            label = QLabel(f"{col}:")
            line_edit = QLineEdit()
            line_edit.setPlaceholderText("Enter value")
            self.inputs[col] = line_edit
            h_layout.addWidget(label)
            h_layout.addWidget(line_edit)
            self.layout.addLayout(h_layout)
        self.predict_btn = QPushButton("Predict")
        self.predict_btn.clicked.connect(self.predict_cancer)
        self.layout.addWidget(self.predict_btn)
        self.result_label = QLabel("")
        self.layout.addWidget(self.result_label)
        
        # Add metrics
        metrics_text = f"Model Accuracy: {accuracy:.2f} | Precision: {precision:.2f} | Recall: {recall:.2f}"
        self.metrics_label = QLabel(metrics_text)
        self.layout.addWidget(self.metrics_label)
        
        self.setLayout(self.layout)
        self.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                padding: 10px;
                border-radius: 5px;
            }
            QLabel {
                font-size: 14px;
                margin: 5px;
            }
        """)

    def predict_cancer(self):
        try:
            user_data = [float(self.inputs[col].text()) for col in X.columns]
            user_array = np.array(user_data).reshape(1, -1)
            prediction = model.predict(user_array)[0]
            prob = model.predict_proba(user_array)[0][1]
            if prediction == 1:
                result_text = f"Lung cancer detected!\nProbability: {prob:.2f}"
            else:
                result_text = f"No lung cancer.\nProbability: {prob:.2f}"
            self.result_label.setText(result_text)
        except ValueError:
            QMessageBox.warning(self, "Input Error", "Please enter valid numeric values for all fields.")

# ----------------- Run App -----------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LungCancerApp()
    window.show()
    sys.exit(app.exec())