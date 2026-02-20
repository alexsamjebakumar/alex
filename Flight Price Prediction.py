import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("QtAgg")
from PySide6.QtWidgets import (
    QApplication,QWidget,QLabel,QPushButton,QVBoxLayout,QComboBox,QSpinBox,QMessageBox)
from PySide6.QtCore import Qt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

df = pd.read_excel(r"C:\Users\alexs\Downloads\Data_Train.xlsx") 
df.dropna(inplace=True)

df["Journey_Day"] = pd.to_datetime(df["Date_of_Journey"]).dt.day
df["Journey_Month"] = pd.to_datetime(df["Date_of_Journey"]).dt.month
df.drop("Date_of_Journey", axis=1, inplace=True)

df["Dep_hour"] = pd.to_datetime(df["Dep_Time"]).dt.hour
df["Dep_min"] = pd.to_datetime(df["Dep_Time"]).dt.minute
df.drop("Dep_Time", axis=1, inplace=True)

df["Arrival_hour"] = pd.to_datetime(df["Arrival_Time"]).dt.hour
df["Arrival_min"] = pd.to_datetime(df["Arrival_Time"]).dt.minute
df.drop("Arrival_Time", axis=1, inplace=True)

duration = list(df["Duration"])

for i in range(len(duration)):
    if len(duration[i].split()) != 2:
        if "h" in duration[i]:
            duration[i] = duration[i] + " 0m"
        else:
            duration[i] = "0h " + duration[i]

duration_hours = []
duration_mins = []

for i in range(len(duration)):
    duration_hours.append(int(duration[i].split()[0][:-1]))
    duration_mins.append(int(duration[i].split()[1][:-1]))

df["Duration_hours"] = duration_hours
df["Duration_mins"] = duration_mins
df.drop("Duration", axis=1, inplace=True)

df["Total_Stops"] = df["Total_Stops"].replace({
    "non-stop": 0,
    "1 stop": 1,
    "2 stops": 2,
    "3 stops": 3,
    "4 stops": 4
})

if "Route" in df.columns:
    df.drop("Route", axis=1, inplace=True)

df = pd.get_dummies(
    df,
    columns=["Airline", "Source", "Destination", "Additional_Info"],
    drop_first=True
)

X = df.drop("Price", axis=1)
y = df["Price"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = RandomForestRegressor()
model.fit(X_train, y_train)
y_pred=model.predict(X_test)

def show_plot():
    import matplotlib.pyplot as plt

    actual = y_test[:20]
    predicted = y_pred[:20]
    x = range(len(actual))

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.bar(x, actual)
    plt.title("Actual Flight Prices")

    plt.subplot(1, 2, 2)
    plt.bar(x, predicted)
    plt.title("Predicted Flight Prices")

    plt.tight_layout()
    plt.show()       

def predict_price(inputs:dict):
    input_dict=dict.fromkeys(X.columns,0)
    for key,value in inputs.items():
        if key in input_dict:
            input_dict[key]=value
    input_df= pd.DataFrame([input_dict]) 
    return int(model.predict(input_df)[0])

class FlightApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("✈ Flight Price Prediction")
        self.setMinimumSize(700, 700)

        layout = QVBoxLayout()
        title = QLabel("✈ Flight Price Prediction")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("""QLabel{color:white;font-size:26px;font-weight:bold;padding:10px;}""")
        title.setFixedHeight(60)
        
        layout.addWidget(title)

        self.airline = QComboBox()
        self.airline.addItems(["IndiGo", "Air India", "GoAir", "SpiceJet"])

        self.source = QComboBox()
        self.source.addItems(["Chennai", "Delhi", "Bangalore", "Kolkata"])

        self.destination = QComboBox()
        self.destination.addItems(["Cochin", "Delhi", "Bangalore"])

        self.add_info = QComboBox()
        self.add_info.addItems(["No info", "In-flight meal not included"])

        self.stops = QSpinBox()
        self.stops.setRange(0, 4)

        self.j_day = QSpinBox()
        self.j_day.setRange(1, 31)

        self.j_month = QSpinBox()
        self.j_month.setRange(1, 12)

        self.dep_hour = QSpinBox()
        self.dep_hour.setRange(0, 23)

        self.dep_min = QSpinBox()
        self.dep_min.setRange(0, 59)

        self.arr_hour = QSpinBox()
        self.arr_hour.setRange(0, 23)

        self.arr_min = QSpinBox()
        self.arr_min.setRange(0, 59)

        self.dur_h = QSpinBox()
        self.dur_h.setRange(0, 50)

        self.dur_m = QSpinBox()
        self.dur_m.setRange(0, 59)

        self.button = QPushButton("Predict Price ✈️")
        self.button.clicked.connect(self.on_predict)

        self.result = QLabel("₹ ---")
        self.result.setAlignment(Qt.AlignCenter)
        self.result.setStyleSheet("font-size:26px;font-weight:bold;color:#FF6F00;")

        widgets = [
            ("Airline", self.airline),
            ("Source", self.source),
            ("Destination", self.destination),
            ("Additional Info", self.add_info),
            ("Total Stops", self.stops),
            ("Journey Day", self.j_day),
            ("Journey Month", self.j_month),
            ("Dep Hour", self.dep_hour),
            ("Dep Min", self.dep_min),
            ("Arrival Hour", self.arr_hour),
            ("Arrival Min", self.arr_min),
            ("Duration Hour", self.dur_h),
            ("Duration Min", self.dur_m),
        ]

        layout.addWidget(title)
        for label_text, widget in widgets:
            layout.addWidget(QLabel(label_text))
            layout.addWidget(widget)

        layout.addWidget(self.button)
        layout.addWidget(self.result)

        self.setLayout(layout)

    def on_predict(self):
        if self.source.currentText() == self.destination.currentText():
            QMessageBox.warning(self, "Error", "Source and Destination same, so entered valid data")
            return

        data = {
            "Total_Stops": self.stops.value(),
            "Journey_Day": self.j_day.value(),
            "Journey_Month": self.j_month.value(),
            "Dep_hour": self.dep_hour.value(),
            "Dep_min": self.dep_min.value(),
            "Arrival_hour": self.arr_hour.value(),
            "Arrival_min": self.arr_min.value(),
            "Duration_hours": self.dur_h.value(),
            "Duration_mins": self.dur_m.value(),
            "Airline_" + self.airline.currentText(): 1,
            "Source_" + self.source.currentText(): 1,
            "Destination_" + self.destination.currentText(): 1,
            "Additional_Info_" + self.add_info.currentText(): 1
        }
    
        price = predict_price(data)
        self.result.setText(f"₹ {price}")
        show_plot()

app = QApplication(sys.argv)
window = FlightApp()
window.show()
sys.exit(app.exec())
    

