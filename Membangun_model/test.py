import requests
import json

url = "http://127.0.0.1:5001/invocations"

# Ganti sesuai kolom kamu setelah preprocessing
data = {
    "dataframe_records": [
        {
            "Pregnancies": 2,
            "Glucose": 130,
            "BloodPressure": 70,
            "SkinThickness": 25,
            "Insulin": 100,
            "BMI": 24.5,
            "DiabetesPedigreeFunction": 0.45,
            "Age": 28,
            "BMI_Underweight": 0,
            "BMI_Normal": 1,
            "BMI_Overweight": 0,
            "BMI_Obese": 0
        }
    ]
}

response = requests.post(
    url,
    headers={"Content-Type": "application/json"},
    data=json.dumps(data)
)

print("Response status:", response.status_code)
print("Prediction result:", response.text)
