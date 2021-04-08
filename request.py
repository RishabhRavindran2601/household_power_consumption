import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'Global_reactive_power':0.418, 'Voltage':234.84, 'Global_intensity':18.400, 'Sub_metering_1':0.000, 'Sub_metering_2':1.000,'Sub_metering_3':17.0})

print(r.json())

