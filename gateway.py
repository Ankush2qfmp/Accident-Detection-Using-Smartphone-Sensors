import serial
import json
import requests
import time

# Configure serial port (update with your port and baud rate)
ser = serial.Serial('/dev/ttyUSB0', 9600, timeout=1)

# Flask server endpoint
url = 'http://<YOUR_SERVER_IP>:5000/accident'

def send_data_to_server(data):
    try:
        response = requests.post(url, json=data)
        if response.status_code == 200:
            print("Data sent successfully:", response.json())
        else:
            print("Failed to send data. Status code:", response.status_code)
    except Exception as e:
        print("Error sending data:", e)

while True:
    try:
        line = ser.readline().decode('utf-8').strip()
        if line:
            print("Received from sensor:", line)
            # Assume the sensor sends data in JSON format
            try:
                data = json.loads(line)
                send_data_to_server(data)
            except json.JSONDecodeError:
                print("Invalid JSON received:", line)
        time.sleep(1)
    except KeyboardInterrupt:
        print("Exiting.")
        break
