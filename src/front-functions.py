import requests
import json

# Define the minimum JSON for a notification
data = {
    "equipment_id": "648a15197e30d0e3725d9a6b",
    "origin_field": "predictive",
    "properties": [
        {
            "property": "Temperature.InletTemperature",
            "value": 100
        },
        {
            "property": "Temperature.OutletTemperature",
            "value": 10
        },
        {
            "property": "Pressure.InletPressure",
            "value": 1
        }
    ]
}

# Convert the JSON data to a string
json_data = json.dumps(data)

# Define the IP address and port of the server
ip_address = "172.31.111.103"
port = 447

# Define the URL to send the POST request to
# url = f"http://{ip_address}:{port}/api/prev_maintenance/api/predictive-event"
url = f"http://{ip_address}:{port}/api/predictive-event"

# Send the JSON data to the server
response = requests.post(url, data=json_data, headers={'Content-Type': 'application/json'})

# Check the response status
if response.status_code == 201:
    print("JSON data sent successfully!")
else:
    print("Error sending JSON data:", response.text)
