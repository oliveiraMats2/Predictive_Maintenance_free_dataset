import requests
import json

# Define the minimum JSON for a notification
data1 = {
   "equipment_id": "64909fc47e30d0e3725d9a9a",
   "origin_field": "predictive",
   "evaluation_criticality": False,
   "properties": [
      {
         "property": "Temperature.InletTemperature",
         "value": 80,
         "current_data": [
            {
               "timestamp": "2023-07-12T01:10:17.965Z",
               "value": 90
            },
            {
               "timestamp": "2023-07-12T01:11:18.965Z",
               "value": 91
            }
         ],
         "prevision_data": [
            {
               "timestamp": "2023-07-17T01:11:18.965Z",
               "value": 100
            },
            {
               "timestamp": "2023-07-17T01:11:19.965Z",
               "value": 101
            }
         ],
         "description": []
      },
      {
         "property": "Pressure.InletPressure",
         "value": 8,
         "current_data": [
            {
               "timestamp": "2023-07-12T01:10:17.965Z",
               "value": 9
            },
            {
               "timestamp": "2023-07-12T01:11:18.965Z",
               "value": 9.5
            }
         ],
         "prevision_data": [
            {
               "timestamp": "2023-07-17T01:11:18.965Z",
               "value": 10
            },
            {
               "timestamp": "2023-07-17T01:11:19.965Z",
               "value": 10.5
            }
         ]
      }
   ]
}
print(data1)

data = {
   "equipment_id": "64909fc47e30d0e3725d9a9a",
   "origin_field": "predictive",
   "evaluation_criticality": True,
   "properties": [
      {
         "property": "Temperature.InletTemperature",
         "value": 80,
         "current_data": [
            {
               "timestamp": "2023-07-12T01:10:17.965Z",
               "value": 90
            },
            {
               "timestamp": "2023-07-12T01:11:18.965Z",
               "value": 91
            }
         ],
         "prevision_data": [
            {
               "timestamp": "2023-07-17T01:11:18.965Z",
               "value": 100
            },
            {
               "timestamp": "2023-07-17T01:11:19.965Z",
               "value": 101
            }
         ]
      },
      {
         "property": "Pressure.InletPressure",
         "value": 8,
         "current_data": [
            {
               "timestamp": "2023-07-12T01:10:17.965Z",
               "value": 9
            },
            {
               "timestamp": "2023-07-12T01:11:18.965Z",
               "value": 9.5
            }
         ],
         "prevision_data": [
            {
               "timestamp": "2023-07-17T01:11:18.965Z",
               "value": 10
            },
            {
               "timestamp": "2023-07-17T01:11:19.965Z",
               "value": 10.5
            }
         ]
      }
   ]
}

# Convert the JSON data to a string
json_data = json.dumps(data)

# Define the IP address and port of the server
ip_address = "172.31.111.103"
port = 447

# Define the URL to send the POST request to
url = f"http://{ip_address}:{port}/api/predictive-event"

# Send the JSON data to the server
response = requests.post(url, data=json_data, headers={'Content-Type': 'application/json'})

# Check the response status
if response.status_code == 201:
    print("JSON data sent successfully!")
else:
    print("Error sending JSON data:", response.text)
