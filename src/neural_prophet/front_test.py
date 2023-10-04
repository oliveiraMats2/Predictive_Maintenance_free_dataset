import requests
import json


def send_test():
    data = open("src/neural_prophet/json_data_future_EXAMPLE.json", mode="r").read()

    json_data = json.dumps(eval(json.loads(data)))

    # Define the IP address and port of the server
    ip_address = "172.31.111.103"
    port = 447

    # Define the URL to send the POST request to
    # url = f"http://{ip_address}:{port}/api/predictive-event"
    url = "http://{}:{}/api/predictive-event".format(ip_address, port)

    # Send the JSON data to the server
    response = requests.post(
        url, data=json_data, headers={"Content-Type": "application/json"}
    )

    print(response.status_code)
    # Check the response status
    if response.status_code == 201:
        print("JSON data sent to front-end successfully!")
    else:
        print("Error sending JSON data:", response.text)


if __name__ == '__main__':
    print("Local running")
    send_test()