from __future__ import print_function
import socket
import tqdm
import os
import sys
import csv
import random
import string
import time
import json
import requests
import netifaces as ni
from sys import platform


def check_ip():
    if platform == "linux" or platform == "linux2":
        # port = 'eth0'
        port = 'wlp0s20f3'
    elif platform == "win32":
        port = ni.interfaces()[2]  # [1] VMNet, [4] Wi-Fi [6] ethernet?
        # in my windows the answer is 2, other is 6
    else:
        print("No port found.")
        exit()

    # print(len(ni.interfaces()))
    # print(port)

    # ip = ni.ifaddresses('eth0')[ni.AF_INET][0]['addr']
    ip_address = ni.ifaddresses(port)[ni.AF_INET][0]['addr']
    print(ip_address)  # should print "192.168.100.37"

    # exit()
    # print(s.getsockname()[0])

    if ip_address == "127.0.0.1":
        print("No network available, your localhost is " + ip_address)
        result = True
    else:
        print("Network available with the IP address: " + ip_address)
        result = False

    return [ip_address, result]


def string_generator(str_size, allowed_chars):
    return ''.join(random.choice(allowed_chars) for x in range(str_size))


def send_pc_rasp(ip, file_use):
    filename = ""
    if file_use == "random":
        chars = string.ascii_letters + string.punctuation
        size = 12
        random_string = string_generator(size, chars)
        print('Random String of length 12 =', random_string)
        crop_string = '0; -1; 0; -1;'
        filename = "send_data.csv"

        if not os.path.exists(filename):
            open(filename, "w").close()

        with open(r'M:\\Workspace\\python\\uService_port\\' + filename, 'w') as file_check:
            writer = csv.writer(file_check)
            writer.writerow(random_string)
            writer.writerow(crop_string)
    elif file_use == "json":
        filename = "example_in.json"

    print(filename)
    print(ip)
    print('send')
    socket_send(filename, ip)
    time.sleep(1)


def send_userv_front(use_json: bool):
    print(os.getcwd() + "/example_out.json")
    data_eval = {
        "equipment_id": "64909fc47e30d0e3725d9a9a",
        "origin_field": "predictive",
        "evaluation_criticality": True,
        "properties": [
            {
                "property": "InletPressure",
                # "property": "Temperature.InletTemperature",
                "value": 80,
                "current_data": [
                    {"timestamp": "2023-07-12T01:10:17.965Z", "value": 90},
                    {"timestamp": "2023-07-12T01:11:18.965Z", "value": 91},
                ],
                "prevision_data": [
                    {"timestamp": "2023-07-17T01:11:18.965Z", "value": 100},
                    {"timestamp": "2023-07-17T01:11:19.965Z", "value": 101},
                ],
            },
            {
                "property": "Pressure.InletPressure",
                "value": 8,
                "current_data": [
                    {"timestamp": "2023-07-12T01:10:17.965Z", "value": 9},
                    {"timestamp": "2023-07-12T01:11:18.965Z", "value": 9.5},
                ],
                "prevision_data": [
                    {"timestamp": "2023-07-17T01:11:18.965Z", "value": 10},
                    {"timestamp": "2023-07-17T01:11:19.965Z", "value": 10.5},
                ],
            },
        ],
    }
    data_json = open(os.getcwd() + "/example_out.json", mode="r").read()
    # data = open("~/Workspace/python/example_out.json", mode="r").read()

    # print(data.read())
    # exit()
    if use_json:
        data = data_json
    else:
        data = data_eval

    # Convert the JSON data to a string
    json_data = json.dumps(eval(data))

    # Define the IP address and port of the server
    ip_address = "172.31.111.103"
    port = 447

    # Define the URL to send the POST request to
    # url = f"http://{ip_address}:{port}/api/predictive-event"
    url = "http://{}:{}/api/predictive-event".format(ip_address, port)

    print("Trying to send to front-end:")
    try:
        # Send the JSON data to the server
        response = requests.post(url, data=json_data, headers={'Content-Type': 'application/json'})

        print(response.status_code)
        # Check the response status
        if response.status_code == 201:
            print("JSON data sent successfully!")
        else:
            print("Error when sending JSON data:", response.text)
    except Exception as e:
        print("Error in connection: ", e)


def send_rasp_pc(given_host):
    # cam.crop_frame(1, 100, 1, 200)
    # cam.crop_frame(0, -1, 0, -1)
    time.sleep(1)

    filename = 'sample_0.jpeg'
    socket_send(filename, given_host)

    time.sleep(10)
    filename = 'sample_0.json'
    socket_send(filename, given_host)


def socket_send(file_to_send, selected_host):
    print('Python ver:' + str(sys.version_info.major) + '.' + str(sys.version_info.minor))
    # print(sys.version_info)
    if sys.version_info.major == 3 and sys.version_info.minor >= 8:
        # my_check = True  # Original version
        print('Good version')
    else:
        # my_check = False  # Changed version
        print('Different version, maybe fail')

    separator = "<SEPARATOR>"
    buffer_size = 4096  # send 4096 bytes each time step

    if selected_host == "0":
        # host = "192.168.1.24"
        host = "192.168.56.101"
        # host = "192.168.56.1"
    else:
        host = selected_host

    port = 5001

    dirname = os.path.dirname(__file__)
    filepath = os.path.join(dirname, file_to_send)
    print(filepath)
    filesize = os.path.getsize(filepath)

    s = socket.socket()
    print("[+] Connecting to {}:{}".format(host, port))
    s.connect((host, port))
    print("[+] Connected.")

    # if my_check:
    #    s.send("{filename}{SEPARATOR}{filesize}".encode())
    # else:
    str_send = "{}{}{}".format(file_to_send, separator, filesize)
    s.send(str_send.encode())

    progress = tqdm.tqdm(range(
        filesize), "Sending {}".format(file_to_send), unit="B", unit_scale=True, unit_divisor=1024)
    with open(filepath, "rb") as f:
        while True:
            bytes_read = f.read(buffer_size)
            if not bytes_read:
                break

            s.sendall(bytes_read)
            progress.update(len(bytes_read))

    s.close()


def socket_receive(selected_host):
    print('Python ver: ' + str(sys.version_info.major) + '.' + str(sys.version_info.minor))
    # print(sys.version_info)
    if sys.version_info.major == 3 and sys.version_info.minor >= 8:
        # my_check = True  # Original version
        print('Good version')
    else:
        # my_check = False  # Changed version
        print('Different version, maybe fail')

    if selected_host == "0":
        # server_host = "192.168.1.24"
        # server_host = "192.168.56.101"
        server_host = "192.168.56.1"
    else:
        server_host = selected_host

    server_port = 5001

    buffer_size = 4096
    separator = "<SEPARATOR>"

    s = socket.socket()
    s.bind((server_host, server_port))
    s.listen(5)

    # if my_check:  # python 3.8
    #    print("[*] Listening as {SERVER_HOST}:{SERVER_PORT}")
    # else:       # python 2.7
    print("[*] Listening as {}:{}".format(server_host, server_port))

    client_socket, address = s.accept()
    print("[+] {} is connected".format(address))

    received = client_socket.recv(buffer_size).decode()
    filename, filesize = received.split(separator)
    filename = os.path.basename(filename)
    filesize = int(filesize)

    progress = tqdm.tqdm(range(filesize), "Receiving {}".format(filename), unit="B", unit_scale=True, unit_divisor=1024)
    with open(filename, "wb") as f:
        while True:
            bytes_read = client_socket.recv(buffer_size)
            if not bytes_read:
                break
            f.write(bytes_read)
            progress.update(len(bytes_read))

    client_socket.close()
    s.close()

    if server_host == "192.168.56.1":  # "192.168.1.24":
        print("[+] the file send can be processed here.")
        send_userv_front(True)
        # send_rasp_pc(address[0])
