import time
import socket_tools as ws

# As Client
# ws.send_pc_rasp("0", "json")
# exit()

# As Server
while True:
	ws.socket_receive(ws.check_ip()[0])  # '0'
	print('System wait some time before restart websocket')
	time.sleep(5)
