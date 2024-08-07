import socket

# Create a socket object
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
host = '0.0.0.0'  # Bind to all available interfaces
port = 5005

# Bind the socket to a public host, and a well-known port
server_socket.bind((host, port))
server_socket.listen(5)

print("Server listening on port", port)

while True:
    client_socket, addr = server_socket.accept()
    print('Got connection from', addr)
    message = client_socket.recv(1024).decode('utf-8')
    print('Message from client:', message)
    client_socket.send('Thank you for connecting, Sruthi'.encode('utf-8'))
    client_socket.close()