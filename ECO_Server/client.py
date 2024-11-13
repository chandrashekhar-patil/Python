import socket

HOST = '192.168.6.13' 
PORT = 3002           

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
    client_socket.connect((HOST, PORT))
    
    while True:
        message = input("Enter a message : ")
        
        if message.lower() == 'bye':
            print("Bye....!")
            client_socket.sendall(message.encode())
            break
        
        client_socket.sendall(message.encode())
        data = client_socket.recv(1024)
        print("Server = ", data.decode())