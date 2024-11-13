import socket

HOST = '0.0.0.0'  
PORT = 3002      
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as ss:
    ss.bind((HOST, PORT))
    ss.listen()
    print("Port : ", PORT)

    conn, addr = ss.accept()
    with conn:
        print('Connected = ', addr)
        while True:
            data = conn.recv(1024)
            if not data:
                break  
            message = data.decode()
            print("Client = ", message)
            
            if message.lower() == 'bye':
                print("Bye....!")
                break  
            conn.sendall(data)  
