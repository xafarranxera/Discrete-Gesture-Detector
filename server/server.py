import socket
import time
from random import seed
from random import random
import discretegesturedetector as dgd
import pickle

localIP     = "192.168.196.44"
localPort   = 11000
bufferSize  = 7168
num_gestos = 0
#msgFromServer       = "Hola cliente UDP"
#bytesToSend = str.encode("Mensaje")

gesto = "lolo"

def ProcesandoDatos():
  print("Procesando datos")
  #seed(1)
  aleatorio = random()
  #print("random = " + str(aleatorio))
  global gesto,num_gestos, bytesToSend
  num_gestos += 1
  print(num_gestos)
  if random()<0.5:
    print("Gesto 1 reconocido")
    gesto = "Gesto1"
  else:
    print("No se ha reconocido ningún gesto")
    gesto = "NoGesto"
  print("gesto antes de codificar " + gesto)
  global bytesToSend 
  bytesToSend= str.encode(gesto)
  message = ""
  



#ProcesandoDatos()

# Se crea el datagrama

UDPServerSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)

# Se define dirección IP y puerto

UDPServerSocket.bind((localIP, localPort))

print("Servidor UDP arriba")
 
# Escuchando para mensajes entrantes

model_path = "./discrete_gesture_model.sav"
model = pickle.load(open(model_path, "rb"))

while(True):

    bytesAddressPair = UDPServerSocket.recvfrom(bufferSize)
    message = bytesAddressPair[0]
    address = bytesAddressPair[1]
    clientMsg = "Mensaje del cliente:{}".format(message)
    clientIP  = "IP del cliente:{}".format(address)
    
    print(clientMsg)
    print(clientIP)
    #ProcesandoDatos()
    nframe, detected_gesture = dgd.get_gesture(clientMsg, model)
    
    print("gesto DESPUES de codificar " + detected_gesture)

    # Se manda respuesta al cliente
    # UDPServerSocket.sendto(bytesToSend, address)
    #UDPServerSocket.sendto(bytesToSend, ("127.0.0.1", 5555))
    UDPServerSocket.sendto(bytesToSend, ("192.168.196.177", 11000))
    #UDPServerSocket.sendto(bytesToSend, ("192.168.68.177", 5555))
    

    #
    

