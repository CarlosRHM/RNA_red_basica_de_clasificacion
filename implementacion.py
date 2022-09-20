import mnist_loader
import network
import pickle

training_data, validation_data , test_data = mnist_loader.load_data_wrapper()
training_data = list(training_data)
test_data = list(test_data)

net=network.Network([784,30,10])
#(datos de entrenamiento, epocas, mini_batch_size, eta, gamma:(0,1), test_data)
net.SGD_inercia( training_data, 30, 10, 1.5, 0.9, test_data=test_data)

archivo = open("red_prueba1.pkl",'wb')
pickle.dump(net,archivo)
archivo.close()
exit()

#leer el archivo
archivo_lectura = open("red_prueba.pkl",'rb')
net = pickle.load(archivo_lectura)
archivo_lectura.close()
net.SGD( training_data, 10, 50, 0.5, test_data=test_data)
archivo = open("red_prueba.pkl",'wb')
pickle.dump(net,archivo)
archivo.close()
exit()
