import cv2
import numpy as np
import os
import math
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture as GMM
from sklearn.cluster import KMeans
from sklearn.utils import shuffle



#                Pontificia Universidad Javeriana
#               Procesamiento de imágenes y visión
#                            Taller 4.1
#                 Jeimmy Alejandra Cuitiva Mont
#               Laura Alejandra Estupiñan Martínez

#Función recrear imágenes
def recreate_image(centers, labels, rows, cols):

    d = centers.shape[1]
    image_clusters = np.zeros((rows, cols, d))
    label_idx = 0
    for i in range(rows):
        for j in range(cols):
            image_clusters[i][j] = centers[labels[label_idx]]
            label_idx += 1
    return image_clusters

#Función segmentación de colores
def color_segmentation (path_file, method):
    #Lectura dirección imagen
    image = cv2.imread(path_file)
    #Conversión imagen BGR a RGB
    image_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #Definición centros de color
    n_colors = 10
    #Convertir a flotante
    image_RGB = np.array(image_RGB, dtype=np.float64) / 255
    #Transformar imagen a arreglo 2D
    rows, cols, ch = image_RGB.shape
    image_array = np.reshape(image_RGB, (rows * cols, ch))
    #Determinar 10000 pixeles de muestra
    image_array_sample = shuffle(image_array, random_state=0)[:10000]

    #Método kmeans
    if (method) == 'kmeans':
        datos = []
        #Computar segmentación de color para cada valor de n_colors
        for j in range(n_colors):
            #Aumentar valor variable n_colors
            n_colors = j + 1
            #Generar labels para todos los puntos
            model = KMeans(n_clusters=n_colors, random_state=0).fit(image_array_sample)
            labels = model.predict(image_array)
            centers = model.cluster_centers_
            cluster = 0
            #Sumar distancias intra-cluster
            for i in range(len(labels)):
                if labels[i] == 0:
                    cluster = cluster + (math.sqrt(np.power(image_array[i][0]-centers[0][0],2) + np.power(image_array[i][1]-centers[0][1],2)+ np.power(image_array[i][2]-centers[0][2],2)))
                if labels[i] == 1:
                    cluster = cluster + (math.sqrt(np.power(image_array[i][0]-centers[1][0],2) + np.power(image_array[i][1]-centers[1][1],2)+ np.power(image_array[i][2]-centers[1][2],2)))
                if labels[i] == 2:
                    cluster = cluster + (math.sqrt(np.power(image_array[i][0]-centers[2][0],2) + np.power(image_array[i][1]-centers[2][1],2)+ np.power(image_array[i][2]-centers[2][2],2)))
                if labels[i] == 3:
                    cluster = cluster + (math.sqrt(np.power(image_array[i][0]-centers[3][0],2) + np.power(image_array[i][1]-centers[3][1],2)+ np.power(image_array[i][2]-centers[3][2],2)))
                if labels[i] == 4:
                    cluster = cluster + (math.sqrt(np.power(image_array[i][0]-centers[4][0],2) + np.power(image_array[i][1]-centers[4][1],2)+ np.power(image_array[i][2]-centers[4][2],2)))
                if labels[i] == 5:
                    cluster = cluster + (math.sqrt(np.power(image_array[i][0]-centers[5][0],2) + np.power(image_array[i][1]-centers[5][1],2)+ np.power(image_array[i][2]-centers[5][2],2)))
                if labels[i] == 6:
                    cluster = cluster + (math.sqrt(np.power(image_array[i][0]-centers[6][0],2) + np.power(image_array[i][1]-centers[6][1],2)+ np.power(image_array[i][2]-centers[6][2],2)))
                if labels[i] == 7:
                    cluster = cluster + (math.sqrt(np.power(image_array[i][0]-centers[7][0],2) + np.power(image_array[i][1]-centers[7][1],2)+ np.power(image_array[i][2]-centers[7][2],2)))
                if labels[i] == 8:
                    cluster = cluster + (math.sqrt(np.power(image_array[i][0]-centers[8][0],2) + np.power(image_array[i][1]-centers[8][1],2)+ np.power(image_array[i][2]-centers[8][2],2)))
                if labels[i] == 9:
                    cluster = cluster + (math.sqrt(np.power(image_array[i][0]-centers[9][0],2) + np.power(image_array[i][1]-centers[9][1],2)+ np.power(image_array[i][2]-centers[9][2],2)))
            #Crear vector valores intra-cluster
            datos.append(cluster)
            #Desplegar imágenes centros de color
            plt.figure(n_colors + 1)
            plt.clf()
            plt.axis('off')
            plt.title('Quantized image method=kmeans ({} colors)'.format(n_colors))
            plt.imshow(recreate_image(centers, labels, rows, cols))
            plt.imsave('onePic({} colors).jpg'.format(n_colors), recreate_image(centers, labels, rows, cols))
        print(datos)
        #Producir gráfica suma de distancias intra-cluster
        plt.figure(1)
        plt.clf()
        plt.title('Distancia intra-cluster')
        plt.xlabel('N_Colors')
        plt.ylabel('Distancia')
        plt.plot(datos)
        plt.show()

    #Método gmm
    elif (method) == 'gmm':
        datos = []
        #Computar segmentación de color para cada valor de n_colors
        for j in range(n_colors):
            #Aumentar valor variable n_colors
            n_colors = j + 1
            #Generar labels para todos los puntos
            model = GMM(n_components=n_colors).fit(image_array_sample)
            labels = model.predict(image_array)
            centers = model.means_
            cluster = 0
            #Sumar distancias intra-cluster
            for i in range(len(labels)):
                if labels[i] == 0:
                    cluster = cluster + (math.sqrt(np.power(image_array[i][0]-centers[0][0],2) + np.power(image_array[i][1]-centers[0][1],2)+ np.power(image_array[i][2]-centers[0][2],2)))
                if labels[i] == 1:
                    cluster = cluster + (math.sqrt(np.power(image_array[i][0]-centers[1][0],2) + np.power(image_array[i][1]-centers[1][1],2)+ np.power(image_array[i][2]-centers[1][2],2)))
                if labels[i] == 2:
                    cluster = cluster + (math.sqrt(np.power(image_array[i][0]-centers[2][0],2) + np.power(image_array[i][1]-centers[2][1],2)+ np.power(image_array[i][2]-centers[2][2],2)))
                if labels[i] == 3:
                    cluster = cluster + (math.sqrt(np.power(image_array[i][0]-centers[3][0],2) + np.power(image_array[i][1]-centers[3][1],2)+ np.power(image_array[i][2]-centers[3][2],2)))
                if labels[i] == 4:
                    cluster = cluster + (math.sqrt(np.power(image_array[i][0]-centers[4][0],2) + np.power(image_array[i][1]-centers[4][1],2)+ np.power(image_array[i][2]-centers[4][2],2)))
                if labels[i] == 5:
                    cluster = cluster + (math.sqrt(np.power(image_array[i][0]-centers[5][0],2) + np.power(image_array[i][1]-centers[5][1],2)+ np.power(image_array[i][2]-centers[5][2],2)))
                if labels[i] == 6:
                    cluster = cluster + (math.sqrt(np.power(image_array[i][0]-centers[6][0],2) + np.power(image_array[i][1]-centers[6][1],2)+ np.power(image_array[i][2]-centers[6][2],2)))
                if labels[i] == 7:
                    cluster = cluster + (math.sqrt(np.power(image_array[i][0]-centers[7][0],2) + np.power(image_array[i][1]-centers[7][1],2)+ np.power(image_array[i][2]-centers[7][2],2)))
                if labels[i] == 8:
                    cluster = cluster + (math.sqrt(np.power(image_array[i][0]-centers[8][0],2) + np.power(image_array[i][1]-centers[8][1],2)+ np.power(image_array[i][2]-centers[8][2],2)))
                if labels[i] == 9:
                    cluster = cluster + (math.sqrt(np.power(image_array[i][0]-centers[9][0],2) + np.power(image_array[i][1]-centers[9][1],2)+ np.power(image_array[i][2]-centers[9][2],2)))
            datos.append(cluster)
            plt.figure(n_colors + 1)
            plt.clf()
            plt.axis('off')
            plt.title('Quantized image method=kmeans ({} colors)'.format(n_colors))
            plt.imshow(recreate_image(centers, labels, rows, cols))
        #Crear vector valores intra-cluster
        datos.append(cluster)
        #Desplegar imágenes centros de color
        print(datos)
        #Producir gráfica suma de distancias intra-cluster
        plt.figure(1)
        plt.clf()
        plt.title('Distancia intra-cluster')
        plt.xlabel('N_Colors')
        plt.ylabel('Distancia')
        plt.plot(datos)
        plt.show()

    else:
        print('Ingrese un valor valido')


if __name__ == '__main__':
    ###Modificar por el path de la imagen de prueba###
    path = '/Users/lauestupinan/Desktop'
    image_name = 'bandera.png'
    path_file = os.path.join(path, image_name)
    method = input('Seleccione un método: \n 1. kmeans \n 2. gmm \n')
    color_segmentation(path_file, method)


