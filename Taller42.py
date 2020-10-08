import cv2
import numpy as np
import sys
import os

#                Pontificia Universidad Javeriana
#               Procesamiento de imágenes y visión
#                            Taller 4.2
#                 Jeimmy Alejandra Cuitiva Mont
#               Laura Alejandra Estupiñan Martínez


# Función para anotar 3 puntos de I1 utilizando el mouse
def puntos(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(I1, (x, y), 2, (255, 255, 255), -1)
        pt.append((x, y))

# Función para anotar 3 puntos de I2 utilizando el mouse
def puntos2(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(I2, (x, y), 2, (255, 255, 255), -1)
        pt1.append((x, y))

if __name__ == '__main__':
    # Path lena - Abrir lena
    path1 = 'C:/Users/JEIMM/OneDrive/Escritorio'
    image_name1 = 'lena.png'
    path_file1 = os.path.join(path1, image_name1)
    I1 = cv2.imread(path_file1)
    pt = []

    # llamar función para seleccionar puntos
    cv2.namedWindow('I1')
    cv2.setMouseCallback('I1', puntos)

    while True:
        cv2.imshow('I1', I1)
        k = cv2.waitKey(1) & 0xFF  # mantener abierta la imagen
        if (k == 27) or (len(pt) == 3):  # si se presiona esc o se seleccionan tres puntos, cerrar la imagen
            break
    cv2.destroyAllWindows()
    print("Puntos lena: ", pt)

    # Path lena_warped - Abrir lena_warped
    image_name2 = 'lena_warped.png'
    path_file2 = os.path.join(path1, image_name2)
    I2 = cv2.imread(path_file2)
    pt1 = []  # variable para guardar las posiciones de los tres puntos

    # llamar función para seleccionar puntos en lena_warped
    cv2.namedWindow('I2')
    cv2.setMouseCallback('I2', puntos2)

    while True:
        cv2.imshow('I2', I2)
        k = cv2.waitKey(1) & 0xFF  # mantener abierta la imagen
        if (k == 27) or (len(pt1) == 3):  # si se presiona esc o se seleccionan tres puntos, cerrar la imagen
            break
    cv2.destroyAllWindows()
    print("Puntos lena_warped: ", pt1)  # variable para guardar las posiciones de los tres puntos

    # sacar la matriz afín con los puntos de lena y lena_warped
    pts1 = np.float32(pt)
    pts2 = np.float32(pt1)
    M_affine = cv2.getAffineTransform(pts1, pts2)
    image_affine = cv2.warpAffine(I1, M_affine, I1.shape[:2])

    cv2.imshow("Image_affine", image_affine)
    cv2.waitKey(0)

    # punto e: Sacar parametros de escala, rotación y traslación
    sx = np.sqrt((M_affine[0, 0] ** 2) + (M_affine[1, 0] ** 2))
    sy = np.sqrt((M_affine[0, 1] ** 2) + (M_affine[1, 1] ** 2))
    theta = -np.arctan(M_affine[1, 0] / M_affine[0, 0])
    tx = (M_affine[0, 2] * np.cos(theta) - M_affine[1, 2] * np.sin(theta)) / sx
    ty = (M_affine[0, 2] * np.sin(theta) + M_affine[1, 2] * np.cos(theta)) / sy

    # translation
    M_t = np.float32([[1, 0, tx], [0, 1, ty]])
    image_translation = cv2.warpAffine(I1, M_t, (I1.shape[1], I1.shape[0]))

    # scaling
    M_s = np.float32([[sx, 0, 0], [0, sy, 0]])
    image_scale = cv2.warpAffine(I1, M_s, (700, 700))

    # rotation
    theta_rad = theta * np.pi / 180
    M_rot = np.float32([[np.cos(theta_rad), -np.sin(theta_rad), 0],
                       [np.sin(theta_rad), np.cos(theta_rad), 0]])
    cx = I1.shape[1] / 2
    cy = I1.shape[0] / 2
    M_rot = cv2.getRotationMatrix2D((cx, cy), theta, 1)
    image_rotation = cv2.warpAffine(I1, M_rot, I1.shape[:2])

    # similarity spbre I1
    M_sim = np.float32([[sx * np.cos(theta_rad), -np.sin(theta_rad), tx],
                        [np.sin(theta_rad), sy * np.cos(theta_rad), ty]])
    image_similarity = cv2.warpAffine(I1, M_sim, I1.shape[:2])

    cv2.imshow("Image_similarity", image_similarity)
    cv2.waitKey(0)

    # Método cálculo error
    def ECM(image_similarity, I2):
        M, N, C = image_similarity.shape
        K = 0
        for i in range(1, M):
            for j in range(1, N):
                for l in range(1, C):
                    K += np.power(abs(image_similarity[i, j] - I2[i, j]), 2)
        N = ((1 / (M * N* C)) * K)
        V = np.sqrt(N)
        return V
    error = ECM(image_similarity, I2)
    print("Error: ", error)