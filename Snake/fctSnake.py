# Projet transversal : Robot cartographe

# Détection d'obstacle, et différentiation d'objet spécifique (plot)
# Détection de contours actifs (Snakes)

#Authors : Alice Malosse

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse as sp
import time 

def fctSnake(I,alpha,beta,gamma) :


    th, I = cv2.threshold(I, 0, 255, cv2.THRESH_OTSU)
    Lx, Ly = np.shape(I)

    ###Creation du snake###
    K = 300                  #Nb pts Snake
    deltaSnake = int(2*(Lx+Ly) / K)
    c = np.zeros((K, 1, 2))

    #Forme du snake : rectangle (bordure de l'image)
    for i in range (K) :
        if i*deltaSnake < Ly :
            point = [i*deltaSnake, 5]
        elif (i*deltaSnake >= Ly) and (i*deltaSnake < Ly+Lx) :
            point = [Ly-6, i*deltaSnake-Ly]
        elif (i*deltaSnake >=Ly+Lx) and (i*deltaSnake < 2*Ly+Lx) :
            point = [Ly-(i*deltaSnake-Ly-Lx) , Lx-6]
        else :
            point = [5, Lx-(i*deltaSnake-2*Ly-Lx)]
        c[i,:,:] = point
    c = c.astype(int)
    snakeX = c[:,:,0]
    snakeY = c[:,:,1]

    
    start = time.time()
    #Contour = cv2.drawContours(image = cv2.cvtColor(I, cv2.COLOR_GRAY2BGR), contours = c, contourIdx = -1, color = (0,255,0), thickness = 8)
    #cv2.imwrite("im_contour.png", Contour)
    ### Parametres ###

    ###Creation de D2, D4, D et A###
    Id = np.identity(K)
    D2 = np.roll(Id, -1, axis=1) + Id*(-2) + np.roll(Id,1, axis=1)
    D4 = (np.roll(Id, -1, axis=1) + np.roll(Id,1, axis=1))*-4 + (np.roll(Id, -2, axis=1) + np.roll(Id,2, axis=1)) + Id*(6)
    D = alpha*D2 - beta*D4
    A = np.linalg.inv(Id - D)

    # Le Gradient
    [Gx,Gy] = np.gradient(I)
    NormeGrad = np.square(Gx)+np.square(Gy)

    # Gradient de la norme 
    [GGx,GGy] = np.gradient(NormeGrad)

    # Algo ITERATIF
    limite = 20000
    iteration = 0
    nbfigure = 1

    Energie = list()

    Xn = snakeX
    Yn = snakeY

    plt.plot(Xn,Yn)
    while iteration < limite:
        # itération du SNAKE
        Xn1 = np.dot(A, Xn + gamma*GGx[Yn.astype(int),Xn.astype(int)] )
        Yn1 = np.dot(A, Yn + gamma*GGy[Yn.astype(int),Xn.astype(int)] )     
        Xn = Xn1
        Yn = Yn1   
        #print(Xn)
        iteration += 1


    c = np.zeros((K,1,2))
    #print(c.shape)
    c[:,:,0] = Xn1.reshape((K,1))
    c[:,:,1] = Yn1.reshape((K,1))
    contour_list = []
    contour_list.append(c.astype(int))
    end = time.time()
    ###Affichage###
    

    FinalIm = cv2.drawContours(image = cv2.cvtColor(I, cv2.COLOR_GRAY2BGR), contours = contour_list, contourIdx = -1, color = (0,255,0), thickness = 4)
    cv2.imwrite("im_final.png", FinalIm)
    print(end-start)
    return contour_list
