# Projet transversal : Robot cartographe

# Détection d'obstacle, et différentiation d'objet spécifique (plot)
# Programme principale du traitement d'image

#Authors : Alice Malosse

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse as sp

#from Hough import fctHough
from fctSnake import fctSnake

#Initialisation nécessaire
    #boolean plot détecter

#IF no plot
    #Snake + Hough
    #MAJ plot detecté

#ELSE 
    #Snake
    #Reccupération du sommet
    #Comparaison milieu de l'image
    #Plot en position == True

### Test Snake ###
IMAGE = cv2.imread('im10.png', 0)

alpha = 1
beta = 10000
gamma = 0.001

Snake = fctSnake(IMAGE,alpha,beta,gamma)

### Test Hough ###
#fctHough(IMAGE, Snake)