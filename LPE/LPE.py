from ast import Is
from operator import itruediv
import numpy as np
from matplotlib import pyplot as plt
from scipy import sparse as sp
import cv2

# Booléen d'Affichage
affichage = False


# Lecture de l'image
I = cv2.imread('smarties.png',0)

if (affichage):
    plt.figure()
    plt.subplot(221)
    plt.imshow(I,'gray')

# Seuillage de l'image
ret,thresh1 = cv2.threshold(I,248,255,cv2.THRESH_BINARY)
# Inversion des couleurs
thresh1 = 255 - thresh1

if (affichage):
    plt.subplot(222)
    plt.imshow(thresh1,'gray')

# Opérations Morphologiques --> pour déterminer les marqueurs
S = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(40,25))
IMorph = cv2.morphologyEx(thresh1, cv2.MORPH_ERODE, S)

# IMorph = 255 - (thresh1 - IMorph)
if (affichage):
    plt.subplot(223)
    plt.imshow(IMorph,'gray')
#cv2.imwrite('Morph.png',IMorph)

ret,labels = cv2.connectedComponents(IMorph)

if (affichage):
    plt.subplot(224)
    plt.imshow(labels)

#cv2.imwrite('Labels.png',labels)


# Partie : Obtention de la carte des distances
ITrans = cv2.distanceTransform(thresh1,cv2.DIST_L2,5)

ITrans = ITrans / np.amax(ITrans) * 255   # Normalisation de ITrans

ITrans = np.uint8(ITrans)   # Conversion en entier

ITrans = 255 - ITrans   # On prend le négatif pour l'affichage 

thresh1 = thresh1/255   # Création du masque

ITrans = cv2.multiply(np.uint8(thresh1),ITrans) # Application du masque

if (affichage):
    plt.figure()
    plt.imshow(ITrans,'gray')

# Partie algorithmique : 










if (affichage):
    plt.show()

