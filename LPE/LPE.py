import numpy as np
from matplotlib import pyplot as plt
from scipy import sparse as sp
import cv2

# Booléen d'Affichage
affichage = True

# Lecture de l'image
I = cv2.imread('smarties.png',0)

if (affichage):
    plt.figure()
    plt.subplot(221)
    plt.imshow(I,'gray')
    plt.title('Image de base')

# Seuillage de l'image
ret,thresh1 = cv2.threshold(I,248,255,cv2.THRESH_BINARY)
# Inversion des couleurs
thresh1 = 255 - thresh1

if (affichage):
    plt.subplot(222)
    plt.imshow(thresh1,'gray')
    plt.title('Image seuillée')

# Opérations Morphologiques --> pour déterminer les marqueurs
S = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(40,25))
IMorph = cv2.morphologyEx(thresh1, cv2.MORPH_ERODE, S)

# IMorph = 255 - (thresh1 - IMorph)
if (affichage):
    plt.subplot(223)
    plt.imshow(IMorph,'gray')
    plt.title('Image érodée')
#cv2.imwrite('Morph.png',IMorph)

ret,labels = cv2.connectedComponents(IMorph)

if (affichage):
    plt.subplot(224)
    plt.imshow(labels)
    plt.title('Marqueurs pour LPE')

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
    plt.title('Carte des distances')

##########################
# Partie algorithmique : #
##########################

# Initialisation de la FAH
FAHX=[[] for i in range(256)]
FAHY=[[] for i in range(256)]

(h,w) = np.shape(ITrans)           

# Attribution de la priorité pour chaque pixel en fonction du label que possède le pixel (forcément != 0)
for x in range(h):
    for y in range(w):
        numpixel = ITrans[x][y]         
        if labels[x][y] != 0 :
            FAHX[numpixel].append(x)
            FAHY[numpixel].append(y)

# Traitement LPE 
for i in range (256):               #On parcourt la FAH
    while (len(FAHX[i]) != 0) :  # tant qu'elle n'est pas vide
        for j in range(len(FAHX[i])) :
            # Extraction du jeton x et y
            x,y = FAHX[i].pop(-1),FAHY[i].pop(-1)
            ListeX,ListeY = [],[]
            # Ajout des voisins n'ayant pas de labels
            # Voisin de gauche (x-1 > 0 pour appartenir à l'image)
            if x-1 > 0 :                
                if(labels[x-1][y]==0):  # Test s'il n'a pas de label
                    ListeX.append(x-1)
                    ListeY.append(y)
            # Voisin de droite (...)
            if x+1 < h :
                if(labels[x+1][y]==0):
                    ListeX.append(x+1)
                    ListeY.append(y)
            # Voisin du Haut
            if y-1 > 0 :
                if(labels[x][y-1]==0):
                    ListeX.append(x)
                    ListeY.append(y-1)
            # Voisin du Bas
            if y+1 < w: 
                if labels[x][y+1]==0:
                    ListeX.append(x)
                    ListeY.append(y+1)

            # Pour traiter les pixels non traités dans la FAH
            m = len(ListeX)
            if( m != 0):
                for k in range(m):
                    labels[ListeX[k]][ListeY[k]] = labels[x][y]
                    numpixel = ITrans[ListeX[k]][ListeY[k]]
                    if numpixel < i :
                        numpixel = i
                    FAHX[numpixel].append(ListeX[k])
                    FAHY[numpixel].append(ListeY[k])

if (affichage):
    plt.figure()
    plt.imshow(labels)
    plt.title("LPE")

if (affichage):
    plt.show()
