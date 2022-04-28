import numpy as np
import cv2
import matplotlib.pyplot as plt

I=cv2.imread('smarties.png', 0)     #On importe l'imagevet on la visualise
plt.figure()
plt.subplot(2,2,1)
plt.imshow(I,cmap='gray')
plt.title("Image originale")



RET,thresh1 =  cv2.threshold(I,245,255,cv2.THRESH_BINARY_INV)        #On seuille l'image en séparant juste les smarties du fond
plt.subplot(2,2,2)
plt.imshow(thresh1,cmap='gray')
plt.title("Seuillage")



S = cv2.getStructuringElement(cv2.MORPH_RECT,(35,15))           #On créé un élément structurant rectangulaire
I_ERODE = cv2.erode(thresh1,S, iterations = 1)                       #On erode l'image seuillée avec cet élément strucurant pour obtenir les marqueurs
plt.subplot(2,2,3)
plt.imshow(I_ERODE,cmap='gray')
plt.title("Erosion")


ret1, labels=cv2.connectedComponents(I_ERODE)
ret2, labels2=cv2.connectedComponents(255-thresh1)



Im = labels2+labels             #On obtient l'image avec tt les marqueurs
plt.subplot(2,2,4)
plt.imshow(Im)
plt.title("Marqueurs initiaux utilisés pour la LPE ")


dist_transform = cv2.distanceTransform(thresh1,cv2.DIST_L2,3)            #On créé la carte des distances
dist_transform1 = dist_transform.astype(np.uint8)
dist_transform1 = cv2.equalizeHist(dist_transform1)
dist_transform1 = thresh1-dist_transform1
plt.figure(7)
plt.imshow(dist_transform1,cmap='gray')
plt.title("Carte des distances")



#LPE


FAH_x=[[] for i in range(256)]                 #On initalise la FAH (une liste pour x et pour y)
FAH_y=[[] for i in range(256)]

(h,w) = np.shape(dist_transform1)           #On récupère la taille de l'image

for x in range(h):                          #On parcourt l'image
    for y in range(w):
        pix = dist_transform1[x][y]         
        if Im[x][y] != 0 :                  #Si il un pixel a un label (différent de 0)
            FAH_x[pix].append(x)            #Alors on ajoute les cordonnées de ce pixel dans la liste n°pix (priorité dans la carte des distances)
            FAH_y[pix].append(y)

    
for i in range (256):               #On parcourt la FAH
    fin = True                      #On créé une condition d'arrêt pour la boucle
    while (fin == True) :  
        n=len(FAH_x[i])             #On récupère la taille de la FAH (identique pour x et y)
        if n != 0 :                         #Tant qu'elle est différente de zéro on la parcourt
            for j in range(n) :
                x = FAH_x[i].pop(-1)        #On enlève (et récupère) la der,ière valeur de la FAH
                y = FAH_y[i].pop(-1)
                Lx = []                     #On initialise deux listes
                Ly = []
                if x != 0 :                 # Stockage des pixels voisins sans label
                    if(Im[x-1][y]==0):
                        Lx.append(x - 1)
                        Ly.append(y)
                if x != h-1 :
                    if(Im[x+1][y]==0):
                        Lx.append(x + 1)
                        Ly.append(y)
                if y != 0 :
                    if(Im[x][y-1]==0):
                        Lx.append(x)
                        Ly.append(y - 1)
                if y != w-1: 
                    if Im[x][y+1]==0:
                        Lx.append(x)
                        Ly.append(y + 1)
                        
                m = len(Lx)                                 # 
                if( m != 0):
                    for k in range(m):
                        Im[Lx[k]][Ly[k]] = Im[x][y]
                        pix = dist_transform1[Lx[k]][Ly[k]]
                        if pix < i :
                            pix = i
                        FAH_x[pix].append(Lx[k])
                        FAH_y[pix].append(Ly[k])
                n=len(FAH_x[i])
        else :
            fin = False


plt.figure(8)
plt.imshow(Im)
plt.title("Image finale")
plt.show()
image = cv2.cvtColor(Im,cv2.COLOR_)