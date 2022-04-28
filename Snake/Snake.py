from ctypes import sizeof
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.ndimage import gaussian_filter
import cv2

def moyenne(liste):
    return sum(liste)/len(liste)

# Choix de l'image à traiter ("goutte" or "im10")
choixImage = "im10"

##########################
# Acquisition de l'image #
##########################
if choixImage== "goutte":
    I = plt.imread("im_goutte.png")
if choixImage == "im10":
    I = plt.imread("im10.png")
I = I[:,:,0]
# Ajout d'un flou pour fluidifier l'approche du snake
I = cv2.GaussianBlur(I, (3,3),cv2.BORDER_DEFAULT)
li,col = len(I),len(I[0])
################

##############
# Parametres #
##############
if choixImage=="im10":
    alpha = 1
    beta = 0.5
    gamma = 15
if choixImage == "goutte":
    alpha = 5
    beta = 2
    gamma = 10
################

#####################
# Creation du snake #
#####################
centre=[int(col/2),int(li/2)]
rayon=min(int((col-5)/2),int((li-5)/2))
delta = 1.
K = 1000
snakeX,snakeY = [],[]
pas = (2*np.pi)/K
for i in range(K):
    teta = i*pas
    snakeX = np.append(snakeX, int(centre[0] + rayon * np.cos(teta)))
    snakeY = np.append(snakeY, int(centre[1] + rayon * np.sin(teta)))
#######################

##################################
# Creation de D1, D2, D4, D et A #
##################################
Id = np.identity(K)
D1 = np.roll(Id, 1, axis=-1) + Id*(0) - np.roll(Id,-1, axis=1)
D2 = np.roll(Id, -1, axis=1) + Id*(-2) + np.roll(Id,1, axis=1)
D4 = (np.roll(Id, -1, axis=1) + np.roll(Id,1, axis=1))*-4 + (np.roll(Id, -2, axis=1) + np.roll(Id,2, axis=1)) + Id*(6)
D = alpha*D2 - beta*D4
A = np.linalg.inv(Id - D)
################################

#########################################
#Calcul des differents Grad necessaires #
#########################################
Gy,Gx = np.gradient(I.astype(float))
NormGrad = Gx**2 + Gy**2
GGy, GGx = np.gradient(NormGrad)
# plt.figure()
# plt.subplot(131)
# plt.imshow(Gx)
# plt.subplot(132)
# plt.imshow(Gy)
# plt.subplot(133)
# plt.imshow(NormGrad,'gray')
#########################################

#######################
# Algorithme iteratif #
#######################
NRJ,NRJELA,NRJCOURB,NRJEXT = [],[],[],[]    # différentes énergies du snake
MOY,MINMAX,DELTA = [],[],[] # différents calculs pour critère d'arrêt sur l'énergie
GxSnake = np.zeros(snakeX.shape)
GySnake = np.zeros(snakeY.shape)
it=0  # nombre d'itération
flag = True
j=0
if choixImage == "goutte":
    limite = 3270
if choixImage == "im10":
    limite = 14600

while flag and it<limite:
    for i in range(K):
        Y=int(snakeY[i])
        X=int(snakeX[i])
        GxSnake[i] = GGx[Y][X]
        GySnake[i] = GGy[Y][X]
    snakeX = np.dot(A, snakeX+gamma*GxSnake)
    snakeY = np.dot(A, snakeY+gamma*GySnake)
    # Calcul de l'energie
    ELA,COURB,EXT = 0,0,0
    Xnprime = np.dot(D1, snakeX)
    Ynprime = np.dot(D1, snakeY)
    Xnseconde = np.dot(D2, snakeX)
    Ynseconde = np.dot(D2, snakeY)
    for k in range(K):
        ELA += alpha*0.5*np.sqrt(np.square(Xnprime[k]) + np.square(Ynprime[k]))
        COURB += beta*0.5*np.sqrt(np.square(Xnseconde[k]) + np.square(Ynseconde[k]))
        EXT += NormGrad[int(snakeY[k]),int(snakeX[k])]**2
    NRJ.append(ELA+COURB-EXT)
    NRJEXT.append(EXT)
    NRJCOURB.append(COURB)
    NRJELA.append(ELA)
    
    ###################
    # Critère d'arrêt #
    ###################
    if it>300:
        delta1 = [NRJ[it-i] for i in range(250)]
        MOY.append(moyenne(delta1)) # Lissage de l'énergie en faisant une moyenne sur 250 valeurs
    
    # Test condition d'arrêt sur l'énergie #
    # à partir du lissage de l'énergie, on essaye de détermine 
    # l'itération pour laquelle il y a un palier quasiment constant
    # --> non fonctionnel pour l'im10 car profil différent  
    # if it>1000:
    #     delta = [NRJ[it-i] for i in range(20)]
    #     minmax=max(delta)-min(delta)
    #     print(round(minmax,2))
    #     if minmax<0.00003:
    #         j+=1
    #         #print(it)
    #         #print(j)
    #     if j > 100:
    #         flag = False
    #     print(sum(delta)/len(delta))
    #     i=1
    #     delta = abs(abs(NRJ[it-i]-NRJ[it-i-2])-abs(NRJ[it-i-1]-NRJ[it-i-3]))
    #     print(delta)
    #     if (delta<0.5):
    #         flag = False
    
    it += 1
#########################

# Affichage
plt.figure()
plt.imshow(I,'gray')
plt.plot(snakeX, snakeY, 'r', linewidth=1)
plt.text(col/6,15,"Alpha = "+str(alpha)+" ; Beta = "+str(beta)+" ; Gamma = "+str(gamma))
plt.text(col/3,25,"Pour "+str(it)+" itérations")

plt.figure()
#plt.subplot(221)
plt.plot(NRJ)
# plt.title('NRJ globale')
# plt.subplot(222)
# plt.plot(NRJCOURB)
# plt.title('NRJ Courbe')
# plt.subplot(223)
# plt.plot(NRJELA)
# plt.title('NRJ Elastique')
# plt.subplot(224)
# plt.plot(NRJEXT)
# plt.title('NRJ Exterieure')

# plt.figure()
# plt.plot(MOY)
plt.show()