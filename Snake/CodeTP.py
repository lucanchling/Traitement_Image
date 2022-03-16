from re import X
import numpy as np
from matplotlib import pyplot as plt
from scipy import sparse as sp
import cv2

# Lecture de l'image
I = cv2.imread('im_goutte.png',0)

theta = np.linspace(0, 2*np.pi, 100)

radius = 80

x = 100 + radius*np.cos(theta)
y = 120 + radius*np.sin(theta)

plt.figure()
plt.plot(x,y)
plt.imshow(I,'gray')
plt.show()


# Choix des paramètres :
alpha = 0
beta = 0
gamma = 0

# Calcul des différentes grandeurs 