from re import X
from cv2 import grabCut
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

# plt.figure()
# plt.plot(x,y)
# plt.imshow(I,'gray')
# plt.show()


# Choix des paramètres :
alpha = 0
beta = 0
gamma = 0

# Calcul des différentes grandeurs

# Le Gradient
[Gx,Gy] = np.gradient(I)
Gx_norm = Gx/np.max(Gx)
Gy_norm = Gy/np.max(Gy)
GradNorme = np.sqrt(Gx_norm**2+Gy_norm**2)

plt.figure()
plt.imshow(GradNorme,'gray')
plt.show()

# Opérateur Différentiel

D2 = sp.diags([1,1,-2,1,1],[-len(x)+1,-1,0,1,len(x)-1],(len(x),len(x)))

D4 = sp.diags([-4,1,1,-4,6,-4,1,1,-4],[-len(x)+1,-len(x)+2,-2,-1,0,1,2,len(x)-2,len(x)-1],(len(x),len(x)))

