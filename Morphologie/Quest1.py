from cv2 import MORPH_ELLIPSE
import matplotlib.pyplot as plt
import numpy as np
import cv2

I = cv2.imread('Ampoule.png',0)
S = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))

n = 4
# Dilatation
D = cv2.dilate(I,S,iterations = n)
# Erosion
E = cv2.erode(I,S,iterations=n)
# Calcul du gradient
grad = D - E

plt.figure()
plt.subplot(2,2,1)
plt.imshow(I,'gray')
plt.axis('off')
# Affichage
cv2.imwrite('Ampoule_grad.png',grad)
plt.subplot(2,2,2)
plt.imshow(grad,'gray')
plt.axis('off')
# Elimination des reflets
S = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
ouv = cv2.morphologyEx(I, cv2.MORPH_OPEN, S)

cv2.imwrite('Ampoule_sans_reflet.png',ouv)
plt.subplot(2,2,3)
plt.imshow(ouv,'gray')
plt.axis('off')
S = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
n = 5
# Dilatation
D = cv2.dilate(ouv,S,iterations = n)
# Erosion
E = cv2.erode(ouv,S,iterations=n)
# Calcul du gradient
grad = D - E

# Affichage
cv2.imwrite('Ampoule_grad_sans_reflet.png',grad)
plt.subplot(2,2,4)
plt.imshow(grad,'gray')
plt.axis('off')
plt.show()