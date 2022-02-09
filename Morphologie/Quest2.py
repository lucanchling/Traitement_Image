from cv2 import connectedComponents
import matplotlib.pyplot as plt
import numpy as np
import cv2

I = cv2.imread('blobs2.png',0)

O = cv2.distanceTransform(255-I,cv2.DIST_L2,5)

plt.figure()
plt.subplot(2,2,1)
plt.imshow(I,'gray')
plt.axis('off')
plt.subplot(2,2,2)
plt.imshow(O,'gray')
plt.axis('off')


S = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
n=2
# Erosion
E = cv2.erode(255-I,S,iterations=n)
plt.subplot(2,2,3)
plt.imshow(E,'gray')
plt.axis('off')
plt.show()

print("Le nombre de cercles est de :",connectedComponents(E)[0]-1)
# maxi = 0
# for i in range(1,len(O)):
#     if max(O[i])>maxi:
#         maxi = max(O[i])
# print(maxi)