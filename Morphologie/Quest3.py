import matplotlib.pyplot as plt
import numpy as np
import cv2

I = cv2.imread('angiogram.png',0)

plt.figure()
plt.subplot(2,2,1)
plt.imshow(I,'gray')
plt.axis('off')

S = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(25,25))

Top = cv2.morphologyEx(I, cv2.MORPH_TOPHAT, S)


plt.subplot(2,2,2)
plt.imshow(Top,'gray')
plt.axis('off')

S = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))

ouv = cv2.morphologyEx(I, cv2.MORPH_OPEN, S)

plt.subplot(2,2,3)
plt.imshow(ouv,'gray')
plt.axis('off')


plt.show()