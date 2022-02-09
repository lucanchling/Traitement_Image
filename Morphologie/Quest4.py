import matplotlib.pyplot as plt
import numpy as np
import cv2

I = cv2.imread('calculator.png',0)

S = cv2.getStructuringElement(cv2.MORPH_RECT,(7,25))


plt.figure()
plt.subplot(2,2,1)
plt.imshow(I,'gray')
plt.axis('off')

ouv = cv2.morphologyEx(I, cv2.MORPH_OPEN, S)

plt.subplot(2,2,2)
plt.imshow(ouv,'gray')
plt.axis('off')


# Reconstruction par Dilatation
S = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
n=15
D = cv2.dilate(ouv,S,iterations = n)

plt.subplot(2,2,3)
plt.imshow(D,'gray')
plt.axis('off')
plt.show()