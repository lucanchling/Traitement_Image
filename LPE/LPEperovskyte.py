import numpy as np
import cv2
import matplotlib.pyplot as plt

I=cv2.imread('perovskyte.png')
IMAGE_HSV = cv2.cvtColor(I, cv2.COLOR_BGR2HSV)
h,s,v = cv2.split(IMAGE_HSV)
#plt.show()
G = I[:,:,0]
B = I[:,:,1]
R = I[:,:,2]

plt.figure()
plt.subplot(2,2,1)
plt.imshow(h,cmap='gray')
plt.title('Image Originale')


ret,thresh1 =  cv2.threshold(h,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
#RET,I1 =  cv2.threshold(s,130,100,cv2.THRESH_BINARY)

plt.subplot(2,2,2)
plt.imshow(thresh1,cmap='gray')
plt.title('Seuillage')

#création d'un élément structurant circulaire de rayon 7 pixels
#S = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5)) 
#I_ERODE = cv2.erode(I1,S, iterations = 1)
#I_ERODE = cv2.morphologyEx(I1, cv2.MORPH_OPEN, S)
S = cv2.getStructuringElement(cv2.MORPH_RECT,(6,11)) 
I_ERODE = cv2.erode(thresh1,S, iterations = 1)
plt.subplot(2,2,3)
plt.imshow(I_ERODE,cmap='gray')
plt.title('Erosion')


ret1, labels=cv2.connectedComponents(I_ERODE)

#RET,I2 =  cv2.threshold(R,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
#I2 = 255 - I2
#S = cv2.getStructuringElement(cv2.MORPH_RECT,(3,2)) 
#I2 =  cv2.erode(I2,S, iterations = 1)


ret2, labels2=cv2.connectedComponents(255-thresh1)

Im = 10*labels2+labels
plt.subplot(2,2,4)
plt.imshow(Im)
plt.title("Marqueurs initiaux utilisés pour la LPE")


dist_transform = cv2.distanceTransform(thresh1,cv2.DIST_L2,3)
dist_transform1 = dist_transform.astype(np.uint8)
dist_transform1 = cv2.equalizeHist(dist_transform1)
dist_transform1 = thresh1-dist_transform1
plt.figure(7)
plt.imshow(dist_transform1,cmap='gray')
plt.title("Carte des distances")



#LPE


FAH_x=[[] for i in range(256)]
FAH_y=[[] for i in range(256)]

(h,w) = np.shape(dist_transform1)

for x in range(h):
    for y in range(w):
        pix = dist_transform1[x][y]
        if Im[x][y] != 0 :
            FAH_x[pix].append(x)
            FAH_y[pix].append(y)

    
for i in range (256):
    fin = True
    while (fin) :  
        n=len(FAH_x[i])
        if n != 0 :
            for j in range(n) :
                x = FAH_x[i].pop(-1)
                y = FAH_y[i].pop(-1)
                Lx = []
                Ly = []
                if x != 0 :
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
                        
                m = len(Lx)
                if( m != 0):
                    for k in range(m):
                        Im[Lx[k]][Ly[k]] = Im[x][y]
                        pix = dist_transform1[Lx[k]][Ly[k]]
                        if pix < i :
                            pix = i
                        FAH_x[pix].append(Lx[k])
                        FAH_y[pix].append(Ly[k])
        else :
            fin = False

plt.figure(8)
plt.imshow(Im,cmap='tab20')
plt.title("Image finale")
plt.show()     