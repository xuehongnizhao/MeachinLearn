from PIL import Image
from pylab import *

im = array(Image.open('/Users/zq/Downloads/640.jpeg').convert('L'),'f')
im2 = 255 - im
im3 = (100.0/255) * im +100
im4 = 255.0 * (im/255.0)**2
subplot(221)
title('f(x) = x',fontsize=9)
gray()
imshow(im)

subplot(222)
title('f(x) = 255 - x',fontsize=9)
gray()
imshow(im2)

subplot(223)
title('f(x) = (100/255)*x +100',fontsize=9)
gray()
imshow(im3)

subplot(224)
title('f(x) = 255*(x/255)^2',fontsize=9)
gray()
imshow(im4)

show()
# import cv2  
# import numpy as np  
# from matplotlib import pyplot as plt  
# img=cv2.imread('/Users/zq/Downloads/640.jpeg')  
# GrayImage=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  
# ret,thresh1=cv2.threshold(GrayImage,127,255,cv2.THRESH_BINARY)  
# ret,thresh2=cv2.threshold(GrayImage,127,255,cv2.THRESH_BINARY_INV)  
# ret,thresh3=cv2.threshold(GrayImage,127,255,cv2.THRESH_TRUNC)  
# ret,thresh4=cv2.threshold(GrayImage,127,255,cv2.THRESH_TOZERO)  
# ret,thresh5=cv2.threshold(GrayImage,127,255,cv2.THRESH_TOZERO_INV)  
# titles = ['Gray Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']  
# images = [GrayImage, thresh1, thresh2, thresh3, thresh4, thresh5]  
# for i in xrange(6):  
#    plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')  
#    plt.title(titles[i])  
#    plt.xticks([]),plt.yticks([])  
# plt.show() 