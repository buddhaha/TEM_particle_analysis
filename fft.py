import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys

def plotFFT(img):
	f = np.fft.fft2(img)

	fshift = np.fft.fftshift(f)

	magnitude_spectrum = 20*np.log(np.abs(fshift))

	plt.subplot(121),plt.imshow(img, cmap = 'gray')
	plt.title('Input Image'), plt.xticks([]), plt.yticks([])
	plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
	plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
	plt.savefig('fft.png')

	plt.clf() # .clear()

#cv2.imshow('pure_fft.png',magnitude_spectrum)



## PLOT ELLIPSE
# horizontally oriented:

# a = 100 
# b =  50
# rot = 45, counter-clock wise
def ellipseMask(img, a, b, rot):
	# img in grayscale, img.shape = (sizeX, sizeY, colorspace = 1)
	mask_shape = img.shape
	print mask_shape
	midx = float(mask_shape[0]) / 2
	midy = float(mask_shape[1]) / 2
	mid = (midx, midy)
	axis = (a, b)
	mask = np.zeros((mask_shape[0],mask_shape[1], 3), np.uint8)
	# full ellipse: 0,360 
	# color: 255
	# thickness: -1, filled
	cv2.ellipse(img,(midx, midy),(a,b),rot,0,360,255,-1)
	return mask


img_name = sys.argv[1]
img = cv2.imread(img_name,0)


plotFFT(img)
print 'fft done!'
elMask = ellipseMask(img, 100., 50., 45.)
cv2.imwrite('ellipse.png',elMask)

'''

# ellipse(img,(center),(a, b axis),anit-clock rotation,startAngle,endAngle(0,360 for full),255,-1)

cv2.ellipse(img,(256,256),(100,50),45,0,360,255,-1)
cv2.imwrite('ellipse.png',img)
'''
