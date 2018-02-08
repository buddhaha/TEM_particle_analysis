from matplotlib import pyplot as plt
import numpy as np
import cv2
import sys

# Create a black image

a = 2518
mask = np.zeros((a,a), np.uint8)
mid = a / 2
# Draw a diagonal blue line with thickness of 5 px


#cv2.line(img,(0,0),(511,511),(255,0,0),5)

mask = cv2.ellipse(mask,(mid,mid),(220,75),45,0,360,255,-1)

cv2.imwrite('ell_try.jpg', mask)



img_name = sys.argv[1]
img = cv2.imread(img_name,0)

f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)

#plt.imshow(fshift)
#plt.savefig('fshift.png')
#plt.clear()

magnitude_spectrum = 20*np.log(np.abs(fshift))

print 'FFT done'

print magnitude_spectrum.shape

'''
plt.subplot(131),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(132),plt.imshow(mask, cmap = 'gray')
plt.title('mask'), plt.xticks([]), plt.yticks([])
plt.subplot(133),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
'''

# apply mask
#fft_w_mask = cv2.bitwise_and(magnitude_spectrum,magnitude_spectrum,mask = mask)
print 'no mask applied...'
#cv2.imwrite('fft.png', fft_w_mask)
#plt.savefig('fft.png')

#invf = np.fft.ifft2(fft_w_mask)
print 'IFFT w mask done!'

f_ishift = np.fft.ifftshift(magnitude_spectrum) #fft_w_mask
img_back = np.fft.ifft2(f_ishift)
img_back = np.abs(img_back)

plt.imshow(img_back, cmap = 'gray')
plt.title('IFFT wMask'), plt.xticks([]), plt.yticks([])
plt.savefig('ifft_w_mask.png', cmap = 'gray')
plt.show()


