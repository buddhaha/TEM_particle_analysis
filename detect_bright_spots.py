'''
Detecting multiple bright spots in an image with Python and OpenCVPython
'''
# import the necessary packages

from imutils import contours
from skimage import measure
import numpy as np
import argparse
import imutils
import cv2
import matplotlib.pyplot as plt


class Micrograph:
	'''
	Micrograph class to handle & analyze TEM micrographs
	'''
	def __init__(self, path):
		self.path = path

	def __str__(self):
		str_to_print = '''
		File: {}
		Res (pix): {} x {}
		Pix res (nm): {} x {}
		'''.format(self.path, self.w, self.l, self.pixSizeX, self.pixSizeY)

		return str_to_print

	def __repr__(self):
		return self.__str__()

	def getInfo(self):
		print('printing info {}...'.format(self.path))

	def loadInfo(self):
		(self.w, self.l, self.pixSizeX, self.pixSizeY) = getImageResNM(self.path)
		self.pixAera = self.pixSizeX * self.pixSizeY


def main():
	print Micrograph.__doc__


if __name__ == '__main__':
	main()



# returns (sizeX (pix), sizeY (pix), pixResX (nm), pixResY (nm))
def getImageResNM(img_path):
	f = open( img_path, 'rb')
	tags = exifread.process_file(f)
	f.close()
	'''
	# show img metadata
	print type(tags)
	for key, val in tags.iteritems(): # PYTHON 3: tags.items()
		print key

	print tags['Image ImageLength']
	print tags['Image ImageWidth']
	print tags['Image ResolutionUnit']
	print tags['Image XResolution']
	print tags['Image YResolution']
	'''
	l = tags['Image ImageLength']
	w = tags['Image ImageWidth']
	res = tags['Image ResolutionUnit']
	x_res = tags['Image XResolution']
	y_res = tags['Image YResolution']
	print 'ResolutionUnit: ' + str(res)

	l = str(l)
	print 'l (px): ' + l
	w = str(w)
	print 'w (px): ' + w
	x_res = str(x_res)
	pixSizeX = 1 / ( int(x_res) * 100.0 ) #nm
	print 'x_res: ' + x_res
	print 'pixSizeX (nm): ' + str(pixSizeX)
	y_res = str(y_res)
	pixSizeY = 1 / ( int(y_res) * 100.0 ) #nm
	print 'pixSizeY (nm): ' + str(pixSizeY)
	print 'y_res: ' + y_res
	pixAera = pixSizeX * pixSizeY
	print 'pixAera (nm^2): ' + str(pixAera)
	'''
	if l == x_res and w == y_res:
		print 'symmetric'
	else:
		print 'non-symmetric'
	'''
	size_x = int(l) * 1.0 / int(x_res)
	size_y = int(w) * 1.0 / int(y_res)
	#print size_x, size_y
	print 'image size x [nm]: ' + str( size_x * 10 ** 7)
	print 'image size y [nm]: ' + str( size_y * 10 ** 7)

	return (w, l, pixSizeX, pixSizeY)


def radiFromAera(aera):
	# aera = np.pi * r^2
	return np.sqrt( aera / np.pi)



#plot current version of @imgname with its histogram as .png
def plotPixHist(img_data, img_name):

	plt.subplot(1,2,1)
	plt.imshow(img_data,'gray')
	plt.title( str(img_name) ), plt.xticks([]), plt.yticks([])

	plt.subplot(1,2,2)
	plt.hist(img_data.ravel(),256) #, normed=True)
	plt.tick_params(axis='y', which='both', labelleft='off', labelright='on')
	plt.title('histogram '), plt.xticks([0,256]), plt.xlim(0,256)#, plt.yticks([0,1])
	plt.savefig( str(img_name) + '_wHist.png')
	plt.close()


# perform a connected component analysis on the thresholded
# image, then initialize a mask to store only the "large"
# components
# min_size (px)
# return mask (cv2 object)
def findMask(img, min_size):
	labels = measure.label(img, neighbors=8, background=0)
	mask = np.zeros(img.shape, dtype="uint8")
	pix_dist = []

	#print 'shape ' + str(thresh.shape)
	#cv2.imwrite('mask00.png',mask)

	# loop over the unique components
	for label in np.unique(labels):
		# if this is the background label, ignore it
		if label < 0:
			continue
	 
		# otherwise, construct the label mask and count the
		# number of pixels 
		labelMask = np.zeros(img.shape, dtype="uint8")
		labelMask[labels == label] = 255

		#cv2.imwrite('labelMask{}.png'.format(i),mask)
		numPixels = cv2.countNonZero(labelMask)
		# if the number of pixels in the component is sufficiently
		# large, then add it to our mask of "large blobs"
		if numPixels > min_size:
			pix_dist.append(numPixels)
			#print 'numPixels: ' + str(numPixels)
#			print pix_dist
			mask = cv2.add(mask, labelMask)

	return mask, pix_dist


# plot the size distribution of detected features from a mask, #of occurrences vs radius (nm)
# radius is reconstructed from a circle with the same area as a feature

def plotPixDist(img_data, img_name):

	plt.subplot(1,2,1)
	plt.imshow(img_data,'gray')
	plt.title( str(img_name) ), plt.xticks([]), plt.yticks([])

	plt.subplot(1,2,2)
	plt.hist(img_data.ravel(),256) #, normed=True)
	plt.tick_params(axis='y', which='both', labelleft='off', labelright='on')
	plt.title('histogram '), plt.xticks([0,256]), plt.xlim(0,256)#, plt.yticks([0,1])
	plt.savefig( str(img_name) + '_wHist.png')
	plt.close()


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the image file")
args = vars(ap.parse_args())

# load the image 
image = cv2.imread(args["image"])

#
# return cv2 image object, dict/list of parameters
def loadImage(img_name):

	return

''''
####------ SCRIPT STRUCTURE ------####
# 1) load the image (w resolution)
# 2) Preprocess: crop, blur, 'erosions and dilations'
# 3) Process/analyze: detect features
#		- RIP: create mask from detected particles
#		- FL: perform FFT, apply mask, IFFT
# 4) Plot (detected, histograms)
'''


#----------- PREPROCESSINGS --------------
# convert it to grayscale,
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# and blur it
gray = cv2.GaussianBlur(gray, (3, 3), 0)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
gray = cv2.GaussianBlur(gray, (7, 7), 0)
blurred = cv2.GaussianBlur(gray, (11, 11), 0)

#cv2.imwrite('blurred.png',blurred)
plotPixHist(blurred, 'blurred')

# threshold the image to reveal light regions in the
# blurred image
thresh = cv2.threshold(blurred, 40, 255, cv2.THRESH_BINARY)[1]

#cv2.imwrite('thresh01.png',thresh)

# perform a series of erosions and dilations to remove
# any small blobs of noise from the thresholded image
thresh = cv2.erode(thresh, None, iterations=4)
thresh = cv2.dilate(thresh, None, iterations=6)


plotPixHist(thresh, 'thresh')
#cv2.imwrite('thresh02.png',thresh)

#-------------- ANALYSIS -----------------

mask, pix_dist = findMask(thresh, 15)


# apply the overlay
alpha = 0.5 #transparency value


#color_mask = mask #* [0, 0, 1] # only blue, [R G B]

#rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
#color_mask = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

#cv2.imwrite('image_w_mask.png',img_w_features)

color_mask = mask
color_mask[mask == 255] = [0, 0, 255]

w_color_mask = cv2.addWeighted(color_mask, alpha, image, 1 - alpha, 0) #, color_mask)
cv2.imwrite('w_color_mask.png',w_color_mask)

#cv2.imwrite('rgb_image.png',rgb_image)
#cv2.imwrite('color_mask.png',rgb_mask)
print mask.shape
#print 'image type' + str(type(image))
#print image.shape
#print 'color_mask type' + str(type(color_mask))
#print color_mask.shape
#print np.unique(color_mask)

cv2.imwrite('wColor_mask.png',w_color_mask)



bins = np.arange(-100, 100, 5) # fixed bin size
plt.xlim([min(pix_dist)-5, max(pix_dist)+5])

# bins = bins or bins = 'auto'

plt.hist(pix_dist, bins='auto', alpha=0.5)
plt.hist(pix_dist) #, normed=True)
plt.title('histogram ')#, plt.xticks([0,256]), plt.xlim(0,256)#, plt.yticks([0,1])
plt.savefig( 'thresh_mask_wHist.png')
'''
# perform a connected component analysis on the thresholded
# image, then initialize a mask to store only the "large"
# components
labels = measure.label(thresh, neighbors=8, background=0)
mask = np.zeros(thresh.shape, dtype="uint8")

print 'shape ' + str(thresh.shape)

cv2.imwrite('mask00.png',mask)

# loop over the unique components
i = 0
for label in np.unique(labels):
	i +=1
	print label
	# if this is the background label, ignore it
	if label < 1:
		continue
	
	print label 
	# otherwise, construct the label mask and count the
	# number of pixels 
	labelMask = np.zeros(thresh.shape, dtype="uint8")
	labelMask[labels == label] = 255

	#cv2.imwrite('labelMask{}.png'.format(i),mask)
	
	numPixels = cv2.countNonZero(labelMask)
 
	# if the number of pixels in the component is sufficiently
	# large, then add it to our mask of "large blobs"
	if numPixels > 11:
		print numPixels
		mask = cv2.add(mask, labelMask)
'''

cv2.imwrite('maskFc.png',mask)
print pix_dist

# find the contours in the mask, then sort them from left to
# right
cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
cnts = contours.sort_contours(cnts)[0]
 
print 'len cnts'
print len(cnts)

# loop over the contours
for (i, c) in enumerate(cnts):
	# draw the bright spot on the image
	(x, y, w, h) = cv2.boundingRect(c)
	((cX, cY), radius) = cv2.minEnclosingCircle(c)
	cv2.circle(image, (int(cX), int(cY)), int(radius),
		(0, 0, 255), 3)
	cv2.putText(image, "#{}".format(i + 1), (x, y - 15),
		cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
 
# show the output image

cv2.imwrite('image.png',image)
#cv2.imshow("Image", image)
#cv2.waitKey(0)
