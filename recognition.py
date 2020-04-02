#!/usr/bin/python3

# Usage: ./recognition.py --apikey trnsl.1.1.20200401T091642Z.afc596326aefca71.245a5fb159cca6cf82634f1ec65dead7c67a5c21 --padding 0.275 --image images/gag0.webp

from imutils.object_detection import non_max_suppression
from PIL import Image, ImageDraw, ImageFont
from spellchecker import SpellChecker
from yandex.Translater import Translater
import argparse
import cv2
import numpy as np
import os
import pytesseract
import rectango as rtg
import textwrap

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-l", "--language", default="best-eng",
	type=str,
	help="trained language data name")
ap.add_argument("-i", "--image",
	type=str,
	help="path to input image")
ap.add_argument("-east", "--east",
	type=str, default="data/frozen_east_text_detection.pb",
	help="path to input EAST text detector")
ap.add_argument("-c", "--min-confidence",
	type=float, default=0.5,
	help="minimum probability required to inspect a region")
ap.add_argument("-w", "--width",
	type=int, default=320,
	help="nearest multiple of 32 for resized width")
ap.add_argument("-e", "--height",
	type=int, default=320,
	help="nearest multiple of 32 for resized height")
ap.add_argument("-p", "--padding",
	type=float, default=0.0,
	help="amount of padding to add to each border of ROI")
ap.add_argument("-k", "--apikey",
	type=str, default=
	"trnsl.1.1.20200401T091642Z.afc596326aefca71"
	".245a5fb159cca6cf82634f1ec65dead7c67a5c21",
	help="Yandex Translate Api Key")
args = vars(ap.parse_args())

# Get grayscale image
def get_grayscale(image):
	return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Thresholding
def thresholding(image):
	return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

# Set enviroment variable with path to tesseract trained data
os.environ["TESSDATA_PREFIX"] = "data"

# Setup spellchecker
spell = SpellChecker()

# Setup translater
translater = Translater()
translater.set_key(args["apikey"])
translater.set_from_lang('en')
translater.set_to_lang('ru')

# Load the input image and grab the image dimensions
image = cv2.imread(args["image"])
orig = image.copy()
(origH, origW) = image.shape[:2]

# Set the new width and height and then determine the ratio in change
# for both the width and height
(newW, newH) = (args["width"], args["height"])
rW = origW / float(newW)
rH = origH / float(newH)

# Resize the image and grab the new image dimensions
image = cv2.resize(image, (newW, newH))
(H, W) = image.shape[:2]

# Define the two output layer names for the EAST detector model that
# we are interested -- the first is the output probabilities and the
# second can be used to derive the bounding box coordinates of text
layerNames = [
	"feature_fusion/Conv_7/Sigmoid",
	"feature_fusion/concat_3"]

# Load the pre-trained EAST text detector
print("[INFO] loading EAST text detector...")
net = cv2.dnn.readNet(args["east"])

# Construct a blob from the image and then perform a forward pass of
# the model to obtain the two output layer sets
blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
	(123.68, 116.78, 103.94), swapRB=True, crop=False)
net.setInput(blob)
(scores, geometry) = net.forward(layerNames)

# Grab the number of rows and columns from the scores volume, then
# initialize our set of bounding box rectangles and corresponding
# confidence scores
(numRows, numCols) = scores.shape[2:4]
rects = []
multiboxes = []
multipultiboxes = []
confidences = []

# Loop over the number of rows
for y in range(0, numRows):
	# Extract the scores (probabilities), followed by the geometrical
	# data used to derive potential bounding box coordinates that
	# surround text
	scoresData = scores[0, 0, y]
	xData0 = geometry[0, 0, y]
	xData1 = geometry[0, 1, y]
	xData2 = geometry[0, 2, y]
	xData3 = geometry[0, 3, y]
	anglesData = geometry[0, 4, y]

	# Loop over the number of columns
	for x in range(0, numCols):
		# If our score does not have sufficient probability, ignore it
		if scoresData[x] < args["min_confidence"]:
			continue

		# Compute the offset factor as our resulting feature maps will
		# be 4x smaller than the input image
		(offsetX, offsetY) = (x * 4.0, y * 4.0)

		# Extract the rotation angle for the prediction and then
		# compute the sin and cosine
		angle = anglesData[x]
		cos = np.cos(angle)
		sin = np.sin(angle)

		# Use the geometry volume to derive the width and height of
		# the bounding box
		h = xData0[x] + xData2[x]
		w = xData1[x] + xData3[x]

		# Compute both the starting and ending (x, y)-coordinates for
		# the text prediction bounding box
		endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
		endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
		startX = int(endX - w)
		startY = int(endY - h)

		# Add the bounding box coordinates and probability score to
		# our respective lists
		rects.append((startX, startY, endX, endY))
		confidences.append(scoresData[x])

# Apply non-maxima suppression to suppress weak, overlapping bounding boxes
boxes = non_max_suppression(np.array(rects), probs=confidences)

for (startX, startY, endX, endY) in boxes:
	# In order to obtain a better OCR of the text we can potentially
	# apply a bit of padding surrounding the bounding box -- here we
	# are computing the deltas in both the x and y directions
	dX = int((endX - startX) * args["padding"])
	dY = int((endY - startY) * args["padding"])

	# Apply padding to each side of the bounding box, respectively
	startX = max(0, startX - dX)
	startY = max(0, startY - dY)
	endX = min(origW, endX + (dX * 2))
	endY = min(origH, endY + (dY * 2))
	multiboxes.append([startX, startY, endX, endY])

# Merges all overlapping rectangles
multipultiboxes = rtg.merge(multiboxes)

cv2.imshow("Original", orig)

# Loop over the bounding boxes
for (startX, startY, endX, endY) in multipultiboxes:
	# Scale the bounding box coordinates based on the respective ratios
	startX = int(startX * rW)
	startY = int(startY * rH)
	endX = int(endX * rW)
	endY = int(endY * rH)

	roi = orig[startY:endY, startX:endX]
	(roiH, roiW) = roi.shape[:2]
	config = ("--oem 1 --psm 6 -l " + args["language"])

	roi = get_grayscale(roi)
	roi = thresholding(roi)

	cv2.imshow("Text Detection " + str(startX), roi)

	# Recognize text in box
	text = pytesseract.image_to_string(roi, config=config)

	print("OCR TEXT")
	print("========")
	print("Raw:")
	print("{}\n".format(text))

	print("Corrected:")
	corrected = text.replace("|", "I")
	corrected = corrected.replace("1", "l")
	corrected = corrected.replace("0", "O")
	corrected = spell.correction(corrected)
	print("{}\n".format(corrected))

	# Translation
	translater.set_text(corrected)
	translation = translater.translate()
	print("Translated:")
	print("{}\n".format(translation))

	img = Image.new('RGB', (roiW, roiH), color = (255, 255, 255))
	font_color = (0,0,0)
	font_size = 12

	d = ImageDraw.Draw(img)
	unicode_font = ImageFont.truetype("DejaVuSans.ttf", font_size)
	wrapped = textwrap.fill(translation, int(roiW / 8))
	print("Wrapped:")
	print("{}\n".format(wrapped))

	d.text((4,4), wrapped, font=unicode_font, fill=font_color)

	img = np.array(img)
	x_offset = startX
	y_offset = startY
	img_w = img.shape[1]
	img_h = img.shape[0]
	
	orig[y_offset: y_offset + img_h, x_offset: x_offset + img_w] = img
	
cv2.imshow("Result", orig)
# cv2.imwrite("res.jpg", orig)
cv2.waitKey(0)
