# Importing necessary libraries
import numpy as np               # Importing the numpy library for numerical operations
import cv2                      # Importing the OpenCV library for image processing
from matplotlib import pyplot as plt  # Importing the matplotlib library for plotting

# Loading and preparing the image
image = cv2.imread(r'coin1.jpg')  # Loading an image
# Blurring the image to reduce noise and smooth intensity transitions
image = cv2.GaussianBlur(image, (5, 5), 0)  # You can adjust the kernel size as needed
blue, green, red = cv2.split(image)         # Splitting the image into its color channels (Blue, Green, Red)
rgb_image = cv2.merge([red, green, blue])   # Reordering the channels to display the image in RGB format
grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Converting the image to grayscale
ret, binary_threshold = cv2.threshold(grayscale_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)  # Applying Otsu's binary thresholding for segmentation

# Performing noise removal
structuring_element = np.ones((2, 2), np.uint8)  # Defining a 2x2 kernel for morphological operations

# Applying closing operation on the thresholded image
closed_image = cv2.morphologyEx(binary_threshold, cv2.MORPH_CLOSE, structuring_element, iterations=2)

# Identifying the sure background area
dilated_image = cv2.dilate(closed_image, structuring_element, iterations=3)  # Dilating the closed image to identify the sure background

# Calculating the distance transform to find the sure foreground area
distance_transform = cv2.distanceTransform(dilated_image, cv2.DIST_L2, 3)

# Thresholding the distance transform to find the sure foreground
ret, sure_foreground = cv2.threshold(distance_transform, 0.1 * distance_transform.max(), 255, 0)

# Identifying the unknown region
sure_foreground = np.uint8(sure_foreground)
unknown_region = cv2.subtract(dilated_image, sure_foreground)  # Calculating the unknown region by subtracting sure foreground from sure background

# Labeling markers
ret, markers = cv2.connectedComponents(sure_foreground)  # Labeling the sure foreground region

# Adding one to all labels to ensure that the sure background label is not 0 but 1
markers = markers + 1

# Marking the region of the unknown with zero
markers[unknown_region == 255] = 0

# Applying the Watershed algorithm for image segmentation
segmented_image = cv2.watershed(image, markers)

# Marking the segmented objects in the original image with red
image[segmented_image == -1] = [255, 0, 0]

# Plotting the original and segmented images
plt.subplot(211), plt.imshow(rgb_image)
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(212), plt.imshow(binary_threshold, 'gray')

# Saving the segmented image
plt.imsave(r'segmented_image.png', binary_threshold)

plt.title("Otsu's binary threshold"), plt.xticks([]), plt.yticks([])

plt.tight_layout()
plt.show()
