import cv2
import numpy as np

# Load the image
image = cv2.imread('your_image_path.jpg')

# Convert the image to HSV color space
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define the color range for segmentation (adjust these values based on your image)
lower_color = np.array([0, 0, 0])  # lower threshold for the color
upper_color = np.array([255, 255, 255])  # upper threshold for the color

# Create a mask using the color range
mask = cv2.inRange(hsv_image, lower_color, upper_color)

# Apply the mask to the original image
segmented_image = cv2.bitwise_and(image, image, mask=mask)

# Display the original and segmented images
cv2.imshow('Original Image', image)
cv2.imshow('Segmented Image', segmented_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
