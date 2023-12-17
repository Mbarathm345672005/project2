import matplotlib.pyplot as plt
import numpy as np
import cv2

def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def gaussian_blur(image, kernel_size=5):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def canny(image, low_threshold, high_threshold):
    return cv2.Canny(image, low_threshold, high_threshold)

def region_of_interest(image, vertices):
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, vertices, 255)
    return cv2.bitwise_and(image, mask)

def draw_lines(image, lines, color=[255, 0, 0], thickness=5):
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(image, (x1, y1), (x2, y2), color, thickness)

def hough_lines(image, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(image, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((*image.shape, 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

def weighted_img(initial_img, img, α=0.8, β=1., γ=0.):
    return cv2.addWeighted(initial_img, α, img, β, γ)

def lane_detection(image_path):
    # Load the image
    image = plt.imread(image_path)

    # Convert image to grayscale
    gray = grayscale(image)

    # Apply Gaussian smoothing
    blur = gaussian_blur(gray, kernel_size=5)

    # Apply Canny edge detection
    edges = canny(blur, low_threshold=50, high_threshold=150)

    # Define region of interest
    imshape = image.shape
    vertices = np.array([[(50, imshape[0]), (450, 320), (490, 320), (imshape[1]-50, imshape[0])]], dtype=np.int32)
    masked_edges = region_of_interest(edges, vertices)

    # Apply Hough transform
    lines = hough_lines(masked_edges, rho=2, theta=np.pi/180, threshold=15, min_line_len=40, max_line_gap=20)

    # Draw the detected lines on the original image
    result = weighted_img(image, lines)

    return result

# Take user input for the image path
image_path = input("Enter the path of the image: ")

# Perform lane detection
result = lane_detection(image_path)

# Display the original and processed images side by side
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.imshow(plt.imread(image_path))
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(result)
plt.title('Lane Detection Result')

plt.show()
