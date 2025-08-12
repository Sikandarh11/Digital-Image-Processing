import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = r"E:\2. DIP\2. LAB\pythonProject\Projects\Lab 10\image4.png"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Error: Image not found. Check the path.")
    exit()

image = cv2.resize(image, (64, 128))  # Resize for simplicity

sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
angle = np.arctan2(sobel_y, sobel_x) * (180 / np.pi)

angle[angle < 0] += 180  # Convert negative angles to 0-180 range

# 2. Parameters
cell_size = 8
block_size = 2
bin_count = 9
bin_width = 180 // bin_count

h, w = image.shape
cell_x = w // cell_size
cell_y = h // cell_size

# 3. Histogram of Oriented Gradients for each cell
cell_hist = np.zeros((cell_y, cell_x, bin_count))

for i in range(cell_y):
    for j in range(cell_x):
        mag_cell = magnitude[i * cell_size:(i + 1) * cell_size, j * cell_size:(j + 1) * cell_size]
        angle_cell = angle[i * cell_size:(i + 1) * cell_size, j * cell_size:(j + 1) * cell_size]

        hist = np.zeros(bin_count)
        for y in range(cell_size):
            for x in range(cell_size):
                angle_val = angle_cell[y, x]
                if angle_val == 180:
                    angle_val = 179.9  # Avoid bin index 9 (out of bounds)
                bin_idx = int(angle_val // bin_width)
                hist[bin_idx] += mag_cell[y, x]

        cell_hist[i, j] = hist

# 4. Normalize histograms using 2x2 blocks
hog_vector = []

for i in range(cell_y - block_size + 1):
    for j in range(cell_x - block_size + 1):
        block = cell_hist[i:i + block_size, j:j + block_size].flatten()
        norm = np.linalg.norm(block) + 1e-6
        block = block / norm
        block = np.clip(block, 0, 0.2)
        hog_vector.extend(block)

hog_vector = np.array(hog_vector)

# Print and visualize
print("HoG descriptor shape:", hog_vector.shape)
print("HoG descriptor:", hog_vector)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.title("Gradient Magnitude")
plt.imshow(magnitude, cmap='gray')
plt.subplot(1, 2, 2)
plt.title("Gradient Direction")
plt.imshow(angle, cmap='hsv')
plt.show()
