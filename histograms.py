import cv2
import numpy as np
import matplotlib.pyplot as plt

def calculate_cdf(hist):
    cdf = hist.cumsum()
    cdf_normalized = cdf / cdf[-1]
    return cdf_normalized

def match_histograms(source, template):
    # 1. Flatten the images and calculate histogram
    source_hist, bins = np.histogram(source.flatten(), 256, [0,256])
    template_hist, bins = np.histogram(template.flatten(), 256, [0,256])

    # 2. Calculate CDFs
    source_cdf = calculate_cdf(source_hist)
    template_cdf = calculate_cdf(template_hist)

    # 3. Create lookup table
    lookup_table = np.zeros(256)
    for src_pixel in range(256):
        closest_value = np.abs(template_cdf - source_cdf[src_pixel]).argmin()
        lookup_table[src_pixel] = closest_value

    # 4. Apply the mapping
    matched = cv2.LUT(source, lookup_table.astype(np.uint8))
    return matched

# 이미지 불러오기 (흑백으로)
source_img = cv2.imread('source.jpg', cv2.IMREAD_GRAYSCALE)
template_img = cv2.imread('template.jpg', cv2.IMREAD_GRAYSCALE)

matched_img = match_histograms(source_img, template_img)

# 결과 출력
plt.figure(figsize=(15,5))
plt.subplot(1,3,1); plt.imshow(source_img, cmap='gray'); plt.title("Source Image")
plt.subplot(1,3,2); plt.imshow(template_img, cmap='gray'); plt.title("Template Image")
plt.subplot(1,3,3); plt.imshow(matched_img, cmap='gray'); plt.title("Matched Image")
plt.show()
