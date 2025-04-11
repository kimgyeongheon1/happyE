import numpy as np

def histogram_matching(source_img, template_img):
    # 1. 각 이미지의 히스토그램 계산
    source_hist = np.zeros(256)
    template_hist = np.zeros(256)

    for pixel in source_img.flatten():
        source_hist[pixel] += 1

    for pixel in template_img.flatten():
        template_hist[pixel] += 1

    # 2. 정규화 (확률 분포로)
    source_pdf = source_hist / source_img.size
    template_pdf = template_hist / template_img.size

    # 3. CDF 계산
    source_cdf = np.cumsum(source_pdf)
    template_cdf = np.cumsum(template_pdf)

    # 4. 매핑 테이블 생성
    mapping = np.zeros(256, dtype=np.uint8)
    for src_pixel in range(256):
        # source CDF와 가장 가까운 template CDF를 찾음
        diff = np.abs(template_cdf - source_cdf[src_pixel])
        mapping[src_pixel] = np.argmin(diff)

    # 5. 매핑 적용
    matched_img = np.zeros_like(source_img)
    height, width = source_img.shape
    for i in range(height):
        for j in range(width):
            matched_img[i, j] = mapping[source_img[i, j]]

    return matched_img
