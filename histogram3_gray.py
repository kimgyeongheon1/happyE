import numpy as np         # 숫자 배열 계산을 위해 사용
import matplotlib.pyplot as plt  # 결과 시각화를 위해 사용
import cv2             # 이미지 파일을 불러오고 저장하는 데 사용 (PIL 대신 사용)

# 이미지를 흑백으로 불러오는 함수
def load_image_gray(path):
    img = cv2.imread(path)         # 이미지 읽기 (컬러일 수도 있음)
    if len(img.shape) == 3:            # 만약 컬러 이미지면
        img = img[:, :, 0]             # 첫 번째 채널(R)을 사용하여 흑백처럼 사용
    return img.astype(np.uint8)        # 정수형으로 변환해서 리턴

# 히스토그램을 직접 계산하는 함수
def compute_histogram(image):
    hist = np.zeros(256)               # 픽셀 값 0~255를 위한 빈 배열
    for i in range(image.shape[0]):    # 이미지 세로 방향
        for j in range(image.shape[1]):  # 이미지 가로 방향
            value = image[i, j]        # 현재 픽셀 값 가져오기
            hist[value] += 1           # 해당 값에 해당하는 히스토그램 카운트 증가
    return hist

# 확률 분포 함수 (PDF) 계산
def compute_pdf(hist, total_pixels):
    return hist / total_pixels         # 각 빈도를 전체 픽셀 수로 나눔

# 누적 분포 함수 (CDF)를 수작업으로 계산
def compute_cdf_manual(pdf):
    cdf = np.zeros_like(pdf)           # 같은 크기의 배열 생성
    cum = 0.0                          # 누적합을 저장할 변수
    for i in range(len(pdf)):
        cum += pdf[i]                  # 이전 값들과 더해가며 누적
        cdf[i] = cum                   # CDF 배열에 저장
    return cdf

# CDF를 기반으로 픽셀 매핑 테이블 생성
def create_mapping(cdf_source, cdf_reference):
    mapping = np.zeros(256, dtype=np.uint8)  # 0~255의 매핑 저장용
    for src_val in range(256):
        diff = np.abs(cdf_source[src_val] - cdf_reference)  # 차이 계산
        mapping[src_val] = np.argmin(diff)   # 가장 비슷한 값의 인덱스를 매핑
    return mapping

# 히스토그램 매칭 수행 함수
def histogram_matching(source_img, reference_img):
    hist_source = compute_histogram(source_img)           # 원본 히스토그램
    hist_reference = compute_histogram(reference_img)     # 기준 히스토그램

    pdf_source = compute_pdf(hist_source, source_img.size)       # PDF 계산
    pdf_reference = compute_pdf(hist_reference, reference_img.size)

    cdf_source = compute_cdf_manual(pdf_source)            # CDF 계산
    cdf_reference = compute_cdf_manual(pdf_reference)

    mapping = create_mapping(cdf_source, cdf_reference)    # 매핑 테이블 생성

    # 실제 매핑 적용
    matched_img = np.zeros_like(source_img)
    for i in range(source_img.shape[0]):
        for j in range(source_img.shape[1]):
            matched_img[i, j] = mapping[source_img[i, j]]
    return matched_img

# 히스토그램을 출력하는 함수
def plot_histogram(image, title):
    plt.hist(image.flatten(), bins=256, range=(0, 255), color='gray')
    plt.title(title)
    plt.xlabel('Pixel value')
    plt.ylabel('Frequency')
    plt.grid(True)

# 결과를 한 화면에 보여주는 함수
def show_results(source, reference, matched):
    plt.figure(figsize=(12, 8))

    # 이미지 출력
    plt.subplot(2, 3, 1)
    plt.imshow(source, cmap='gray')
    plt.title("Source Image")

    plt.subplot(2, 3, 2)
    plt.imshow(reference, cmap='gray')
    plt.title("Reference Image")

    plt.subplot(2, 3, 3)
    plt.imshow(matched, cmap='gray')
    plt.title("Matched Image")

    # 히스토그램 출력
    plt.subplot(2, 3, 4)
    plot_histogram(source, "original Histogram")

    plt.subplot(2, 3, 5)
    plot_histogram(reference, "Reference Histogram")

    plt.subplot(2, 3, 6)
    plot_histogram(matched, "Matched Histogram")

    plt.tight_layout()
    plt.show()

# 메인 함수 역할
if __name__ == "__main__":
    # 이미지 불러오기
    source = load_image_gray("C:/vscode/histogram/Original.jpeg")
    reference = load_image_gray("C:/vscode/histogram/Reference.jpeg")

    # 히스토그램 매칭 수행
    matched = histogram_matching(source, reference)

    # 결과 보기
    show_results(source, reference, matched)

    # 저장도 가능
    cv2.imwrite("C:/vscode/histogram/Result.jpg", matched)
