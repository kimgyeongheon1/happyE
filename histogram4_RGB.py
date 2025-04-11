import numpy as np                      # 숫자 배열 계산을 위해 사용
import matplotlib.pyplot as plt         # 결과 시각화를 위해 사용
import cv2

# 이미지를 그대로 불러옴
def imageload(path):
    img = cv2.imread(path)         # 이미지 읽기 -> 이거 BGR로 읽음
    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return image_rgb.astype(np.uint8)        # 정수형으로 변환해서 리턴

# 히스토그램 구하기
def compute_histogram(image):
    height = len(image) # 전체 행렬 길이 -> 행 개수
    width = len(image[0]) # 첫 행의 길이 -> 열 개수
    
    histogram = np.zeros(256)  # 데이터 공간을 bin으로 나누어 발생 빈도를 기록하기 위한 배열 
                                # 0~255, 각 인덱스는 0으로 초기화
    # 총 픽셀만큼 반복
    for i in range(height): # 행 수 -> 세로 방향
        for j in range(width): # 열 수 -> 가로 방향
            value = image[i][j] # 현재 픽셀 값 가져와서
            histogram[value] += 1 # 1 추가
    return histogram

# 확률 분포 함수 (PDF) 계산
def PDF(histogram, total_pixels):
    return histogram / total_pixels         # 히스토그램를 총 화소 수로 나눔

# 누적 분포 함수 (CDF)를 수작업으로 계산
def CDF(pdf):
    cdf = np.zeros_like(pdf)           # 같은 크기의 배열 생성
    L = 0.0                          # 누적합을 저장할 변수
    for i in range(len(pdf)):
        L += pdf[i]                  # 이전 값들과 더해가며 누적
        cdf[i] = L                   # CDF 배열에 저장
    return cdf

# CDF를 기반으로 픽셀 매핑 테이블 생성
def create_mapping(cdf_source, cdf_reference):
    mapping = np.zeros(256, dtype=np.uint8)  # 0~255의 매핑 저장용
    for src_val in range(256):
        diff = np.abs(cdf_source[src_val] - cdf_reference)  # 차이 계산
        mapping[src_val] = np.argmin(diff)   # 가장 비슷한 값의 인덱스를 매핑
    return mapping

# 히스토그램 1채널 매칭 수행 함수
def histogram_matching(source_img, reference_img):
    hist_source = compute_histogram(source_img)           # 원본 히스토그램
    hist_reference = compute_histogram(reference_img)     # 기준 히스토그램

    pdf_source = PDF(hist_source, source_img.size)       # PDF 계산 -> CDF 만들 때 사용
    pdf_reference = PDF(hist_reference, reference_img.size)

    cdf_source = CDF(pdf_source)            # CDF 계산
    cdf_reference = CDF(pdf_reference)

    mapping = create_mapping(cdf_source, cdf_reference)    # 매핑 테이블 생성

    # 실제 매핑 적용
    matched_img = np.zeros_like(source_img)
    for i in range(source_img.shape[0]):
        for j in range(source_img.shape[1]):
            matched_img[i, j] = mapping[source_img[i, j]]
    return matched_img

# RGB 이미지에 대해 R, G, B 각각 매칭 수행
def histogram_matching_rgb(source_img, reference_img):
    matched_img = np.zeros_like(source_img)  # 같은 크기 빈 이미지 준비
    for channel in range(3):  # 0=R, 1=G, 2=B
        matched_img[:, :, channel] = histogram_matching(
            source_img[:, :, channel],
            reference_img[:, :, channel]
        )
    return matched_img

# 히스토그램을 출력하는 함수
def plot_histogram(image):
    plt.hist(image.flatten(), bins=256, range=(0, 255), color='red') # .hist()는 히스토그램을 그리는 함수
    plt.xlabel('Pixel value')
    plt.ylabel('Frequency')

    plt.yticks([0, 1000, 2000, 3000, 4000])
    # plt.grid(True) -> 이거 쓰면 축 보임

# 결과를 한 화면에 보여주는 함수
def show_results(source, reference, matched):
    plt.figure(figsize=(12, 8))

    # 이미지 출력
    plt.subplot(2, 3, 1)
    plt.imshow(source)
    plt.title("Original")
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.imshow(reference)
    plt.title("Reference")
    plt.axis('off')
    
    plt.subplot(2, 3, 3)
    plt.imshow(matched)
    plt.title("Result")
    plt.axis('off')

    # 히스토그램 출력
    plt.subplot(2, 3, 4)
    plot_histogram(source)

    plt.subplot(2, 3, 5)
    plot_histogram(reference)

    plt.subplot(2, 3, 6)
    plot_histogram(matched)

    plt.tight_layout()
    plt.show()

# 메인 함수 역할
if __name__ == "__main__":
    # 이미지 불러오기
    source = imageload("C:/vscode/histogram/Original2.jpeg")
    reference = imageload("C:/vscode/histogram/Reference2.jpeg")

    # 히스토그램 매칭 수행
    matched = histogram_matching_rgb(source, reference)

    # 결과 보기
    show_results(source, reference, matched)

    # 저장도 가능
    cv2.imwrite("C:/vscode/histogram/Result2.jpg", matched)

""" 이미지 픽셀 숫자 보려고 적어 둠
print("Source image size:", source.shape)
print("Total pixels:", source.size)
print("Source image size:", reference.shape)
print("Total pixels:", reference.size)
"""