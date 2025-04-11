import cv2

# 이미지 불러오기
img = cv2.imread("Original.jpg")

# 크기 조절
resized_img = cv2.resize(img, (300, 300))  # (width, height)
# 해상도를 늘릴 때는 따로 지정 안 하면 선형보간 씀, 너무 늘리면 블러 현상
# 줄일 때는 주변 픽셀 값 평균내서 줄임, 줄이면 정보 손실이 생김 -> 디테일 손실

# 저장 (선택)
cv2.imwrite("resized.jpg", resized_img)

# 확인 (선택)
cv2.imshow("Resized Image", resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
