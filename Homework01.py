import cv2
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

image_path =  r"image/1eecab90-1a92-43a7-b952-0204384e1fae.jpg"

img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    print("Error: Image could not be loaded. Please check the path.")
else:
    # 应用阈值分割
    _, thresholded_img = cv2.threshold(img, 170, 255, cv2.THRESH_BINARY)
    images_combined = np.hstack((img, thresholded_img))

    # 寻找轮廓
    contours, _ = cv2.findContours(thresholded_img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    # 遍历轮廓并绘制边界框
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    plt.figure(figsize=(10, 5))  # 设置图像大小
    plt.imshow(images_combined, cmap='gray')  # 显示图像
    plt.title("原图与处理后的图像")  # 设置图像标题
    plt.xticks([img.shape[1] // 2, img.shape[1] + img.shape[1] // 2], ["原图", "处理后的图像"])
    plt.yticks([])  # 禁用y轴刻度
    plt.show()
