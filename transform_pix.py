import os
import cv2

# 定义训练集和测试集文件夹路径（根据实际情况修改）
train_folder = "dataset/train_folder"
test_folder = "dataset/test_folder"

# 定义函数用于转换图像分辨率
def convert_images_resolution(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg") or file.endswith(".JPG"):
                file_path = os.path.join(root, file)
                try:
                    img = cv2.imread(file_path)
                    img = cv2.resize(img, (512, 512))
                    cv2.imwrite(file_path, img)
                except Exception as e:
                    print(f"无法处理文件 {file_path}，错误信息：{e}")

# 转换训练集图像分辨率
convert_images_resolution(train_folder)
# 转换测试集图像分辨率
convert_images_resolution(test_folder)