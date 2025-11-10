import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# -------------------------------------------------------------------
# -------------------------- 🚀 用户配置 --------------------------
# -------------------------------------------------------------------

# 图片和标签文件的路径
IMAGE_PATH = Path("your_image.jpg") # 例如: Path("data/images/train/hand_001.jpg")
LABEL_PATH = Path("your_label.txt") # 例如: Path("data/labels/train/hand_001.txt")

# 关键点数量 (手部通常是 21 个)
NUM_KEYPOINTS = 21

# 绘制参数
BBOX_COLOR = (0, 255, 0) # BGR: 绿色
BBOX_THICKNESS = 2
KEYPOINT_COLOR = (0, 0, 255) # BGR: 红色
KEYPOINT_RADIUS = 3
SKELETON_COLOR = (255, 0, 0) # BGR: 蓝色
SKELETON_THICKNESS = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
FONT_THICKNESS = 1

# 手部关键点连接关系 (索引从0开始)
# 这个列表定义了如何连接21个手部关节来形成骨架
# 每个子列表是一个连接 [点1索引, 点2索引]
# 这是一个常见的21个手部关键点的连接顺序
HAND_SKELETON_CONNECTIONS = [
    [0, 1], [1, 2], [2, 3], [3, 4],       # 拇指
    [0, 5], [5, 6], [6, 7], [7, 8],       # 食指
    [0, 9], [9, 10], [10, 11], [11, 12],  # 中指
    [0, 13], [13, 14], [14, 15], [15, 16], # 无名指
    [0, 17], [17, 18], [18, 19], [19, 20]  # 小指
]

# -------------------------------------------------------------------
# -------------------------- 📜 脚本主体 --------------------------
# -------------------------------------------------------------------

def parse_yolo_label(label_path: Path, img_width: int, img_height: int):
    """
    解析 YOLO pose 格式的标签文件。
    返回一个列表，其中每个元素是一个字典，包含 'bbox' 和 'keypoints'。
    bbox: [x_min, y_min, x_max, y_max] (像素坐标)
    keypoints: [[x1, y1, v1], [x2, y2, v2], ...] (像素坐标)
    """
    annotations = []
    if not label_path.exists():
        print(f"警告: 标签文件不存在: {label_path}")
        return annotations

    with open(label_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = list(map(float, line.strip().split()))
            if not parts:
                continue

            class_id = int(parts[0])
            
            # Bounding box (cx, cy, w, h) - 归一化
            cx_norm, cy_norm, w_norm, h_norm = parts[1:5]

            # 转换边界框到像素坐标 (xmin, ymin, xmax, ymax)
            x_center = cx_norm * img_width
            y_center = cy_norm * img_height
            width = w_norm * img_width
            height = h_norm * img_height

            x_min = int(x_center - width / 2)
            y_min = int(y_center - height / 2)
            x_max = int(x_center + width / 2)
            y_max = int(y_center + height / 2)
            
            bbox = [x_min, y_min, x_max, y_max]

            # 关键点 (x, y, v) - 归一化
            keypoints_data = parts[5:]
            keypoints = []
            
            # 每个关键点有3个值 (x, y, v)
            if len(keypoints_data) < NUM_KEYPOINTS * 3:
                print(f"警告: 标签行关键点数据不足 ({len(keypoints_data)}/{NUM_KEYPOINTS*3}): {line}")
                continue

            for i in range(0, NUM_KEYPOINTS * 3, 3):
                kp_x_norm, kp_y_norm, kp_v = keypoints_data[i:i+3]
                
                kp_x = int(kp_x_norm * img_width)
                kp_y = int(kp_y_norm * img_height)
                kp_v = int(kp_v) # 可见性通常是 0, 1, 2

                keypoints.append([kp_x, kp_y, kp_v])
            
            annotations.append({'class_id': class_id, 'bbox': bbox, 'keypoints': keypoints})
            
    return annotations


def draw_annotations(image_path: Path, label_path: Path):
    """
    在图片上绘制边界框和关键点。
    """
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"错误: 无法读取图片 {image_path}")
        return

    img_height, img_width = img.shape[:2]

    # 解析标签
    annotations = parse_yolo_label(label_path, img_width, img_height)

    if not annotations:
        print(f"没有找到有效的标注在 {label_path} 中。")
        # 仍然显示原图
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(f"No Annotations Found for {image_path.name}")
        plt.axis('off')
        plt.show()
        return

    for anno in annotations:
        bbox = anno['bbox']
        keypoints = anno['keypoints']
        class_id = anno['class_id']

        # 1. 绘制边界框
        x_min, y_min, x_max, y_max = bbox
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), BBOX_COLOR, BBOX_THICKNESS)
        
        # 绘制类别ID (可选)
        label_text = f"Class: {class_id}"
        cv2.putText(img, label_text, (x_min, y_min - 10), FONT, FONT_SCALE, BBOX_COLOR, FONT_THICKNESS, cv2.LINE_AA)

        # 2. 绘制关键点
        kp_coords = [] # 存储可见关键点的坐标，用于绘制骨架
        for i, (kp_x, kp_y, kp_v) in enumerate(keypoints):
            if kp_v > 0: # 只绘制可见的关键点 (v=1 或 v=2)
                cv2.circle(img, (kp_x, kp_y), KEYPOINT_RADIUS, KEYPOINT_COLOR, -1) # -1 表示填充
                kp_coords.append((kp_x, kp_y)) # 添加到骨架连接列表
            else:
                kp_coords.append(None) # 如果不可见，占位 None

        # 3. 绘制骨架 (连接关键点)
        for connection in HAND_SKELETON_CONNECTIONS:
            p1_idx, p2_idx = connection
            
            # 确保两个点都在范围内且可见
            if p1_idx < len(kp_coords) and p2_idx < len(kp_coords) and \
               kp_coords[p1_idx] is not None and kp_coords[p2_idx] is not None:
                
                point1 = kp_coords[p1_idx]
                point2 = kp_coords[p2_idx]
                
                cv2.line(img, point1, point2, SKELETON_COLOR, SKELETON_THICKNESS, cv2.LINE_AA)

    # 显示图片
    # OpenCV 默认是 BGR 格式，matplotlib 期望 RGB 格式，所以需要转换
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(f"Annotations for {image_path.name}")
    plt.axis('off') # 不显示坐标轴
    plt.show()

# 主执行块
if __name__ == "__main__":
    # 请确保 IMAGE_PATH 和 LABEL_PATH 指向您准备好的文件
    # 例如：
    # IMAGE_PATH = Path("my_dataset/images/test_image.jpg")
    # LABEL_PATH = Path("my_dataset/labels/test_image.txt")
    
    # 示例用法:
    draw_annotations(IMAGE_PATH, LABEL_PATH)
