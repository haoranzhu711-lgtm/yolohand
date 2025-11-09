import os
import shutil
import cv2  # 用于读取图片尺寸
from tqdm import tqdm  # 用于显示进度条
from pathlib import Path

# -------------------------------------------------------------------
# -------------------------- 🚀 用户配置 --------------------------
# -------------------------------------------------------------------

# 1. 主图片列表文件：包含所有需要提取的图片路径（每行一个）
#    假设这个文件里的路径是 'img_001.jpg' 或 'subfolder/img_002.png'
MAIN_IMAGE_LIST_FILE = r"C:\path\to\your\main_list.txt"

# 2. 四个源数据文件夹的路径列表
SOURCE_FOLDERS = [
    r"C:\path\to\folder_1",
    r"C:\path\to\folder_2",
    r"C:\path\to\folder_3",
    r"C:\path\to\folder_4"
]

# 3. 指定哪个文件夹用作测试集（索引从 0 开始）
#    例如：0, 1, 2 将成为训练集，3 将成为测试集
TEST_FOLDER_INDEX = 3  # 将 folder_4 (索引为3) 作为测试集

# 4. YOLO 格式的类别 ID（class_id）
#    所有检测框都将使用这个 ID。
CLASS_ID = 0

# 5. 关键点的可见性（visibility）
#    YOLO 关键点格式通常是 (x, y, v)
#    v=2: 标记并可见
#    v=1: 标记但遮挡
#    v=0: 未标记
#    我们这里假设所有关键点都是标记并可见的
KEYPOINT_VISIBILITY = 2

# 6. 新的 YOLO 数据集输出目录
#    脚本将在此处创建 'images' 和 'labels' 文件夹
OUTPUT_DATASET_DIR = r"C:\my_yolo_dataset"


# -------------------------------------------------------------------
# -------------------------- 📜 脚本主体 --------------------------
# -------------------------------------------------------------------

def load_annotations(data_txt_path: Path) -> dict:
    """
    加载单个 data.txt 文件
    返回一个字典：{ '图片相对路径': [x, y, h, w, x1, y1, ...] }
    """
    annotations = {}
    if not data_txt_path.exists():
        print(f"警告：找不到标注文件 {data_txt_path}")
        return annotations

    with open(data_txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue

            # 假设第一项是路径，后面都是数字
            image_relative_path = parts[0]
            try:
                data = [float(p) for p in parts[1:]]
                # 至少需要 bbox (x, y, h, w)
                if len(data) >= 4:
                    annotations[image_relative_path] = data
                else:
                    print(f"警告：标注行格式错误（数据不足）：{line}")
            except ValueError:
                print(f"警告：标注行格式错误（非数字）：{line}")

    return annotations


def convert_to_yolo(img_width: int, img_height: int, bbox: list, keypoints: list, class_id: int,
                    visibility: int) -> str:
    """
    将标注转换为 YOLO 格式字符串 (bbox + keypoints)
    bbox 格式假定为 [x, y, h, w] (左上角x, 左上角y, 高度, 宽度)
    """

    # 1. 转换 Bounding Box
    # 原始数据：x, y, h, w
    # YOLO 格式：center_x_norm, center_y_norm, width_norm, height_norm
    x_min, y_min, box_h, box_w = bbox

    cx = x_min + box_w / 2
    cy = y_min + box_h / 2

    cx_norm = cx / img_width
    cy_norm = cy / img_height
    w_norm = box_w / img_width
    h_norm = box_h / img_height

    yolo_bbox_str = f"{class_id} {cx_norm:.6f} {cy_norm:.6f} {w_norm:.6f} {h_norm:.6f}"

    # 2. 转换 Keypoints
    yolo_kpts_parts = []
    if len(keypoints) % 2 != 0:
        print(f"警告：关键点数量不是偶数！将忽略最后一个点。")
        keypoints = keypoints[:-1]

    for i in range(0, len(keypoints), 2):
        kp_x = keypoints[i]
        kp_y = keypoints[i + 1]

        kp_x_norm = kp_x / img_width
        kp_y_norm = kp_y / img_height

        yolo_kpts_parts.append(f"{kp_x_norm:.6f} {kp_y_norm:.6f} {visibility}")

    yolo_kpts_str = " ".join(yolo_kpts_parts)

    # 3. 组合
    if yolo_kpts_str:
        return f"{yolo_bbox_str} {yolo_kpts_str}"
    else:
        return yolo_bbox_str


def create_yolo_dataset():
    """
    主函数：执行整个数据集创建过程
    """
    print("🚀 开始创建 YOLO 数据集...")

    # 将所有路径转换为 Path 对象
    main_list_path = Path(MAIN_IMAGE_LIST_FILE)
    source_folder_paths = [Path(p) for p in SOURCE_FOLDERS]
    output_dir = Path(OUTPUT_DATASET_DIR)

    # 1. 创建输出目录结构
    train_img_dir = output_dir / "images" / "train"
    val_img_dir = output_dir / "images" / "val"  # YOLO 常用 'val' 作为测试/验证集
    train_label_dir = output_dir / "labels" / "train"
    val_label_dir = output_dir / "labels" / "val"

    for d in [train_img_dir, val_img_dir, train_label_dir, val_label_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"已创建输出目录：{output_dir}")

    # 2. 加载所有文件夹的标注
    #    数据结构：{ '图片相对路径': {'data': [...], 'folder_path': Path(...), 'folder_index': int} }
    all_annotations = {}
    print("正在加载所有 data.txt 标注文件...")
    for i, folder_path in enumerate(source_folder_paths):
        data_txt = folder_path / "data.txt"
        annotations = load_annotations(data_txt)
        for relative_path, data in annotations.items():
            if relative_path in all_annotations:
                print(f"警告：发现重复的图片路径 '{relative_path}'。")
                print(f"       将使用来自文件夹 {i} ('{folder_path}') 的条目。")
            all_annotations[relative_path] = {
                'data': data,
                'folder_path': folder_path,
                'folder_index': i
            }
    print(f"总共加载了 {len(all_annotations)} 条唯一的标注。")

    # 3. 加载主图片列表
    if not main_list_path.exists():
        print(f"错误：找不到主图片列表文件：{main_list_path}")
        return

    with open(main_list_path, 'r', encoding='utf-8') as f:
        target_image_paths = [line.strip() for line in f if line.strip()]

    print(f"从主列表加载了 {len(target_image_paths)} 个目标图片。")

    # 4. 遍历主列表，处理每张图片
    processed_count = 0
    skipped_count = 0
    print("开始处理图片和标签...")

    for relative_path in tqdm(target_image_paths):
        # 4.1 查找标注
        if relative_path not in all_annotations:
            print(f"警告：在 data.txt 中未找到 '{relative_path}' 的标注。跳过...")
            skipped_count += 1
            continue

        info = all_annotations[relative_path]
        annotation_data = info['data']
        source_folder = info['folder_path']
        folder_index = info['folder_index']

        # 4.2 检查源图片是否存在
        src_img_path = source_folder / relative_path
        if not src_img_path.exists():
            print(f"警告：图片文件不存在 '{src_img_path}'。跳过...")
            skipped_count += 1
            continue

        # 4.3 确定是训练集还是测试集
        if folder_index == TEST_FOLDER_INDEX:
            split = "val"
            dest_img_dir = val_img_dir
            dest_label_dir = val_label_dir
        else:
            split = "train"
            dest_img_dir = train_img_dir
            dest_label_dir = train_label_dir

        # 4.4 读取图片尺寸
        img = cv2.imread(str(src_img_path))
        if img is None:
            print(f"警告：无法读取图片 '{src_img_path}'。跳过...")
            skipped_count += 1
            continue
        img_height, img_width = img.shape[:2]

        # 4.5 转换格式
        # 假设格式：x, y, h, w, x1, y1, x2, y2, ...
        bbox = annotation_data[0:4]
        keypoints = annotation_data[4:]

        try:
            yolo_label_str = convert_to_yolo(img_width, img_height, bbox, keypoints, CLASS_ID, KEYPOINT_VISIBILITY)
        except Exception as e:
            print(f"错误：转换 '{relative_path}' 时出错：{e}。跳过...")
            skipped_count += 1
            continue

        # 4.6 定义输出路径（处理潜在的文件名冲突）
        # 将 'sub/img.jpg' 转换为 'folder0_sub_img.jpg' 和 'folder0_sub_img.txt'
        p = Path(relative_path)
        # 组合父目录和文件名，替换路径分隔符
        flat_name_parts = [str(part) for part in p.parts]
        flat_name = "_".join(flat_name_parts)  # e.g., 'sub_img.jpg'

        output_stem = f"folder{folder_index}_{Path(flat_name).stem}"  # e.g., 'folder0_sub_img'
        output_ext = p.suffix  # e.g., '.jpg'

        dest_img_path = dest_img_dir / f"{output_stem}{output_ext}"
        dest_label_path = dest_label_dir / f"{output_stem}.txt"

        # 4.7 复制图片和写入标签
        try:
            shutil.copy2(src_img_path, dest_img_path)
            with open(dest_label_path, 'w', encoding='utf-8') as f:
                f.write(yolo_label_str)
            processed_count += 1
        except Exception as e:
            print(f"错误：复制或写入文件时出错：{e}。跳过...")
            skipped_count += 1

    print("\n" + "=" * 30)
    print("🎉 处理完成！")
    print(f"总共处理图片：{processed_count}")
    print(f"总共跳过图片：{skipped_count}")
    print(f"数据集已保存到：{output_dir}")
    print("=" * 30)


if __name__ == "__main__":
    create_yolo_dataset()