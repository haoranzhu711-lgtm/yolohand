import os
import shutil
import cv2  # 用于读取图片尺寸
from tqdm import tqdm  # 用于显示进度条
from pathlib import Path

# -------------------------------------------------------------------
# -------------------------- 🚀 用户配置 --------------------------
# -------------------------------------------------------------------

# 1. 主图片列表文件：
#    假设里面的路径是 'subfolder1/img_001.jpg', 'subfolder2/img_abc.png' ...
#    这些路径是 *相对于* 下面 SOURCE_FOLDERS 中某一个的路径
MAIN_IMAGE_LIST_FILE = r"C:\path\to\your\main_list.txt"

# 2. 四个 *根* 源数据文件夹的路径列表
SOURCE_FOLDERS = [
    r"C:\path\to\Source_Folder_1",
    r"C:\path\to\Source_Folder_2",
    r"C:\path\to\Source_Folder_3",
    r"C:\path\to\Source_Folder_4"
]

# 3. 指定哪个文件夹用作测试集（索引从 0 开始）
#    例如：0, 1, 2 将成为训练集，3 将成为测试集
TEST_FOLDER_INDEX = 3  # 将 Source_Folder_4 (索引为3) 作为测试集

# 4. YOLO 格式的类别 ID（class_id）
CLASS_ID = 0

# 5. 关键点的可见性（visibility）
#    YOLO 关键点格式 (x, y, v), v=2 表示可见
KEYPOINT_VISIBILITY = 2

# 6. 新的 YOLO 数据集输出目录
OUTPUT_DATASET_DIR = r"C:\my_yolo_dataset"

# -------------------------------------------------------------------
# -------------------------- 📜 脚本主体 --------------------------
# -------------------------------------------------------------------

def load_annotations_from_file(data_txt_path: Path) -> dict:
    """
    加载 *单个* data.txt 文件。
    假设此 data.txt 中的路径是相对于该文件本身的（例如 'img_001.jpg'）
    返回一个字典：{ '图片文件名': [x, y, h, w, x1, y1, ...] }
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
            
            # 假设第一项是路径（文件名），后面都是数字
            # 我们只取文件名作为key，以防万一路径中包含 './'
            image_filename = Path(parts[0]).name
            try:
                data = [float(p) for p in parts[1:]]
                if len(data) >= 4:
                    annotations[image_filename] = data
                else:
                    print(f"警告：标注行格式错误（数据不足）：{line} @ {data_txt_path}")
            except ValueError:
                print(f"警告：标注行格式错误（非数字）：{line} @ {data_txt_path}")
                
    return annotations

def convert_to_yolo(img_width: int, img_height: int, bbox: list, keypoints: list, class_id: int, visibility: int) -> str:
    """
    将标注转换为 YOLO 格式字符串 (bbox + keypoints)
    bbox 格式假定为 [x, y, h, w] (左上角x, 左上角y, 高度, 宽度)
    """
    
    # 1. 转换 Bounding Box
    # 原始数据：x_min, y_min, box_h, box_w
    # YOLO 格式：center_x_norm, center_y_norm, width_norm, height_norm
    x_min, y_min, box_h, box_w = bbox
    
    # 确保h和w是正数
    if box_w <= 0 or box_h <= 0:
        raise ValueError(f"Bounding box 尺寸无效: w={box_w}, h={box_h}")

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
        kp_y = keypoints[i+1]
        
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
    print("🚀 开始创建 YOLO 数据集（嵌套结构版）...")
    
    main_list_path = Path(MAIN_IMAGE_LIST_FILE)
    source_folder_paths = [Path(p) for p in SOURCE_FOLDERS]
    output_dir = Path(OUTPUT_DATASET_DIR)

    # 1. 创建输出目录结构
    train_img_dir = output_dir / "images" / "train"
    val_img_dir = output_dir / "images" / "val"
    train_label_dir = output_dir / "labels" / "train"
    val_label_dir = output_dir / "labels" / "val"
    
    for d in [train_img_dir, val_img_dir, train_label_dir, val_label_dir]:
        d.mkdir(parents=True, exist_ok=True)
        
    print(f"已创建输出目录：{output_dir}")

    # 2. 加载主图片列表
    if not main_list_path.exists():
        print(f"错误：找不到主图片列表文件：{main_list_path}")
        return
        
    with open(main_list_path, 'r', encoding='utf-8') as f:
        # 使用 Path(line.strip()).as_posix() 来标准化路径分隔符
        target_image_paths = [Path(line.strip()).as_posix() for line in f if line.strip()]
        
    print(f"从主列表加载了 {len(target_image_paths)} 个目标图片。")

    # 3. 准备一个缓存来存储已加载的 data.txt 内容
    #    键: data.txt 的绝对路径
    #    值: { 'img1.jpg': [data...], 'img2.jpg': [data...] }
    annotation_cache = {}

    # 4. 遍历主列表，处理每张图片
    processed_count = 0
    skipped_count = 0
    print("开始处理图片和标签...")
    
    for relative_path_str in tqdm(target_image_paths):
        
        relative_path = Path(relative_path_str)
        
        # 4.1 查找这张图片在哪个源文件夹中
        found_source = False
        full_img_path = None
        source_folder_index = -1
        
        for i, source_root in enumerate(source_folder_paths):
            test_path = source_root / relative_path
            if test_path.exists():
                full_img_path = test_path
                source_folder_index = i
                found_source = True
                break
        
        if not found_source:
            print(f"警告：在所有源文件夹中都找不到图片 '{relative_path_str}'。跳过...")
            skipped_count += 1
            continue
            
        # 4.2 确定 data.txt 的路径和图片文件名
        # full_img_path = C:\path\to\Source_Folder_1\subfolder1\img_001.jpg
        sub_folder_path = full_img_path.parent
        image_filename = full_img_path.name
        data_txt_path = sub_folder_path / "data.txt"
        data_txt_path_str = str(data_txt_path)
        
        # 4.3 从缓存加载或读取 data.txt
        if data_txt_path_str not in annotation_cache:
            # print(f"加载新的标注文件：{data_txt_path_str}")
            annotation_cache[data_txt_path_str] = load_annotations_from_file(data_txt_path)
            if not annotation_cache[data_txt_path_str]:
                print(f"警告：标注文件为空或不存在：{data_txt_path_str}")
        
        annotations_in_subfolder = annotation_cache[data_txt_path_str]
        
        # 4.4 查找该图片的标注
        if image_filename not in annotations_in_subfolder:
            print(f"警告：在 '{data_txt_path_str}' 中未找到 '{image_filename}' 的标注。跳过...")
            skipped_count += 1
            continue
            
        annotation_data = annotations_in_subfolder[image_filename]

        # 4.5 确定是训练集还是测试集
        if source_folder_index == TEST_FOLDER_INDEX:
            dest_img_dir = val_img_dir
            dest_label_dir = val_label_dir
        else:
            dest_img_dir = train_img_dir
            dest_label_dir = train_label_dir
            
        # 4.6 读取图片尺寸
        img = cv2.imread(str(full_img_path))
        if img is None:
            print(f"警告：无法读取图片 '{full_img_path}'。跳过...")
            skipped_count += 1
            continue
        img_height, img_width = img.shape[:2]
        
        # 4.7 转换格式
        bbox = annotation_data[0:4]
        keypoints = annotation_data[4:]
        
        try:
            yolo_label_str = convert_to_yolo(img_width, img_height, bbox, keypoints, CLASS_ID, KEYPOINT_VISIBILITY)
        except Exception as e:
            print(f"错误：转换 '{relative_path_str}' 时出错：{e}。跳过...")
            skipped_count += 1
            continue

        # 4.8 定义输出路径（处理文件名冲突）
        # 我们使用 "源文件夹索引" + "相对路径" 来创建唯一的扁平化名称
        # 例如: 'subfolder1/img_001.jpg' 变为 'folder0_subfolder1_img_001.jpg'
        
        # 将 'subfolder1/img_001.jpg' 替换路径分隔符为 '_'
        flat_name = relative_path_str.replace('/', '_').replace('\\', '_')
        
        output_stem = f"folder{source_folder_index}_{Path(flat_name).stem}"
        output_ext = relative_path.suffix
        
        dest_img_path = dest_img_dir / f"{output_stem}{output_ext}"
        dest_label_path = dest_label_dir / f"{output_stem}.txt"
        
        # 4.9 复制图片和写入标签
        try:
            shutil.copy2(full_img_path, dest_img_path)
            with open(dest_label_path, 'w', encoding='utf-8') as f:
                f.write(yolo_label_str)
            processed_count += 1
        except Exception as e:
            print(f"错误：复制或写入文件时出错：{e}。跳过...")
            skipped_count += 1
            
    print("\n" + "="*30)
    print("🎉 处理完成！")
    print(f"总共处理图片：{processed_count}")
    print(f"总共跳过图片：{skipped_count}")
    print(f"数据集已保存到：{output_dir}")
    print("="*30)

if __name__ == "__main__":
    create_yolo_dataset()
