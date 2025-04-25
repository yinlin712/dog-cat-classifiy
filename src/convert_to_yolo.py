import os
import shutil
from PIL import Image


def convert_dataset_to_yolo(src_dir, dest_dir, class_map):
    for split in ["train", "val"]:
        for class_name, class_id in class_map.items():
            image_dir = os.path.join(src_dir, split, class_name)
            for fname in os.listdir(image_dir):
                if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                img_path = os.path.join(image_dir, fname)
                label_name = os.path.splitext(fname)[0] + ".txt"

                # 创建新路径
                new_img_dir = os.path.join(dest_dir, "images", split)
                new_label_dir = os.path.join(dest_dir, "labels", split)
                os.makedirs(new_img_dir, exist_ok=True)
                os.makedirs(new_label_dir, exist_ok=True)

                # 拷贝图片
                new_img_path = os.path.join(new_img_dir, fname)
                shutil.copy(img_path, new_img_path)

                # 获取图像大小
                with Image.open(img_path) as img:
                    w, h = img.size

                # 写入 label（整张图为 bbox）
                with open(os.path.join(new_label_dir, label_name), "w") as f:
                    f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")


if __name__ == "__main__":
    source_dir = "dataset"
    yolo_dir = "yolo_dataset"
    class_id_map = {"cats": 0, "dogs": 1}
    convert_dataset_to_yolo(source_dir, yolo_dir, class_id_map)
    print("YOLO 格式转换完成！")
