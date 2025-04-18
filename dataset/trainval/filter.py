import os
import shutil
import yaml
import re

def filter_yolo_dataset(dataset_root, classes_to_remove, yaml_path):
    """
    筛选Yolo数据集，删除包含指定类别的标记文件及其对应的图片
    
    参数:
    dataset_root: 数据集根目录路径
    classes_to_remove: 需要删除的类别索引列表
    yaml_path: yaml配置文件路径
    """
    # 将classes_to_remove转换为集合，以便快速查找
    classes_to_remove_set = set(classes_to_remove)
    
    # 更新后的类别映射
    class_mapping = {}
    current_idx = 0
    for i in range(100):  # 假设最多100个类别
        if i not in classes_to_remove_set:
            class_mapping[i] = current_idx
            current_idx += 1
    
    # 处理train和val文件夹
    for split in ['train', 'val']:
        labels_dir = os.path.join(dataset_root, split, 'labels')
        images_dir = os.path.join(dataset_root, split, 'images')
        
        if not os.path.exists(labels_dir) or not os.path.exists(images_dir):
            print(f"警告: {labels_dir} 或 {images_dir} 不存在")
            continue
        
        # 遍历所有标记文件
        for label_file in os.listdir(labels_dir):
            if not label_file.endswith('.txt'):
                continue
                
            label_path = os.path.join(labels_dir, label_file)
            
            # 读取标记文件内容
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            # 检查是否包含需要删除的类别
            should_remove = False
            new_lines = []
            
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:  # 确保有足够的部分
                    class_idx = int(parts[0])
                    if class_idx in classes_to_remove_set:
                        should_remove = True
                        break
                    else:
                        # 更新类别索引
                        parts[0] = str(class_mapping[class_idx])
                        new_lines.append(' '.join(parts) + '\n')
            
            # 如果需要删除标记文件
            if should_remove:
                # 删除标记文件
                os.remove(label_path)
                
                # 删除对应的图片文件
                image_basename = os.path.splitext(label_file)[0]
                for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                    image_path = os.path.join(images_dir, image_basename + ext)
                    if os.path.exists(image_path):
                        os.remove(image_path)
                        print(f"已删除: {image_path}")
                        break
                
                print(f"已删除: {label_path}")
            else:
                # 更新标记文件内容
                with open(label_path, 'w') as f:
                    f.writelines(new_lines)
                print(f"已更新: {label_path}")
    
    # 更新yaml文件
    update_yaml_config(yaml_path, classes_to_remove_set)

def update_yaml_config(yaml_path, classes_to_remove_set):
    """
    更新yaml配置文件，移除指定类别并更新类别总数
    
    参数:
    yaml_path: yaml配置文件路径
    classes_to_remove_set: 需要删除的类别索引集合
    """
    with open(yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 获取原始类别列表
    original_names = config['names']
    
    # 根据类别索引映射，创建新的类别列表
    new_names = [name for i, name in enumerate(original_names) if i not in classes_to_remove_set]
    
    # 更新类别总数和类别列表
    config['nc'] = len(new_names)
    config['names'] = new_names
    
    # 保存更新后的yaml文件
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    print(f"已更新yaml配置文件: {yaml_path}")
    print(f"新的类别总数: {config['nc']}")
    print(f"新的类别列表: {config['names']}")

if __name__ == "__main__":
    # 配置参数
    dataset_root = 'VOC2007'  # 数据集根目录
    yaml_path = 'trash.yaml'  # yaml配置文件路径
    
    # 需要删除的类别索引（从0开始）
    # 例如: 删除索引为1, 5, 10的类别
    classes_to_remove = [2,3,4,5,8,9,13,15,22,23,24,25,27,34,35,36,38,39,40,43]  # 修改为你需要删除的类别索引
    
    # 确认操作
    print(f"将删除以下类别索引: {classes_to_remove}")
    confirm = input("确认删除这些类别? (y/n): ")
    
    if confirm.lower() == 'y':
        filter_yolo_dataset(dataset_root, classes_to_remove, yaml_path)
        print("筛选完成!")
    else:
        print("操作已取消")