import os
import yaml
import shutil
import glob

def find_class_files(class_id):
    """
    查找train和val文件夹下包含指定类别ID的YOLO格式标签文件
    
    参数:
    class_id: 需要查找的类别ID（整数）
    
    返回:
    包含指定类别ID的文件列表
    """
    # 获取类别名称
    with open('trash.yaml', 'r', encoding='utf-8') as f:
        yaml_data = yaml.safe_load(f)
        class_names = yaml_data.get('names', [])
    
    if 0 <= class_id < len(class_names):
        class_name = class_names[class_id]
        print(f"正在查找类别 {class_id}: '{class_name}' 的文件...")
    else:
        print(f"正在查找类别 {class_id} 的文件...")
    
    # 要查找的目录
    target_class = class_id
    found_files = []
    
    # 处理train和val文件夹
    for split in ['train', 'val']:
        labels_dir = os.path.join('VOC2007', split, 'labels')
        images_dir = os.path.join('VOC2007', split, 'images')
        
        if not os.path.exists(labels_dir):
            print(f"警告: {labels_dir} 目录不存在")
            continue
        
        print(f"正在搜索 {labels_dir}...")
        
        # 遍历所有标签文件
        for label_file in os.listdir(labels_dir):
            if not label_file.endswith('.txt'):
                continue
                
            label_path = os.path.join(labels_dir, label_file)
            
            # 读取标签文件内容
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            # 检查是否包含目标类别
            contains_class = False
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:  # YOLO格式至少有5个部分（类别 + 4个坐标值）
                    if int(parts[0]) == target_class:
                        contains_class = True
                        break
            
            # 如果包含目标类别，添加到结果列表
            if contains_class:
                image_basename = os.path.splitext(label_file)[0]
                image_file = None
                
                # 检查对应的图像文件是否存在
                for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                    img_path = os.path.join(images_dir, image_basename + ext)
                    if os.path.exists(img_path):
                        image_file = image_basename + ext
                        break
                
                found_files.append({
                    'split': split,
                    'label_file': label_file,
                    'image_file': image_file,
                    'label_path': label_path
                })
    
    return found_files

def process_marked_files(file_path, class_id):
    """
    处理带有标记的文件列表
    
    参数:
    file_path: 标记文件的路径
    class_id: 要处理的类别ID
    """
    if not os.path.exists(file_path):
        print(f"错误: 文件 {file_path} 不存在")
        return

    print(f"处理文件: {file_path}")
    deleted_files = 0
    modified_files = 0
    
    # 创建备份目录
    backup_dir = "backup_files"
    os.makedirs(backup_dir, exist_ok=True)
    
    with open(file_path, 'r', encoding='ansi') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
            
        # 检查是否有标记
        if "--" in line:  # 完全删除图片和标签
            file_path = line.split('--')[0].strip()
            delete_files_completely(file_path)
            deleted_files += 1
        elif "-+" in line:  # 只删除标签中特定类别
            file_path = line.split('-+')[0].strip()
            modified = remove_class_from_label(file_path, class_id, backup_dir)
            if modified:
                modified_files += 1
    
    print(f"处理完成: 删除了 {deleted_files} 个文件, 修改了 {modified_files} 个标签文件")

def delete_files_completely(rel_path):
    """
    完全删除图片和标签文件
    
    参数:
    rel_path: 标签文件的相对路径 (如 'train/img_123.txt')
    """
    parts = rel_path.split('/')
    if len(parts) != 2:
        print(f"错误: 无效的文件路径格式 {rel_path}")
        return False
    
    split, label_file = parts
    
    # 构建完整路径
    label_path = os.path.join('VOC2007', split, 'labels', label_file)
    image_basename = os.path.splitext(label_file)[0]
    
    # 如果标签文件存在，删除它
    if os.path.exists(label_path):
        try:
            # 创建备份
            backup_dir = os.path.join("backup_files", split, "labels")
            os.makedirs(backup_dir, exist_ok=True)
            shutil.copy2(label_path, os.path.join(backup_dir, label_file))
            
            # 删除文件
            os.remove(label_path)
            print(f"已删除标签文件: {label_path}")
        except Exception as e:
            print(f"删除 {label_path} 失败: {e}")
            return False
    
    # 查找并删除图片文件
    found_image = False
    for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        image_path = os.path.join('VOC2007', split, 'images', image_basename + ext)
        if os.path.exists(image_path):
            try:
                # 创建备份
                backup_dir = os.path.join("backup_files", split, "images")
                os.makedirs(backup_dir, exist_ok=True)
                shutil.copy2(image_path, os.path.join(backup_dir, image_basename + ext))
                
                # 删除文件
                os.remove(image_path)
                print(f"已删除图片文件: {image_path}")
                found_image = True
                break
            except Exception as e:
                print(f"删除 {image_path} 失败: {e}")
                return False
    
    if not found_image:
        print(f"警告: 未找到对应的图片文件: {image_basename}.*")
    
    return True

def remove_class_from_label(rel_path, class_id, backup_dir):
    """
    从标签文件中删除特定类别的标注行
    
    参数:
    rel_path: 标签文件的相对路径 (如 'train/img_123.txt')
    class_id: 要删除的类别ID
    backup_dir: 备份目录
    
    返回:
    bool: 是否成功修改了文件
    """
    parts = rel_path.split('/')
    if len(parts) != 2:
        print(f"错误: 无效的文件路径格式 {rel_path}")
        return False
    
    split, label_file = parts
    
    # 构建完整路径
    label_path = os.path.join('VOC2007', split, 'labels', label_file)
    
    if not os.path.exists(label_path):
        print(f"警告: 标签文件不存在: {label_path}")
        return False
    
    try:
        # 读取标签文件
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        # 创建备份
        backup_label_dir = os.path.join(backup_dir, split, "labels")
        os.makedirs(backup_label_dir, exist_ok=True)
        with open(os.path.join(backup_label_dir, label_file), 'w') as f:
            f.writelines(lines)
        
        # 过滤掉指定类别的行
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                if int(parts[0]) != class_id:
                    new_lines.append(line)
        
        # 如果文件内容变化，写回文件
        if len(new_lines) != len(lines):
            with open(label_path, 'w') as f:
                f.writelines(new_lines)
            print(f"已从 {label_path} 中删除类别 {class_id}")
            return True
        else:
            print(f"警告: 文件 {label_path} 中未找到类别 {class_id}")
            return False
    except Exception as e:
        print(f"处理 {label_path} 失败: {e}")
        return False

def convert_labels_to_class(folder_path, target_class):
    """
    批量修改指定文件夹下的标签文件，将所有类别统一改为指定类别
    
    参数:
    folder_path: 包含标签文件的文件夹路径
    target_class: 目标类别ID
    """
    if not os.path.exists(folder_path):
        print(f"错误: 文件夹 {folder_path} 不存在")
        return
    
    # 创建备份目录
    backup_dir = os.path.join("backup_files", "converted_labels", os.path.basename(folder_path))
    os.makedirs(backup_dir, exist_ok=True)
    
    # 查找所有.txt文件
    label_files = glob.glob(os.path.join(folder_path, "*.txt"))
    
    if not label_files:
        print(f"警告: 在 {folder_path} 中未找到.txt文件")
        return
    
    modified_count = 0
    failed_count = 0
    
    for label_path in label_files:
        try:
            # 读取文件
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            # 备份原文件
            file_name = os.path.basename(label_path)
            backup_path = os.path.join(backup_dir, file_name)
            with open(backup_path, 'w') as f:
                f.writelines(lines)
            
            # 修改类别
            new_lines = []
            modified = False
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:  # YOLO格式至少有5个部分（类别 + 4个坐标值）
                    current_class = int(parts[0])
                    if current_class != target_class:
                        parts[0] = str(target_class)
                        new_line = ' '.join(parts) + '\n'
                        new_lines.append(new_line)
                        modified = True
                    else:
                        new_lines.append(line)
                else:
                    new_lines.append(line)
            
            # 如果内容有变化，写回文件
            if modified:
                with open(label_path, 'w') as f:
                    f.writelines(new_lines)
                print(f"已将 {label_path} 中的类别修改为 {target_class}")
                modified_count += 1
        except Exception as e:
            print(f"处理 {label_path} 失败: {e}")
            failed_count += 1
    
    print(f"\n批量修改完成:")
    print(f"- 成功修改: {modified_count} 个文件")
    print(f"- 失败: {failed_count} 个文件")
    print(f"- 原文件已备份到: {backup_dir}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='查找并处理特定类别的YOLO标签文件')
    parser.add_argument('--class_id', type=int, default=11, help='要处理的类别ID')
    parser.add_argument('--find', action='store_true', help='查找包含指定类别的文件')
    parser.add_argument('--process', action='store_true', help='处理标记文件')
    parser.add_argument('--file', type=str, default='class_11_files.txt', help='要处理的标记文件')
    parser.add_argument('--convert', action='store_true', help='批量修改标签文件类别')
    parser.add_argument('--folder', type=str, help='要批量修改的标签文件夹路径')
    
    args = parser.parse_args()
    
    # 批量修改标签文件类别
    if args.convert:
        if not args.folder:
            print("错误: 请使用 --folder 参数指定要处理的文件夹路径")
        else:
            print(f"即将批量修改 {args.folder} 中的标签文件，将所有类别统一改为 {args.class_id}")
            confirm = input("确认操作? (y/n): ")
            if confirm.lower() == 'y':
                convert_labels_to_class(args.folder, args.class_id)
            else:
                print("操作已取消")
        # 如果只执行转换操作，则退出
        if not (args.find or args.process):
            exit()
    
    # 默认操作：如果没有指定操作，则同时执行查找和处理
    if not (args.find or args.process or args.convert):
        args.find = True
        args.process = True
    
    if args.find:
        # 查找文件
        found_files = find_class_files(args.class_id)
        
        # 输出结果
        print("\n查找结果:")
        print(f"共找到 {len(found_files)} 个文件包含类别 {args.class_id}")
        
        if found_files:
            print("\n文件列表:")
            for i, file_info in enumerate(found_files, 1):
                print(f"{i}. [{file_info['split']}] 标签文件: {file_info['label_file']} - 图像文件: {file_info['image_file']}")
            
            # 保存文件列表到txt
            output_file = f"class_{args.class_id}_files.txt"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"# 包含类别 {args.class_id} 的文件列表\n")
                for file_info in found_files:
                    f.write(f"{file_info['split']}/{file_info['label_file']}\n")
            
            print(f"\n文件列表已保存到 {output_file}")
    
    if args.process:
        # 确认操作
        print(f"\n即将处理标记文件 {args.file} 中的文件:")
        print(f"1. 带有'--'标记的文件将被完全删除（包括标签和图片）")
        print(f"2. 带有'-+'标记的文件将删除所有类别为 {args.class_id} 的标注行")
        confirm = input("确认操作? (y/n): ")
        
        if confirm.lower() == 'y':
            process_marked_files(args.file, args.class_id)
        else:
            print("操作已取消")