import os
import random
import shutil
from pathlib import Path
from collections import defaultdict
import argparse

def extract_random_diverse_images_multi_source(
    source_folders: list,
    ratios: list,
    destination_folder: str,
    num_images: int = 100,
    chunk_size: int = 10
):
    """
    从多个源文件夹中按照指定比例随机抽取分散的图片到目标文件夹
    
    参数:
        source_folders: 源图片文件夹路径列表
        ratios: 从各个源文件夹抽取图片的比例列表
        destination_folder: 目标文件夹路径
        num_images: 需要抽取的图片总数
        chunk_size: 每个区块的大小，用于确保图片分散性
    """
    if len(source_folders) != len(ratios):
        raise ValueError("源文件夹数量必须与比例数量一致")
    
    # 确保比例合法并归一化
    if any(r < 0 for r in ratios):
        raise ValueError("所有比例必须为非负数")
    
    total_ratio = sum(ratios)
    if total_ratio == 0:
        raise ValueError("至少有一个比例必须大于0")
        
    normalized_ratios = [r / total_ratio for r in ratios]
    
    # 支持的图片文件扩展名
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
    
    # 确保目标文件夹存在
    os.makedirs(destination_folder, exist_ok=True)
    
    # 收集所有图片路径
    all_images_by_source = []
    for folder in source_folders:
        images = []
        for root, _, files in os.walk(folder):
            for file in files:
                if Path(file).suffix.lower() in image_extensions:
                    images.append(os.path.join(root, file))
        all_images_by_source.append(images)
    
    # 检查每个源文件夹是否有图片
    for i, (folder, images) in enumerate(zip(source_folders, all_images_by_source)):
        if not images:
            print(f"警告: 在 {folder} 中没有找到图片文件")
            normalized_ratios[i] = 0
    
    # 重新归一化比例（如果有文件夹没有图片）
    if sum(normalized_ratios) > 0:
        total = sum(normalized_ratios)
        normalized_ratios = [r / total for r in normalized_ratios]
    
    # 计算每个源文件夹需要抽取的图片数量
    images_per_source = [round(num_images * ratio) for ratio in normalized_ratios]
    
    # 如果舍入导致总数不一致，调整最后一个非零来源的数量
    total_selected = sum(images_per_source)
    if total_selected != num_images:
        diff = num_images - total_selected
        for i in range(len(images_per_source) - 1, -1, -1):
            if images_per_source[i] > 0 or diff > 0:
                images_per_source[i] += diff
                break
    
    # 对每个源文件夹执行抽样
    selected_images = []
    for i, (images, n_images) in enumerate(zip(all_images_by_source, images_per_source)):
        if n_images <= 0 or not images:
            continue
            
        # 调整为可用的最大数量
        n_images = min(n_images, len(images))
        
        # 将图片列表分成多个区块
        random.shuffle(images)  # 先随机打乱图片列表
        chunks = [images[j:j+chunk_size] for j in range(0, len(images), chunk_size)]
        
        # 从每个区块中随机选择图片，直到达到所需数量
        source_selected = []
        chunk_index = 0
        images_per_chunk = max(1, n_images // len(chunks))
        remaining = n_images
        
        while remaining > 0 and chunks:
            chunk = chunks[chunk_index % len(chunks)]
            if not chunk:  # 如果当前区块已经没有图片了
                chunks.pop(chunk_index % len(chunks))
                if not chunks:  # 如果所有区块都为空
                    break
                continue
                
            # 从当前区块中随机选择图片
            img_to_take = min(images_per_chunk, len(chunk), remaining)
            if img_to_take > 0:
                selected = random.sample(chunk, img_to_take)
                source_selected.extend(selected)
                
                # 从区块中移除已选择的图片
                for img in selected:
                    chunk.remove(img)
                    
                remaining -= img_to_take
                
            chunk_index += 1
        
        selected_images.extend([(img_path, source_folders[i]) for img_path in source_selected])
        print(f"从 {source_folders[i]} 中选择了 {len(source_selected)} 张图片 (目标: {n_images})")
    
    # 复制选中的图片到目标文件夹
    copied_count = 0
    source_distribution = defaultdict(int)
    class_distribution = defaultdict(int)
    
    for img_path, source_folder in selected_images:
        # 获取文件名和相对路径，用于分析分布情况
        rel_path = os.path.relpath(img_path, source_folder)
        parent_dir = os.path.dirname(rel_path)
        
        # 记录来源分布
        source_distribution[source_folder] += 1
        
        # 记录类别分布（如果按子目录组织）
        if parent_dir:
            class_distribution[f"{source_folder}:{parent_dir}"] += 1
        
        # 生成唯一的目标文件名（避免同名文件覆盖）
        base_name = os.path.basename(img_path)
        file_name, ext = os.path.splitext(base_name)
        
        # 添加来源标识到文件名
        source_id = os.path.basename(source_folder)
        dest_name = f"{file_name}__{source_id}{ext}"
        dest_path = os.path.join(destination_folder, dest_name)
        
        # 如果文件已存在，添加数字后缀
        counter = 1
        while os.path.exists(dest_path):
            dest_name = f"{file_name}__{source_id}_{counter}{ext}"
            dest_path = os.path.join(destination_folder, dest_name)
            counter += 1
        
        # 复制文件
        try:
            shutil.copy2(img_path, dest_path)
            copied_count += 1
        except Exception as e:
            print(f"复制 {img_path} 时出错: {e}")
    
    print(f"\n成功复制了 {copied_count} 张图片到 {destination_folder}")
    
    # 打印来源分布情况
    print("\n来源分布情况:")
    for source, count in source_distribution.items():
        print(f"{os.path.basename(source)}: {count}张图片 ({count/copied_count*100:.1f}%)")
    
    # 打印类别分布情况
    if class_distribution:
        print("\n详细类别分布情况:")
        for class_name, count in class_distribution.items():
            print(f"{class_name}: {count}张图片")

def extract_random_diverse_images(
    source_folder: str,
    destination_folder: str,
    num_images: int = 100,
    chunk_size: int = 10
):
    """
    原始单源文件夹抽取函数的包装，用于向后兼容
    """
    extract_random_diverse_images_multi_source(
        [source_folder], 
        [1.0], 
        destination_folder, 
        num_images, 
        chunk_size
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从一个或多个文件夹中随机抽取分散的图片")
    
    # 多源文件夹支持
    parser.add_argument("--source1", "-s1", required=True, help="第一个源图片文件夹路径")
    parser.add_argument("--source2", "-s2", help="第二个源图片文件夹路径")
    parser.add_argument("--ratio1", "-r1", type=float, default=1.0, 
                        help="从第一个源文件夹抽取的比例，默认为1.0")
    parser.add_argument("--ratio2", "-r2", type=float, default=1.0, 
                        help="从第二个源文件夹抽取的比例，默认为1.0")
    
    # 其他参数
    parser.add_argument("--dest", "-d", required=True, help="目标文件夹路径")
    parser.add_argument("--num", "-n", type=int, default=100, help="需要抽取的图片总数，默认为100")
    parser.add_argument("--chunk", "-c", type=int, default=10, 
                        help="每个区块的大小，用于确保图片分散性，默认为10")
    
    args = parser.parse_args()
    
    # 处理源文件夹和比例
    source_folders = [args.source1]
    ratios = [args.ratio1]
    
    if args.source2:
        source_folders.append(args.source2)
        ratios.append(args.ratio2)
    
    extract_random_diverse_images_multi_source(
        source_folders=source_folders,
        ratios=ratios,
        destination_folder=args.dest,
        num_images=args.num,
        chunk_size=args.chunk
    )