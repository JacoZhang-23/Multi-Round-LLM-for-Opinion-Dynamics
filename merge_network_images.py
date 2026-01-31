"""
合并网络演化图片为横向长图
将多个 network_step_XXX.png 拼接成一张横向长图，只保留最右侧的图例
"""

import os
import argparse
from PIL import Image, ImageDraw, ImageFont
import glob


def merge_network_images(output_dir, steps=None, output_filename="network_evolution_merged.png"):
    """
    将网络演化图片拼接成横向长图
    
    参数:
        output_dir: 包含 network_step_XXX.png 的目录路径
        steps: 要拼接的步骤列表，例如 [1, 2, 3, 4, 5]。如果为 None，则自动检测所有图片
        output_filename: 输出文件名
    """
    viz_dir = os.path.join(output_dir, "visualizations")
    frames_dir = os.path.join(viz_dir, "network_frames")
    
    # 优先使用 network_frames 子目录，如果不存在则使用 visualizations 目录
    if os.path.exists(frames_dir):
        search_dir = frames_dir
    elif os.path.exists(viz_dir):
        search_dir = viz_dir
    else:
        print(f"❌ 错误: 找不到可视化目录 {viz_dir}")
        return
    
    # 如果未指定步骤，自动检测所有 network_step_XXX.png
    if steps is None:
        pattern = os.path.join(search_dir, "network_step_*.png")
        all_files = sorted(glob.glob(pattern))
        if not all_files:
            print(f"❌ 错误: 在 {search_dir} 中找不到 network_step_*.png 文件")
            return
        # 提取步骤编号
        steps = []
        for f in all_files:
            basename = os.path.basename(f)
            step_str = basename.replace("network_step_", "").replace(".png", "")
            try:
                steps.append(int(step_str))
            except ValueError:
                continue
        steps.sort()
    
    if not steps:
        print("❌ 错误: 没有指定要拼接的步骤")
        return
    
    print(f"📊 准备拼接步骤: {steps}")
    
    # 加载所有图片
    images = []
    for step in steps:
        img_path = os.path.join(search_dir, f"network_step_{step:03d}.png")
        if not os.path.exists(img_path):
            print(f"⚠️  警告: 找不到文件 {img_path}，跳过")
            continue
        try:
            img = Image.open(img_path)
            images.append(img)
            print(f"   ✓ 加载 network_step_{step:03d}.png ({img.width}x{img.height})")
        except Exception as e:
            print(f"⚠️  警告: 无法加载 {img_path}: {e}")
    
    if not images:
        print("❌ 错误: 没有成功加载任何图片")
        return
    
    # 获取图片尺寸（假设所有图片尺寸相同）
    img_width, img_height = images[0].size
    
    # 裁切更多以去除右侧分隔线和部分图例
    # 只保留最右侧图的完整图例
    legend_width = int(img_width * 0.17)  # 从 0.15 增加到 0.17，裁切更多
    main_plot_width = img_width - legend_width  # 主图区域宽度
    
    # 计算合并后的总宽度
    # 前 n-1 张图只保留主图区域，最后一张保留完整（包括图例）
    total_width = main_plot_width * (len(images) - 1) + img_width
    total_height = img_height
    
    print(f"\n📐 图片信息:")
    print(f"   单张图片尺寸: {img_width}x{img_height}")
    print(f"   估计图例宽度: {legend_width}px")
    print(f"   主图宽度: {main_plot_width}px")
    print(f"   合并后尺寸: {total_width}x{total_height}")
    
    # 创建新的空白画布
    merged_image = Image.new('RGB', (total_width, total_height), (255, 255, 255))
    
    # 拼接图片
    current_x = 0
    labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']  # 支持最多10张图片
    
    for i, img in enumerate(images):
        if i < len(images) - 1:
            # 前 n-1 张图片：裁剪掉右侧图例
            cropped = img.crop((0, 0, main_plot_width, img_height))
            merged_image.paste(cropped, (current_x, 0))
            current_x += main_plot_width
            print(f"   ✓ 拼接第 {i+1}/{len(images)} 张（裁剪图例）- 标签 {labels[i]}")
        else:
            # 最后一张图片：保留完整（包括图例）
            merged_image.paste(img, (current_x, 0))
            print(f"   ✓ 拼接第 {i+1}/{len(images)} 张（保留图例）- 标签 {labels[i]}")
    
    # === 在每个网络图下方添加标签 A, B, C... ===
    draw = ImageDraw.Draw(merged_image)
    
    # 尝试加载字体，如果失败则使用默认字体
    try:
        # macOS 常见字体路径 - 使用加粗字体
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 100)  # 增大到100
    except:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/HelveticaNeue.ttc", 100)
        except:
            try:
                font = ImageFont.truetype("/Library/Fonts/Arial Bold.ttf", 100)
            except:
                # 使用默认字体
                font = ImageFont.load_default()
    
    # 为每张图片添加标签
    for i in range(len(images)):
        if i < len(images) - 1:
            # 前 n-1 张图的中心位置
            label_x = main_plot_width * i + main_plot_width // 2
        else:
            # 最后一张图的中心位置（考虑图例）
            label_x = main_plot_width * i + (img_width - legend_width) // 2
        
        label_y = total_height - 80  # 调整到距离底部80像素
        label_text = labels[i]
        
        # 获取文本边界框以居中对齐
        bbox = draw.textbbox((0, 0), label_text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # 居中绘制文本
        text_x = label_x - text_width // 2
        text_y = label_y - text_height // 2
        
        # 不绘制边框，只绘制加粗文本
        draw.text((text_x, text_y), label_text, fill='black', font=font)
        print(f"   ✓ 添加标签 {label_text} at ({label_x}, {label_y})")
    
    # 保存合并后的图片
    output_path = os.path.join(viz_dir, output_filename)
    merged_image.save(output_path, dpi=(300, 300), quality=95)
    print(f"\n✅ 合并完成! 保存到: {output_path}")
    print(f"   最终尺寸: {merged_image.width}x{merged_image.height}")


def main():
    parser = argparse.ArgumentParser(
        description="将网络演化图片拼接成横向长图",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 拼接所有步骤（默认）
  python merge_network_images.py simulation_20251020_101549
  
  # 只拼接步骤 1, 3, 5
  python merge_network_images.py simulation_20251020_101549 --steps 1 3 5
  
  # 指定输出文件名
  python merge_network_images.py simulation_20251020_101549 --output network_135.png --steps 1 3 5
        """
    )
    
    parser.add_argument(
        'simulation_dir',
        help='模拟输出目录名称（例如 simulation_20251020_101549）或完整路径'
    )
    
    parser.add_argument(
        '--steps',
        type=int,
        nargs='+',
        default=None,
        help='要拼接的步骤编号（例如: --steps 1 2 3 4 5）。默认拼接所有步骤'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='network_evolution_merged.png',
        help='输出文件名（默认: network_evolution_merged.png）'
    )
    
    args = parser.parse_args()
    
    # 处理目录路径
    if os.path.isabs(args.simulation_dir):
        output_dir = args.simulation_dir
    else:
        # 假设是相对于 data/output 的路径
        output_dir = os.path.join("data", "output", args.simulation_dir)
    
    if not os.path.exists(output_dir):
        print(f"❌ 错误: 找不到目录 {output_dir}")
        return
    
    print(f"🎨 网络演化图片拼接工具")
    print(f"📁 输出目录: {output_dir}")
    
    if args.steps:
        print(f"🔢 指定步骤: {args.steps}")
    else:
        print(f"🔢 自动检测所有步骤")
    
    merge_network_images(output_dir, steps=args.steps, output_filename=args.output)


if __name__ == "__main__":
    main()
