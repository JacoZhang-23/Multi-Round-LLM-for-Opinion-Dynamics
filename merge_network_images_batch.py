"""
合并网络演化图片为横向长图
将多个 network_step_XXX.png 拼接成一张横向长图，只保留最右侧的图例
"""

import os
import sys
import argparse
from PIL import Image, ImageDraw, ImageFont
import glob


def merge_network_images(simulation_dir, steps=None, output_filename="network_evolution_merged.png"):
    """
    将网络演化图片拼接成横向长图
    
    参数:
        simulation_dir: 模拟输出目录路径
        steps: 要拼接的步骤列表，例如 [0, 2, 4, 6, 8, 10]。如果为 None，则自动检测所有图片
        output_filename: 输出文件名
    """
    viz_dir = os.path.join(simulation_dir, "visualizations")
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
    
    print(f"\n📊 准备拼接步骤: {steps}")
    
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
    legend_width = int(img_width * 0.17)  # 图例宽度
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
            # 前面的图片只保留主图区域（裁掉右侧图例）
            cropped = img.crop((0, 0, main_plot_width, img_height))
            merged_image.paste(cropped, (current_x, 0))
            current_x += main_plot_width
        else:
            # 最后一张图片保留完整（包括图例）
            merged_image.paste(img, (current_x, 0))
    
    # 保存合并后的图片
    output_path = os.path.join(viz_dir, output_filename)
    merged_image.save(output_path, quality=95)
    
    print(f"\n✅ 图片拼接完成!")
    print(f"📁 输出文件: {output_path}")
    print(f"📐 最终尺寸: {merged_image.width}x{merged_image.height}")


def main():
    parser = argparse.ArgumentParser(
        description="合并网络演化图片为横向长图",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 拼接所有帧
  python merge_network_images_batch.py simulation_20260115_120020
  
  # 拼接指定步骤（例如：0, 2, 5, 10）
  python merge_network_images_batch.py simulation_20260115_120020 --steps 0 2 5 10
  
  # 使用完整路径
  python merge_network_images_batch.py data/output/simulation_20260115_120020 --steps 0 5 10
        """
    )
    
    parser.add_argument(
        'simulation_dir',
        help='模拟输出目录名称或完整路径'
    )
    
    parser.add_argument(
        '--steps',
        nargs='+',
        type=int,
        help='要拼接的步骤列表，例如: --steps 0 2 4 6 8 10。如果不指定，则拼接所有图片'
    )
    
    parser.add_argument(
        '--output',
        default='network_evolution_merged.png',
        help='输出文件名（默认: network_evolution_merged.png）'
    )
    
    args = parser.parse_args()
    
    # 处理目录路径
    if os.path.isabs(args.simulation_dir):
        simulation_dir = args.simulation_dir
    else:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        simulation_dir = os.path.join(base_dir, "data", "output", args.simulation_dir)
    
    if not os.path.exists(simulation_dir):
        print(f"❌ 错误: 找不到目录 {simulation_dir}")
        return
    
    print("="*60)
    print("🖼️  网络演化图片合并工具")
    print("="*60)
    print(f"📁 模拟目录: {simulation_dir}")
    
    merge_network_images(simulation_dir, args.steps, args.output)
    
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
