"""
根据已有的模拟数据重新生成网络演化图片
使用 network_data.json 和 agent_profiles.json 生成高质量的网络可视化
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import networkx as nx
from matplotlib.patches import FancyBboxPatch
from tqdm import tqdm


def regenerate_network_frames(output_dir):
    """
    根据已保存的数据重新生成网络演化图片
    
    参数:
        output_dir: 模拟输出目录路径
    """
    print(f"🎨 重新生成网络演化可视化")
    print(f"📁 输出目录: {output_dir}")
    
    # 加载网络数据
    network_file = os.path.join(output_dir, "network_data.json")
    profiles_file = os.path.join(output_dir, "agent_profiles.json")
    
    if not os.path.exists(network_file):
        print(f"❌ 错误: 找不到 {network_file}")
        return
    
    if not os.path.exists(profiles_file):
        print(f"❌ 错误: 找不到 {profiles_file}")
        return
    
    # 读取数据
    with open(network_file, 'r') as f:
        network_data = json.load(f)
    
    with open(profiles_file, 'r') as f:
        agent_profiles = json.load(f)
    
    print(f"✓ 加载网络数据: {len(network_data['edges'])} 条边")
    print(f"✓ 加载智能体配置: {len(agent_profiles)} 个智能体")
    
    # 转换布局数据
    network_layout = {int(k): tuple(v) for k, v in network_data['layout'].items()}
    network_edges = [tuple(e) for e in network_data['edges']]
    
    # 创建输出目录
    viz_dir = os.path.join(output_dir, "visualizations", "network_frames")
    os.makedirs(viz_dir, exist_ok=True)
    
    # 确定步骤数
    max_steps = len(agent_profiles[0]['belief_history'])
    print(f"✓ 检测到 {max_steps} 个时间步")
    
    # 设置颜色映射
    cmap = mcolors.LinearSegmentedColormap.from_list(
        'belief_cmap',
        ['#D32F2F', '#F57C00', '#FDD835', '#9CCC65', '#388E3C']
    )
    
    print(f"\n🎬 开始生成 {max_steps} 帧...")
    
    # 为每个步骤生成图片
    for step in tqdm(range(max_steps), desc="生成网络帧"):
        # 收集当前步骤的信念值
        current_beliefs = {}
        for agent in agent_profiles:
            agent_id = agent['agent_id']
            belief = agent['belief_history'][step]
            current_beliefs[agent_id] = belief
        
        # 创建图形 - 增大尺寸让网络图更大
        fig, ax = plt.subplots(figsize=(20, 14), dpi=150)
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
        
        # 将信念值映射到颜色
        norm = mcolors.Normalize(vmin=-1, vmax=1)
        node_colors = [cmap(norm(current_beliefs[node])) for node in sorted(current_beliefs.keys())]
        
        # 创建网络图
        G = nx.Graph()
        G.add_nodes_from(current_beliefs.keys())
        G.add_edges_from(network_edges)
        
        # 绘制边（浅灰色，细线，带透明度）
        nx.draw_networkx_edges(
            G, network_layout, ax=ax,
            edge_color='#9E9E9E',
            width=2.0,
            alpha=0.35,
            style='solid'
        )
        
        # 绘制节点（带阴影效果）
        # 先绘制阴影层
        nx.draw_networkx_nodes(
            G, network_layout, ax=ax,
            node_color='black',
            node_size=1100,  # 从 850 增加到 1100
            alpha=0.15
        )
        
        # 再绘制主节点
        nx.draw_networkx_nodes(
            G, network_layout, ax=ax,
            node_color=node_colors,
            node_size=1200,  # 从 900 增加到 1200
            edgecolors='#212121',
            linewidths=3.0,  # 从 2.5 增加到 3.0
            alpha=0.95
        )
        
        # 绘制节点标签（Agent ID）- 字体大小增加
        labels = {node: f"{node}" for node in current_beliefs.keys()}
        nx.draw_networkx_labels(
            G, network_layout, labels, ax=ax,
            font_size=22,  # 从 18 增加到 22
            font_weight='bold',
            font_color='black',
            font_family='sans-serif'
        )
        
        # 添加标题 - 字体大小增加0.5倍
        title_text = f'Agent Network - Belief Evolution (Step {step})'
        ax.text(
            0.5, 1.02, title_text,
            transform=ax.transAxes,
            fontsize=36,  # 从 24 增加到 36 (1.5倍)
            fontweight='bold',
            ha='center',
            va='bottom'
        )
        
        # 移除坐标轴
        ax.axis('off')
        ax.margins(0.1)
        ax.set_aspect('equal')
        
        # === 创建颜色条图例（右侧）===
        cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.70])
        
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        
        cbar = plt.colorbar(sm, cax=cbar_ax, orientation='vertical')
        cbar.set_label(
            'Belief Score',
            rotation=270,
            labelpad=28,
            fontsize=24,  # 从 16 增加到 24 (1.5倍)
            fontweight='bold'
        )
        cbar.ax.tick_params(labelsize=18, width=2, length=6)  # 从 12 增加到 18 (1.5倍)
        
        # 设置颜色条刻度
        cbar.set_ticks([-1.0, -0.5, 0.0, 0.5, 1.0])
        cbar.set_ticklabels([
            '-1.0\nStrongly\nOppose',
            '-0.5\nOppose',
            '0.0\nNeutral',
            '+0.5\nSupport',
            '+1.0\nStrongly\nSupport'
        ])
        
        # 添加颜色条边框
        cbar.outline.set_edgecolor('#424242')
        cbar.outline.set_linewidth(2)
        
        # === 不再添加左侧统计信息框，让网络图更大 ===
        # 统计信息已删除，网络图将占据更多空间
        
        # 保存图像
        filename = os.path.join(viz_dir, f"network_step_{step:03d}.png")
        plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
    
    print(f"\n✅ 完成! 所有 {max_steps} 帧已保存到:")
    print(f"   {viz_dir}")
    print(f"\n💡 现在可以使用 merge_network_images.py 合并图片:")
    print(f"   python merge_network_images.py {os.path.basename(output_dir)} --steps 1 2 3 4 5")


def main():
    parser = argparse.ArgumentParser(
        description="根据已有数据重新生成网络演化图片",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 重新生成指定模拟的网络图片
  python regenerate_network_frames.py simulation_20251029_093505
  
  # 使用完整路径
  python regenerate_network_frames.py data/output/simulation_20251029_093505
        """
    )
    
    parser.add_argument(
        'simulation_dir',
        help='模拟输出目录名称（例如 simulation_20251029_093505）或完整路径'
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
    
    regenerate_network_frames(output_dir)


if __name__ == "__main__":
    main()
