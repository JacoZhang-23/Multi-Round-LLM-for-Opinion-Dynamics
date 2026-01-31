# main.py

"""
Main entry point for the LLM-based Multi-round Dialogue Vaccination Simulation.
"""
import os
import sys
import traceback
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import json

from model import VaxSimulationModel
from analysis import run_all_analyses
from visualize_batch_results import (
    visualize_comparative_trends,
    visualize_belief_distributions,
    generate_network_evolution,
    plot_influence_scatter,
    compute_average_beliefs
)
import numpy as np

# 定义配色方案（与 analysis.py 保持一致）
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'llm': '#2E86AB',
    'vader': '#D62246',
}


def generate_fixed_initial_beliefs(num_agents: int, seed: int = 42):
    """
    生成固定的初始belief，用于所有batch运行
    确保每个agent在所有运行中的初始belief保持一致
    """
    from config import BELIEF_DISTRIBUTION_TYPE, BELIEF_MEANS, BELIEF_STD
    
    np.random.seed(seed)
    random.seed(seed)
    
    mu = BELIEF_MEANS.get(BELIEF_DISTRIBUTION_TYPE, 0.0)
    initial_beliefs = []
    
    for i in range(num_agents):
        belief = float(np.clip(np.random.normal(mu, BELIEF_STD), -1.0, 1.0))
        initial_beliefs.append(belief)
    
    print(f"\n✓ 生成固定初始belief (seed={seed}):")
    print(f"   - 数量: {num_agents}")
    print(f"   - 平均值: {np.mean(initial_beliefs):.3f}")
    print(f"   - 标准差: {np.std(initial_beliefs):.3f}")
    print(f"   - 范围: [{min(initial_beliefs):.3f}, {max(initial_beliefs):.3f}]")
    
    return initial_beliefs


def visualize_results(output_dir: str, num_agents: int):
    """
    Generates final, advanced visualizations for comparing belief update mechanisms.
    Enhanced with better styling consistent with analysis module.
    """
    print("\n📊 Generating Final Comparative Visualizations...")
    viz_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)

    try:
        model_df = pd.read_csv(os.path.join(output_dir, "model_data.csv"), index_col=0)
        model_df.index.name = 'Step'

        # --- Plot 1: Comparative Belief Trend Plot (Enhanced) ---
        fig, ax = plt.subplots(figsize=(14, 8), dpi=100)
        
        # 绘制趋势线
        ax.plot(model_df.index, model_df['Average_Belief_LLM'], 
               marker='o', markersize=8, linewidth=2.5, 
               color=COLORS['llm'], label='LLM Self-Score Driven', alpha=0.85)
        ax.plot(model_df.index, model_df['Average_Belief_VADER'], 
               marker='s', markersize=8, linewidth=2.5, 
               color=COLORS['vader'], label='VADER Sentiment Driven', alpha=0.85)
        
        # 添加参考线
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
        
        # 设置标题和标签
        ax.set_title('Comparison of Average Belief Evolution', 
                    fontsize=18, fontweight='bold', pad=20)
        ax.set_xlabel('Simulation Step', fontsize=14, fontweight='600')
        ax.set_ylabel('Average Belief Score', fontsize=14, fontweight='600')
        ax.set_ylim(-1.05, 1.05)
        
        # 美化网格和图例
        ax.grid(True, which='both', linestyle='--', linewidth=0.6, alpha=0.4)
        ax.set_axisbelow(True)
        legend = ax.legend(fontsize=12, loc='best', frameon=True, 
                         shadow=True, fancybox=True)
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_alpha(0.9)
        
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, "comparative_belief_trends.png"), 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        # --- Plot 2: 2x2 Belief Distribution Matrix (Enhanced) ---
        with open(os.path.join(output_dir, "agent_profiles.json"), 'r') as f:
            profiles_data = json.load(f)

        fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharex=True, sharey=True, dpi=100)

        # Extract initial and final beliefs
        initial_beliefs_llm = [p['belief_history'][0] for p in profiles_data]
        final_beliefs_llm = [p['belief_history'][-1] for p in profiles_data]
        initial_beliefs_vader = [p['belief_vader_history'][0] for p in profiles_data]
        final_beliefs_vader = [p['belief_vader_history'][-1] for p in profiles_data]

        num_bins = max(10, num_agents // 2)
        final_step = len(profiles_data[0]['belief_history']) - 1

        # Plot A: Initial Distribution (LLM method)
        sns.histplot(initial_beliefs_llm, kde=True, bins=num_bins, 
                    color='gray', ax=axes[0, 0], alpha=0.6, line_kws={'linewidth': 2})
        axes[0, 0].set_title(f'LLM Self-Score Method (Step 0)', 
                           fontsize=14, fontweight='bold', pad=10)
        axes[0, 0].set_ylabel('Count', fontsize=12, fontweight='600')

        # Plot C: Final Distribution (LLM method)
        sns.histplot(final_beliefs_llm, kde=True, bins=num_bins, 
                    color=COLORS['llm'], ax=axes[1, 0], alpha=0.7, line_kws={'linewidth': 2})
        axes[1, 0].set_title(f'LLM Self-Score Method (Step {final_step})', 
                           fontsize=14, fontweight='bold', pad=10)
        axes[1, 0].set_xlabel('Belief Score', fontsize=12, fontweight='600')
        axes[1, 0].set_ylabel('Count', fontsize=12, fontweight='600')

        # Plot B: Initial Distribution (VADER method)
        sns.histplot(initial_beliefs_vader, kde=True, bins=num_bins, 
                    color='gray', ax=axes[0, 1], alpha=0.6, line_kws={'linewidth': 2})
        axes[0, 1].set_title(f'VADER Sentiment Method (Step 0)', 
                           fontsize=14, fontweight='bold', pad=10)

        # Plot D: Final Distribution (VADER method)
        sns.histplot(final_beliefs_vader, kde=True, bins=num_bins, 
                    color=COLORS['vader'], ax=axes[1, 1], alpha=0.7, line_kws={'linewidth': 2})
        axes[1, 1].set_title(f'VADER Sentiment Method (Step {final_step})', 
                           fontsize=14, fontweight='bold', pad=10)
        axes[1, 1].set_xlabel('Belief Score', fontsize=12, fontweight='600')

        # 统一设置所有子图
        for ax in axes.flatten():
            ax.set_xlim(-1.05, 1.05)
            ax.grid(True, linestyle='--', linewidth=0.6, alpha=0.4)
            ax.set_axisbelow(True)
            ax.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)

        fig.suptitle('Belief Distribution Comparison: LLM Self-Score vs. VADER Sentiment', 
                    fontsize=20, fontweight='bold', y=0.995)
        plt.tight_layout(rect=[0, 0.01, 1, 0.99])
        plt.savefig(os.path.join(viz_dir, "belief_distribution_matrix.png"), 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        print(" -> ✓ Comparative visualizations saved.")

    except Exception as e:
        print(f" -> ✗ An error occurred during visualization: {e}")
        traceback.print_exc()


def main():
    """Main function to configure and run the simulation."""
    # Import settings from config
    from config import (
        MAX_STEPS, AGENT_ALPHA, API_KEY, API_URL, MODEL_NAME, MAX_CONCURRENT_CALLS,
        BATCH_RUNS, BELIEF_DISTRIBUTION_TYPE
    )

    print("\n" + "="*50)
    print("🔬 Starting LLM-based Vaccination Simulation")
    print("="*50 + "\n")

    # Define base directory (project root, one level up from src/)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Create output directory with absolute path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, "data", "output", f"simulation_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Output directory: {output_dir}")

    # Define absolute paths to workplace CSV files
    population_csv = os.path.join(base_dir, "data", "input", "workplace_36013030400w1_extended_population.csv")
    network_csv = os.path.join(base_dir, "data", "input", "workplace_36013030400w1_extended_network.csv")
    
    # Verify files exist
    if not os.path.exists(population_csv):
        raise FileNotFoundError(f"Population CSV not found: {population_csv}")
    if not os.path.exists(network_csv):
        raise FileNotFoundError(f"Network CSV not found: {network_csv}")
    
    print(f"\n✅ Loading workplace data from CSV files:")
    print(f"   Population: {os.path.basename(population_csv)}")
    print(f"   Network: {os.path.basename(network_csv)}")
    
    # 读取population数据获取agent数量
    import pandas as pd
    pop_df = pd.read_csv(population_csv)
    num_agents = len(pop_df)
    
    # 生成固定的初始belief（所有batch运行使用相同的初始值）
    fixed_initial_beliefs = generate_fixed_initial_beliefs(num_agents, seed=42)
    
    # Run batch simulations and aggregate results
    model_dfs = []
    all_profiles = []
    network_data = None

    for run_idx in range(BATCH_RUNS):
        run_dir = os.path.join(output_dir, f"run_{run_idx + 1:02d}")
        os.makedirs(run_dir, exist_ok=True)

        print(f"\n🔁 Batch Run {run_idx + 1}/{BATCH_RUNS} (belief={BELIEF_DISTRIBUTION_TYPE})")

        model = VaxSimulationModel(
            max_steps=MAX_STEPS,
            agent_alpha=AGENT_ALPHA,
            api_url=API_URL,
            api_key=API_KEY,
            model_name=MODEL_NAME,
            max_concurrent=MAX_CONCURRENT_CALLS,
            use_workplace_data=True,
            population_csv=population_csv,
            network_csv=network_csv,
            fixed_initial_beliefs=fixed_initial_beliefs  # 传入固定初始belief
        )

        model.run_model()
        model.export_results(run_dir)

        model_dfs.append(model.datacollector.get_model_vars_dataframe())
        
        # 收集agent profiles用于后续batch可视化
        with open(os.path.join(run_dir, "agent_profiles.json"), 'r') as f:
            all_profiles.append(json.load(f))
        
        # 只需要保存一次网络数据
        if network_data is None:
            with open(os.path.join(run_dir, "network_data.json"), 'r') as f:
                network_data = json.load(f)

    # Aggregate model metrics across runs (mean + std)
    combined = pd.concat(model_dfs, keys=range(BATCH_RUNS))
    mean_df = combined.groupby(level=1).mean()
    std_df = combined.groupby(level=1).std()

    mean_df.to_csv(os.path.join(output_dir, "model_data_mean.csv"))
    std_df.to_csv(os.path.join(output_dir, "model_data_std.csv"))

    print("\n📈 Saved aggregated results:")
    print(f"   - Mean: {os.path.join(output_dir, 'model_data_mean.csv')}")
    print(f"   - Std:  {os.path.join(output_dir, 'model_data_std.csv')}")

    # ========== 生成批量可视化 ==========
    print("\n" + "="*60)
    print("🎨 生成批量运行可视化")
    print("="*60)
    
    viz_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    # 1. 计算平均belief
    print(f"\n📊 计算平均belief值...")
    avg_beliefs_llm, avg_beliefs_vader = compute_average_beliefs(all_profiles)
    print(f"   ✓ 计算完成: {num_agents} agents × {len(all_profiles[0][0]['belief_history'])} steps")
    
    # 2. 生成各种可视化
    visualize_comparative_trends(output_dir, viz_dir)
    visualize_belief_distributions(all_profiles, viz_dir)
    
    # 3. 生成网络演化（使用平均belief）
    print("\n🎨 生成网络演化可视化（使用平均belief）...")
    generate_network_evolution(
        output_dir,
        network_data,
        avg_beliefs_llm,
        viz_dir
    )
    
    # 4. 生成影响力散点图
    plot_influence_scatter(output_dir, viz_dir)
    
    print("\n" + "="*60)
    print("✅ 所有可视化已完成!")
    print(f"📁 输出目录: {viz_dir}")
    print("="*60)
    
    # 打印最终统计摘要
    final_mean = mean_df.iloc[-1]
    print("\n🏁 Final Summary (averaged over all runs):")
    print(f"   - Number of Agents: {num_agents}")
    print(f"   - Number of Runs: {BATCH_RUNS}")
    print(f"   - Final Avg Belief (LLM): {final_mean['Average_Belief_LLM']:.3f} ± {std_df.iloc[-1]['Average_Belief_LLM']:.3f}")
    print(f"   - Final Avg Belief (VADER): {final_mean['Average_Belief_VADER']:.3f} ± {std_df.iloc[-1]['Average_Belief_VADER']:.3f}")
    print(f"   - Final Vaccination Rate: {final_mean['Vaccination_Rate']:.3%} ± {std_df.iloc[-1]['Vaccination_Rate']:.3%}")
    print(f"   - Final Belief Polarization (LLM): {final_mean['Belief_Std_Dev_LLM']:.3f} ± {std_df.iloc[-1]['Belief_Std_Dev_LLM']:.3f}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ A critical error occurred: {e}")
        traceback.print_exc()
        sys.exit(1)