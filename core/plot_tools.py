# core/plot_tools.py

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import seaborn as sns
import os

def plot_all_in_one(macro_df, spy_price, strategy_data, tickers):
    """Single figure with 3 stacked subplots: Regime overlay, cumulative returns, heatmap."""
    regime_colors = {
        'Recovery':    '#b0e57c',
        'Overheat':    '#ffe074',
        'Stagflation': '#ff9999',
        'Reflation':   '#a3d2f2',
        'Unclassified':'#d3d3d3'
    }

    # ------------------------ Prep data ------------------------
    gdp_rebased = macro_df['GDP_YoY'] * 100
    gdp_rebased = gdp_rebased / gdp_rebased.iloc[0] * 100

    spy_rebased = spy_price.reindex(macro_df.index).ffill()
    spy_rebased = spy_rebased / spy_rebased.dropna().iloc[0] * 100

    # ------------------------ Create figure ------------------------
    fig, axs = plt.subplots(3, 1, figsize=(12, 8), num="Macro Strategy Overview")

    # -------- Plot 1: Regime Overlay --------
    ax1 = axs[0]
    ax1.plot(macro_df.index, macro_df['CPI_YoY'] * 100, color='red', label='CPI YoY (%)', linewidth=1.5)
    ax1.set_ylabel('CPI YoY (%)', color='red')
    ax1.tick_params(axis='y', labelcolor='red')

    ax2 = ax1.twinx()
    ax2.plot(gdp_rebased.index, gdp_rebased, label='GDP YoY (Rebased)', linewidth=1.5)
    ax2.plot(spy_rebased.index, spy_rebased, '--', linewidth=1.2, label='SPY Price (Rebased)')
    ax2.set_ylabel('Rebased Index (Start = 100)', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')

    current_regime, start_regime = None, None
    for date, reg in macro_df['Regime'].items():
        if reg != current_regime:
            if current_regime is not None:
                ax1.axvspan(start_regime, date, color=regime_colors.get(current_regime, '#eee'), alpha=0.25)
            start_regime, current_regime = date, reg
    if current_regime:
        ax1.axvspan(start_regime, macro_df.index[-1], color=regime_colors.get(current_regime, '#eee'), alpha=0.25)

    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax1.grid(True)
    ax1.set_title('Macro Regimes with CPI, GDP YoY and SPY (Rebased)', fontsize=10)

    regime_patches = [mpatches.Patch(color=regime_colors[r], label=r) for r in regime_colors]
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2 + regime_patches, labels1 + labels2 + [p.get_label() for p in regime_patches], loc='upper left', fontsize=8)

    # -------- Plot 2: Cumulative Returns --------
    ax3 = axs[1]
    ax3.plot(strategy_data.index, strategy_data['Cumulative_Strategy_Net'], label='Macro Regime Strategy', linewidth=2)
    ax3.plot(strategy_data.index, strategy_data['SPY'].add(1).cumprod(), '--', label='SPY Buy & Hold', color='gray')
    ax3.set_title('Strategy vs SPY', fontsize=10)
    ax3.set_ylabel('Growth of $1')
    ax3.legend(fontsize=8)
    ax3.grid(True)

    # -------- Plot 3: Heatmap --------
    avg_ret_by_regime = strategy_data.groupby('Regime')[tickers].mean() * 12 * 100  # Annualized
    sns.heatmap(avg_ret_by_regime.T, annot=True, fmt=".2f", cmap="YlOrRd", ax=axs[2], cbar_kws={'shrink': 0.8})
    axs[2].set_title("Annualized Avg Return by Asset and Regime (%)", fontsize=10)
    axs[2].set_ylabel("Asset")
    axs[2].set_xlabel("Macro Regime")

    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)  # 👈 Ensure the window gets drawn

    # Ensure the output directory exists in the project folder
    output_dir = os.path.join(os.getcwd(), "output")  # Create 'output' in the current working directory
    os.makedirs(output_dir, exist_ok=True)

    # Save the plot to the output directory
    output_path = os.path.join(output_dir, "macro_strategy_overview.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')  # Save the plot with high resolution

    print(f"Plot saved to {output_path}")

    # Show the plot
    plt.show(block=False)
    plt.pause(0.1)  # 👈 Ensure the window gets drawn
    input("🔚 Press Enter to close the chart and quit...")
    plt.close('all')
