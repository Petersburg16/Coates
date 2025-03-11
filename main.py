import src.GatingAlgo as ga

def main():
    # SingleGaussian参数设置
    num_bins = 100
    pulse_pos = 40
    pulse_width = 3
    signal_strength = 0.6
    bg_strength = 0.04
    num_cycles = 5000

    # 创建SingleGaussian实例
    gating_algo = ga.SingleGaussian(num_bins, pulse_pos, pulse_width, signal_strength, bg_strength, num_cycles)
    
    # 使用MLE估计参数并打印结果
    mle_params = gating_algo.estimate_parameters_mle()

    
    # 绘制原始直方图
    gating_algo.plot_hist_plotly()
    
    # 绘制MLE估计结果对比图
    gating_algo.plot_mle_comparison()
    # 绘制原始直方图和MLE估计结果的合并图
    gating_algo.plot_combined_hist_mle()
    

    
    # DoubleGaussian参数设置
    num_bins = 100
    pulse_pos1 = 40
    pulse_pos2 = 70
    pulse_width1 = 2
    pulse_width2 = 2
    signal_strength1 = 0.6
    signal_strength2 = 0.1
    bg_strength = 0.04
    num_cycles = 5000

    # 创建DoubleGaussian实例
    gating_algo_double = ga.DoubleGaussian(num_bins, pulse_pos1, pulse_pos2, pulse_width1, pulse_width2, 
                                          signal_strength1, signal_strength2, bg_strength, num_cycles)
    
    # 使用MLE估计双高斯参数并打印结果
    mle_params_double = gating_algo_double.estimate_parameters_mle()
    
    # 绘制原始直方图
    gating_algo_double.plot_hist_plotly()
    
    # 绘制MLE估计结果对比图
    gating_algo_double.plot_mle_comparison()
    
    # 绘制原始直方图和MLE估计结果的合并图
    gating_algo_double.plot_combined_hist_mle()

if __name__ == '__main__':
    main()
