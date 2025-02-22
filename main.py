import src.GatingAlgo as ga

# 主程序
if __name__ == "__main__":

    # 参数设置
    num_bins = 100
    pulse_pos = 40
    pulse_width = 3
    signal_strength = 0.6
    bg_strength = 0.03
    num_cycles = 5000
    
    # 创建GatingAlgorithm实例
    gating_algo = ga.GatingAlgorithm(num_bins, pulse_pos, pulse_width, signal_strength, bg_strength)
    
  
    
    # 模拟理想泊松检测
    ideal_hist = gating_algo.simulate_ideal_poisson(num_cycles)
    
    # 模拟同步SPAD检测
    sync_hist = gating_algo.simulate_sync_spad(num_cycles)
    
    # 新增计算检测概率
    Pi = gating_algo.calculate_detection_probability()
    
    # 修改绘图调用
    gating_algo.plot_sync_results(ideal_hist, sync_hist, Pi)