from src import SPADSimulator as sim 
double_gaussian_params = {
        "pulse_pos1": 40,
        "pulse_pos2": 70,
        "pulse_width1": 0.5,
        "pulse_width2": 1,
        "signal_strength1": 2,
        "signal_strength2": 0,
        "bg_strength": 0.01,
        "total_strength": 1,
}
DG=sim.DoubleGaussian(double_gaussian_params)
DG.plot_test()



