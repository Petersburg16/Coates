# 在Jupyter笔记本中使用
from IPython.display import display
import src.DataLoader as dl
file_path = r'E:\Mingle\PythonCode\Coates\data\2025-01-17_16-31-55_Delay-0_Width-200.raw'
# file_path = '/Users/ming/Documents/PythonCode/Coates/data/2025-01-17_15-09-54_Delay-0_Width-200.raw'
data_loader = dl.DataChecker(file_path, exposure=9600, gate_info=(0, 200))
# 绘制单个像素的点云
point_cloud_single = data_loader.draw_point_cloud()
point_cloud_single.show()



# 第一，coates估计器输出的光场不是归一化的分布，直接*exposure会爆，继续修改仿真的结果,要参考update_coates_estimation(中生成直方图的思路
# 第二，软件删去前10个bin是有问题的
