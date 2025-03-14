import numpy as np
import time
class DataLoader:
    def __init__(self,file_path,exposure=500,gate_info=(0,200)):
        self._file_path=file_path
        self._exposure=exposure
        # self._gate_info=(gate_info[0] + 15, gate_info[1] - 10)
        self._gate_info=(gate_info[0] , gate_info[1] - 10)

        self.load_data()
        self.update_strength_matrix()
        # 这一步是最慢的，不知道还能不能继续优化，再优化可能要上并行计算了
        self.update_tensor_data()

        
    def load_data(self):
        """
        用于加载数据并存放在_matrix_data中
        注意数据格式，如果输入的raw文件无法被4096整除，说明数据采集时不完整或发生损坏
        最终存取的数据实际上是raw文件中的数据按行排列在lines*4096的矩阵中
        其中lines取决于raw文件拍摄时的采集帧数
        """
        try:
            with open(self._file_path, 'rb') as f:
                raw_data = np.frombuffer(f.read(), dtype=np.uint16)
            if len(raw_data) % 4096 != 0:
                raise ValueError("数据长度无法整除 4096，无法构建矩阵。")
            num_lines = len(raw_data) // 4096
            self._matrix_data = raw_data.reshape(num_lines, 4096).astype(float)
        except FileNotFoundError:
            raise FileNotFoundError(f"无法打开文件: {self._file_path}，请检查文件路径是否正确。")
        except PermissionError:
            raise PermissionError(f"没有权限读取文件: {self._file_path}，请检查文件权限。")
        except IOError as e:
            raise IOError(f"读取文件时发生错误: {self._file_path}，错误信息: {str(e)}")
        
    def get_strength_matrix(self):
        """
        计算并返回强度图矩阵
        后续需要检查这里的返回的矩阵是否需要翻转、转置等操作，不再修改绘图函数
        Returns:
            numpy.ndarray: 64x64的强度图矩阵
        """
        if self._exposure > self._matrix_data.shape[0]:
            raise ValueError("曝光时间超过文件大小")

        # 使用切片视图而非复制数据
        selected_data = self._matrix_data[:self._exposure]
        
        # 使用布尔索引一次性计算
        low, high = self._gate_info
        count_under = np.sum((selected_data >= low) & (selected_data < high), axis=0) / self._exposure
        
        # 一次性操作
        return np.round(count_under * 255).astype(int).reshape(64, 64)
    def update_strength_matrix(self):
        self._strength_matrix=self.get_strength_matrix()
        
    
        
    def get_channel_data(self, x_idx=None, y_idx=None, index=None):
        """
        获取指定像素的通道数据，可以通过(x_idx, y_idx)或直接通过index指定像素
        
        Args:
            x_idx: X坐标 (可选)
            y_idx: Y坐标 (可选)
            index: 像素的线性索引 (可选)
        
        Returns:
            tuple: (过滤后的数据, 直方图计数, 光通量值)
        
        Raises:
            ValueError: 如果未提供有效的坐标或索引
        """
        if index is None:
            if x_idx is not None and y_idx is not None:
                index = y_idx * 64 + x_idx
            else:
                raise ValueError("必须提供(x_idx, y_idx)或index中的一个")
        
        data = self._matrix_data[:self._exposure, index]
        low, high = self._gate_info
        
        mask = (data >= low) & (data < high)
        filtered = data[mask]
        
        bin_count = high - low
        hist_counts, _ = np.histogram(filtered, bins=bin_count, range=(low, high))
        
        flux_value = (np.sum(mask) / self._exposure) * 100
        
        return filtered, hist_counts, flux_value
                
    def get_tensor_data(self):
        """
        使用高效向量化操作计算所有像素的直方图数据，组织为三维张量
        没再用逐个像素点扫描然后调用get_channel_data的方式了
        这样更快一点，numpyNB

        Returns:
            numpy.ndarray: 形状为(64, 64, gate_length)的三维张量
        """
        low, high = self._gate_info
        gate_length = high - low

        data = self._matrix_data[:self._exposure].reshape(self._exposure, 64, 64)

        # 创建结果张量
        result_tensor = np.zeros((64, 64, gate_length), dtype=int)

        # 对于数据范围内的每个值，计算其在每个位置出现的次数
        for val in range(low, high):
            mask = (data == val)
            counts = np.sum(mask, axis=0)  # 形状为 [64, 64]
            result_tensor[:, :, val - low] = counts

        return result_tensor
    def update_tensor_data(self):
        self._tensor_data=self.get_tensor_data()
        
        
def test():
    import DataLoader as dl

    file_path = '/Users/ming/Documents/PythonCode/Coates/data/2025-01-17_15-09-54_Delay-0_Width-200.raw'
    data_loader = dl.DataLoader(file_path, exposure=9600, gate_info=(0, 200))

if __name__ == '__main__':
    test()