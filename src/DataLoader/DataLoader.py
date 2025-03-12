import numpy as np

class DataLoader:
    def __init__(self,file_path,exposure=500,gate_info=(0,200)):
        self._file_path=file_path
        self._exposure=exposure
        # self._gate_info=(gate_info[0] + 15, gate_info[1] - 10)
        self._gate_info=(gate_info[0] , gate_info[1] - 10)

        self.load_data()
        self.update_strength_matrix()
        
    def load_data(self):
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
        
        
        
    def get_channel_data(self, x_idx, y_idx):
        """
        获取指定像素的通道数据
        
        Args:
            x_idx: X坐标
            y_idx: Y坐标
        
        Returns:
            tuple: (过滤后的数据, 直方图计数, 光通量值)
        """
        index = y_idx * 64 + x_idx
        data = self._matrix_data[:self._exposure, index]
        low, high = self._gate_info
        filtered = data[(data >= low) & (data < high)]
        hist_counts, bin_edges = np.histogram(filtered, bins=range(low, high + 1))
        flux_value = (len(filtered) / self._exposure) * 100
        
        return filtered, hist_counts, flux_value
    


            
            
            
            
