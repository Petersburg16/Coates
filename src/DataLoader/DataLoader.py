import numpy as np
import time
class DataLoader:
    """
    A class to load data from raw file and process it
    """
    def __init__(self,file_path,exposure=500,gate_info=(0,200)):
        """
        Init method for DataLoader, loads data from a raw file and processes it.
        Args:
            file_path: str, the path of the raw file, needs to be adjusted for different OS.
            exposure: int, the exposure time of the data, limited by the raw file settings and data size.
            gate_info: tuple, the gate information of the data, including gate begin and gate end. Note that the second value differs from the raw file's setting method (gate width).
        """
        self._file_path=file_path
        self._exposure=exposure
        self._gate_info=(gate_info[0] , gate_info[1] - 10)

        self.load_data()
        self.update_strength_matrix()
        self.update_tensor_data()

        
    def load_data(self):
        """
        Load data from the raw file and reshape it into a matrix.
        Ensure that the data stored in the raw file is in uint16 format. 
        This method will automatically reshape the data into a matrix.
        
        self._matrix_data: numpy.ndarray, the data matrix loaded from the raw file(num_lines,4096), where each frame is stored as a single line.
        """
        try:
            with open(self._file_path, 'rb') as f:
                raw_data = np.frombuffer(f.read(), dtype=np.uint16)
            if len(raw_data) % 4096 != 0:
                raise ValueError("Data length is not divisible by 4096. Please check the file type or file integrity.")
            num_lines = len(raw_data) // 4096
            self._matrix_data = raw_data.reshape(num_lines, 4096).astype(float)
        except FileNotFoundError:
            raise FileNotFoundError(f"Unable to open file: {self._file_path}. Please check if the file path is correct.")
        except PermissionError:
            raise PermissionError(f"No permission to read the file: {self._file_path}. Please check the file permissions.")
        except IOError as e:
            raise IOError(f"An error occurred while reading the file: {self._file_path}. Error message: {str(e)}")
        

    def update_strength_matrix(self):
        self._strength_matrix=self.get_strength_matrix()
    def update_tensor_data(self):
        self._tensor_data=self.get_tensor_data()
        
    def get_strength_matrix(self):

        """
        Returns the strength matrix of the data, which is a 64x64 matrix.
        The strength matrix is calculated by counting the number of data points in each pixel that fall within the gate range.

        Further checks are needed to determine whether the returned matrix requires flipping, transposing, or other operations, without modifying the plotting function.
        Returns:
            numpy.ndarray: A 64x64 intensity matrix.
        """
        if self._exposure > self._matrix_data.shape[0]:
            raise ValueError("Exposure time exceeds the number of frames in the data.")
        selected_data = self._matrix_data[:self._exposure]
        low, high = self._gate_info
        count_under = np.sum((selected_data >= low) & (selected_data < high), axis=0) / self._exposure
        return np.round(count_under * 255).astype(int).reshape(64, 64)
                
    def get_tensor_data(self):
        """
        Transform the raw data into a 3D tensor, where each pixel's histogram is stored in a 1D array(third dimension).
        Returns:
            numpy.ndarray: A 3D tensor of shape (64, 64, gate_length). Each bin stores the strength of the pixel at that position.
        """
        low, high = self._gate_info
        gate_length = high - low

        data = self._matrix_data[:self._exposure].reshape(self._exposure, 64, 64)
        result_tensor = np.zeros((64, 64, gate_length), dtype=int)


        for val in range(low, high):
            mask = (data == val)
            counts = np.sum(mask, axis=0)  # 形状为 [64, 64]
            result_tensor[:, :, val - low] = counts

        return result_tensor

    def get_tensor_data_coates(self):
        """
        应用Coates估计器处理张量数据中的每个像素直方图
        返回经过Coates估计处理后的三维张量

        Returns:
            numpy.ndarray: 形状为(64, 64, gate_length)的三维张量，每个像素的直方图已通过Coates估计器处理
        """
        from ..GatingSimulator.SPADSimulateEngine import SPADSimulateEngine
        
        low, high = self._gate_info
        gate_length = high - low
        
        # 创建结果张量
        result_tensor = np.zeros((64, 64, gate_length), dtype=float)
        
        # 原始张量数据
        original_tensor = self._tensor_data
        
        # 对每个像素应用Coates估计器
        for x in range(64):
            for y in range(64):
                # 获取当前像素的直方图
                histogram = original_tensor[x, y, :]
                
                # 添加溢出仓，按照Coates估计器的要求
                # 溢出仓数量 = 总曝光次数 - 已检测到的光子计数
                detected_count = np.sum(histogram)
                overflow_count = self._exposure - detected_count
                
                # 创建包含溢出仓的直方图
                histogram_with_overflow = np.append(histogram, overflow_count)
                
                # 应用Coates估计器
                coates_result = SPADSimulateEngine.coates_estimator(histogram_with_overflow)
                
                # 处理结果长度可能与gate_length不同的情况
                result_length = min(len(coates_result), gate_length)
                result_tensor[x, y, :result_length] = coates_result[:result_length]
        
        return result_tensor

    def update_tensor_data_coates(self):
        """
        更新经过Coates估计器处理后的张量数据
        """
        self._tensor_data_coates = self.get_tensor_data_coates()

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
        
def test():
    import DataLoader as dl

    file_path = '/Users/ming/Documents/PythonCode/Coates/data/2025-01-17_15-09-54_Delay-0_Width-200.raw'
    data_loader = dl.DataLoader(file_path, exposure=9600, gate_info=(0, 200))

if __name__ == '__main__':
    test()