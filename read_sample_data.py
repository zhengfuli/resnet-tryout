# -*- coding:utf-8 -*-
import numpy as np

class ReadData():

    def __init__(self, sample_count):
        self.file_incomplete_data = open("./data/sample_data0723.npy", "rb")
        self.sample_count = sample_count
        self.total_data = self.read_data()

    def read_data(self):
        total_data = []
        for i in range(self.sample_count):
            data = np.load(self.file_incomplete_data)
            total_data.append(data)
        self.file_incomplete_data.close()
        return total_data

# 测试代码
if __name__ == '__main__':
    read_data = ReadData(100000)
    # total_data = read_data.read_data()
    # print(read_data.total_data)
    # print("data_number:", len(read_data.total_data))
    print(read_data.total_data[10000])
