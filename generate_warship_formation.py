# -*- coding:utf-8 -*-

import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
import random
import settings

# 两列船的列数的间隔
delta_column = 5
#调整第二个阵型的拐点位置
change_direction_point = 5
#第五种阵型的每列拥有的舰船数
each_column_num = 6
#一排中两艘船的列数间隔
each_warship_delta_column = 2
#整体向右移动的列数,最大只能向右平移25，否则移动超出活动范围
remove_column_num = 15
# 是否进行阵型测试
test_flag = False


def generate_warships_formation():
    warships_formation = dict()
    if not test_flag:
        warships_formation[0] = remove_warship_position(_first_formation())
        warships_formation[1] = remove_warship_position(_second_formation())
        warships_formation[2] = remove_warship_position(_five_formation())
        warships_formation[3] = remove_warship_position(_seven_formation())
        warships_formation[4] = remove_warship_position(_eight_formation())
        warships_formation[5] = remove_warship_position(_nine_formation())
        warships_formation[6] = remove_warship_position(_reset_first_formation())
        warships_formation[7] = remove_warship_position(_reset_second_formation())
        warships_formation[8] = remove_warship_position(_reset_three_formation())
        warships_formation[9] = remove_warship_position(_reset_four_formation())
    else:
        warships_formation[0] = remove_warship_position(_reset_five_formation())
    return warships_formation

def generate_bases_position():
    all_base_position = []
    for i in range(1,5):
        for j in range(10):
            all_base_position.append([-i,j])
    base_position = random.sample(all_base_position, settings.bases_num)
    return np.array(base_position)

def generate_targets_position():
    all_targets_position = []
    for i in range(61,65):
        for j in range(10):
            all_targets_position.append([i,j])
    target_position = random.sample(all_targets_position, settings.targets_num)
    return np.array(target_position)

def remove_warship_position(warship_position):
    position = _transform_formation_to_position(warship_position)
    new_position = []
    for unit in position:
        unit[1] += remove_column_num
        new_position.append(unit)
    new_warship_formation = _transform_position_to_formation(new_position)
    assert np.sum(new_warship_formation) == 30, 'The number of warship less than 30 '
    return new_warship_formation

def _transform_formation_to_position(warship_position):
    row,column = warship_position.shape
    position = []
    for i in range(row):
        for j in range(column):
            if warship_position[i,j] == 1:
                position.append([i,j])
    return position

def _transform_position_to_formation(position):
    warship_formation = np.zeros((10,60))
    for unit in position:
        if unit[0] < 10 and unit[1] < 60:
            warship_formation[unit[0],unit[1]] = 1
    return warship_formation


def _first_formation():
    warship_formation = np.zeros((10,60))
    start_column = 0
    for i in range(3):
        warship_formation[:,start_column] = 1
        start_column += delta_column

    return warship_formation

def _second_formation():
    warship_formation = np.zeros((10,60))
    start_column = change_direction_point * each_warship_delta_column + 1
    for i in range(10):
        warship_formation[i,start_column] = 1
        warship_formation[i,start_column+delta_column] = 1
        warship_formation[i,start_column+2*delta_column] =1
        if i < change_direction_point:
            start_column -= each_warship_delta_column
        else:
            start_column += each_warship_delta_column
    return warship_formation

def _three_formation():
    warship_formation = np.zeros((10,60))
    start_column = 0
    for i in range(10):
        warship_formation[i, start_column] = 1
        warship_formation[i, start_column+delta_column] =1
        warship_formation[i, start_column+2*delta_column] =1
        start_column += each_warship_delta_column
    return warship_formation

def _four_formation():
    warship_formation = np.zeros((10, 60))
    start_column = 10 * each_warship_delta_column
    for i in range(10):
        warship_formation[i, start_column] = 1
        warship_formation[i, start_column + delta_column] = 1
        warship_formation[i, start_column + 2 * delta_column] = 1
        start_column -= each_warship_delta_column
    return warship_formation

def _five_formation():
    warship_formation = np.zeros((10,60))
    start_row = 0
    start_column = 0
    for i in range(30 // each_column_num):
        warship_formation[start_row:(start_row+each_column_num), start_column] =1
        start_row += 1
        start_column +=delta_column
    return warship_formation

def _six_formation():
    warship_formation = np.zeros((10, 60))
    start_column = change_direction_point * each_warship_delta_column +1
    start_column2 = start_column + 2
    start_column3 = 10*each_warship_delta_column + start_column
    for i in range(10):
        warship_formation[i, start_column] = 1
        warship_formation[i, start_column2] = 1
        warship_formation[i, start_column3] = 1
        if i < change_direction_point:
            start_column -= each_warship_delta_column
        else:
            start_column += each_warship_delta_column
        start_column2 += each_warship_delta_column
        start_column3 -= each_warship_delta_column
    return warship_formation

def _seven_formation():
    warship_formation = np.zeros((10,60))
    start_column1 = 0
    start_column2 = 10 * each_warship_delta_column + 2*delta_column
    start_column21 = start_column1 + delta_column
    start_column22 = start_column2 - delta_column
    for i in range(10):
        if i % 2 == 0:
            warship_formation[i, start_column1] = 1
            warship_formation[i, start_column2] = 1
            if i+2 <10:
                warship_formation[i+2, start_column1+2*delta_column] = 1
                warship_formation[i+2, start_column2-2*delta_column] = 1
            else:
                warship_formation[i + 1, start_column1 + 2 * delta_column] = 1
                warship_formation[i + 1, start_column2 - 2 * delta_column] = 1
            start_column1 += each_warship_delta_column
            start_column2 -= each_warship_delta_column
        if (i+1)%2 == 0:
            warship_formation[i, start_column21] = 1
            warship_formation[i, start_column22] = 1
            start_column21 += each_warship_delta_column
            start_column22 -= each_warship_delta_column

    return warship_formation

def _eight_formation():
    pre_warship_formation = _seven_formation()
    cur_warship_list = []
    for i in range(10):
        cur_warship_list.append(pre_warship_formation[9-i].tolist())
    cur_warship_formation = np.array(cur_warship_list)
    return cur_warship_formation

def _nine_formation():
    warship_formation = np.zeros((10,60))
    start_row = 4
    end_row = 6
    start_column = 0
    for i in range(5):
        warship_formation[start_row:end_row, start_column] = 1
        start_row -= 1
        end_row += 1
        start_column +=each_warship_delta_column
    return warship_formation

def _ten_formation():
    warship_formation = np.zeros((10, 60))
    start_row = 0
    end_row = 10
    start_column = 0
    for i in range(5):
        warship_formation[start_row:end_row, start_column] = 1
        start_row += 1
        end_row -= 1
        start_column += each_warship_delta_column
    return warship_formation

def _reset_first_formation():
    warship_formation = np.zeros((10,60))
    for i in range(7):
        warship_formation[:i+1, i+ i* delta_column] = 1
    left_warship_number = int(settings.warships_num - sum(sum(warship_formation))) + 7
    warship_formation[7:left_warship_number, 6+6*delta_column] = 1
    return warship_formation

def _reset_second_formation():
    warship_formation = _reset_first_formation()
    new_formation = []
    for i in range(10):
        new_formation.append(warship_formation[9-i])
    return np.array(new_formation)

def _reset_three_formation():
    warship_formation = np.zeros((10,60))
    for j in range(10):
        if j < 5:
            for i in range(5-j):
                warship_formation[j,i*delta_column] = 1
        else:
            for i in range(j-4):
                warship_formation[j,i*delta_column] = 1
    warship_count = sum(sum(warship_formation))
    return warship_formation

def _reset_four_formation():
    warship_formation = np.zeros((10, 60))
    for j in range(7):
        for i in range(j+1):
            warship_formation[j,i*delta_column] = 1
    left_warship = int(settings.warships_num - sum(sum(warship_formation)))
    for i in range(left_warship):
        warship_formation[7+i,0] = 1
    warship_count = sum(sum(warship_formation))
    return warship_formation

def _reset_five_formation():
    warship_formation = _reset_four_formation()
    new_formation = []
    for i in range(10):
        new_formation.append(warship_formation[9 - i])
    return np.array(new_formation)


def _test_first_formation():
    warship_formation = np.zeros((10,60))
    warship_formation[:,1] = 1
    for i in range(1,5):
        warship_formation[:5,i+i*delta_column] = 1
    return  warship_formation

# 随机产生阵型
def _test_second_formation():
    warship_formation = np.zeros((10,60))
    all_position = []
    for i in range(10):
        for j in range(60):
            all_position.append([i,j])
    ship_position = random.sample(all_position, settings.warships_num)
    for unit in ship_position:
        warship_formation[unit[0],unit[1]] = 1
    return warship_formation


def _transform_array_to_list(position_info):
    x = []
    y = []
    [row, column] = position_info.shape
    for i in range(row):
        for j in range(column):
            if position_info[i,j] == 1:
                x.append(j)
                y.append(i)
    return x, y

def plot(x,y):
    plt.scatter(x, y, c='r', alpha=1, marker='o', label='pickup')
    plt.grid(True)
    plt.show()

if __name__ == '__main__':

    warship_formation = generate_warships_formation()
    for id, unit in warship_formation.items():
        xx, yy = _transform_array_to_list(unit)
        plt.subplot(2,5,id+1)
        # plt.axis([0,50,-1,10])
        plt.scatter(xx,yy,color='g')
    plt.show()

    # for i in range(len(x)):
    #     plt.subplot(2, 5, i + 1)
    #     plt.scatter(x[i][0], x[i][1])
    # plt.show()

    # print('test_first_formation:')
    # test_x, test_y = _transform_array_to_list(remove_warship_position(_test_first_formation()))
    # plot(test_x, test_y)
    #
    # print('test_second_formation:')
    # test_x1, test_y1 = _transform_array_to_list(_test_second_formation())
# plot(test_x1, test_y1)