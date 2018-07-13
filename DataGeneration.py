# -*- coding:utf-8 -*-

import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt

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


def generate_warships_formation():
    warships_formation = dict()
    warships_formation[0] = remove_warship_position(_first_formation())
    warships_formation[1] = remove_warship_position(_second_formation())
    warships_formation[2] = remove_warship_position(_three_formation())
    warships_formation[3] = remove_warship_position(_four_formation())
    warships_formation[4] = remove_warship_position(_five_formation())
    warships_formation[5] = remove_warship_position(_six_formation())
    warships_formation[6] = remove_warship_position(_seven_formation())
    warships_formation[7] = remove_warship_position(_eight_formation())
    warships_formation[8] = remove_warship_position(_nine_formation())
    warships_formation[9] = remove_warship_position(_ten_formation())
    return warships_formation

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
    print('first_formation:')
    print(_first_formation())
    first_x, first_y = _transform_array_to_list(remove_warship_position(_first_formation()))
    # plot(first_x,first_y)

    print('second_formation:')
    print(_second_formation())
    second_x, second_y = _transform_array_to_list(remove_warship_position(_second_formation()))
    # plot(second_x, second_y)

    print('three_formation:')
    print(_three_formation())
    three_x, three_y = _transform_array_to_list(remove_warship_position(_three_formation()))
    # plot(three_x,three_y)

    print('four_formation:')
    print(_four_formation())
    four_x, four_y = _transform_array_to_list(remove_warship_position(_four_formation()))
    # plot(four_x, four_y)

    print('five_formation:')
    print(_five_formation())
    five_x, five_y = _transform_array_to_list(remove_warship_position(_five_formation()))
    # plot(five_x, five_y)

    print('six_formation:')
    print(_six_formation())
    six_x, six_y = _transform_array_to_list(remove_warship_position(_six_formation()))
    print(len(six_x))
    # plot(six_x, six_y)

    print('seven_formation:')
    print(_seven_formation())
    seven_x, seven_y = _transform_array_to_list(remove_warship_position(_seven_formation()))
    print(len(seven_x))
    # plot(seven_x, seven_y)

    print('eight_formation:')
    print(_eight_formation())
    eight_x, eight_y = _transform_array_to_list(remove_warship_position(_eight_formation()))
    print(len(eight_x))
    # plot(eight_x, eight_y)

    print('nine_formation:')
    print(_nine_formation())
    nine_x, nine_y = _transform_array_to_list(remove_warship_position(_nine_formation()))
    print(len(nine_x))
    # plot(nine_x, nine_y)

    print('ten_formation:')
    print(_ten_formation())
    ten_x, ten_y = _transform_array_to_list(remove_warship_position(_ten_formation()))
    print(len(ten_x))
    # plot(ten_x, ten_y)

    x = [[first_x,first_y],[second_x,second_y],[three_x,three_y],[four_x,four_y],[five_x,five_y], \
         [six_x,six_y], [seven_x,seven_y],[eight_x,eight_y],[nine_x,nine_y],[ten_x,ten_y]]

    for i in range(len(x)):
        plt.subplot(2, 5, i + 1)
        plt.scatter(x[i][0], x[i][1])
    plt.show()
