from __future__ import division
import numpy as np
import make_strategy
import generate_warship_formation
import settings
import random
import time
import matplotlib

matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt

BASE_POSITION = np.array([[-1, 0], [-1, 1], [-1, 2], [-1, 3], [-1, 4], [-1, 5], [-1, 6], [-1, 7], [-1, 8], [-1, 9]])
CITY_POSITION = np.array([[61, 0], [61, 1], [61, 2], [61, 3], [61, 4], [61, 5], [61, 6], [61, 7], [61, 8], [61, 9]])
MAX_MISSILE = 4  # 每个基地最多发射DD数
MAX_MISSILE_EACH_TARGET = 3 # 每个目标最多发射DD数
available_remove_missiles = 10  # 可移动的DD数
DAMAGE_THRESHOLD_VALUE = 4.9  # 蓝方是否打击的阈值条件
SAMPLE_NUMBER = 5000  # 同样的阵型与部署随机抽样的次数
NOT_DETECTED_MISSILES_DELTA = 2  # 蓝方可观测的DD数与实际最大相差2


class NewGenerateData():

    def __init__(self):
        self.warships_formation = generate_warship_formation.generate_warships_formation()
        self.bases_position = BASE_POSITION
        self.target_position = CITY_POSITION

    def get_blue_warship_formation(self):
        return self.warships_formation

    # 随机生成一种部署
    def get_red_random_deployment(self):
        each_base_missiles_num = []
        missiles_num_sum = 0
        for i in range(settings.bases_num):
            left_remove_missiles = available_remove_missiles - missiles_num_sum
            if left_remove_missiles < 0:
                left_remove_missiles = 0
            temp_missiles_num = random.randint(0, min(MAX_MISSILE - 1, left_remove_missiles))
            if i == settings.bases_num - 1:
                each_base_missiles_num.append(left_remove_missiles + 1)
            else:
                each_base_missiles_num.append(temp_missiles_num + 1)
            missiles_num_sum += temp_missiles_num
        while max(each_base_missiles_num) > 4:
            each_base_missiles_num[each_base_missiles_num.index(min(each_base_missiles_num))] += 1
            each_base_missiles_num[each_base_missiles_num.index(max(each_base_missiles_num))] -= 1
        assert sum(each_base_missiles_num) == settings.missiles_num, "The number of missiles is not equal 20"
        assert max(each_base_missiles_num) <= MAX_MISSILE, "The missiles number of base more than MAX_MISSILE"
        return each_base_missiles_num

    def _condition(self, all_random_target):
        target_missiles_count = []
        for i in range(settings.targets_num):
            target_missiles_count.append(all_random_target.count(i))
        if max(target_missiles_count) >= MAX_MISSILE_EACH_TARGET:
            remove_target_id = target_missiles_count.index(max(target_missiles_count))
            return True, remove_target_id
        else:
            return False, -1

    # 已知红方部署的情况下，按照一定的规则生成一种分配方案
    def get_red_random_single_assign_strategy(self, red_deployment):
        assign_strategy = np.zeros((10, 4, 10))
        target_list = list(range(10))
        assign_target = []
        all_random_target = []
        for i in range(len(red_deployment)):
            each_base_target = []
            for j in range(red_deployment[i]):
                target_id = random.sample(target_list,1)
                each_base_target.extend(target_id)
                all_random_target.extend(target_id)
                remove_flag, remove_target = self._condition(all_random_target)
                if remove_flag:
                    target_list.remove(remove_target)
                    while remove_target in all_random_target:
                        all_random_target.remove(remove_target)
            assign_target.append(each_base_target)

        for i in range(settings.bases_num):
            for j in range(red_deployment[i]):
                assign_strategy[i, j, assign_target[i][j]] = 1
        assert sum(sum(sum(assign_strategy))) == settings.missiles_num, "The number of missiles is not equal 20"
        return assign_strategy

    # 计算每局游戏的总毁伤(随机产生一种分配方案)
    def computer_damage(self, red_deployment, single_warship_formation):
        assign_strategy = self.get_red_random_single_assign_strategy(red_deployment)
        damage = make_strategy.computeDamageOfEachRoute(single_warship_formation, assign_strategy, self.bases_position, \
                                                        self.target_position)
        sum_damage = sum(sum(sum(damage)))
        return sum_damage

    # 计算每种部署与阵型下蓝方的胜利率，随机抽样sample_count局游戏
    def generate_single_data_output(self, red_deployment, sample_count, single_warship_formation):
        win_result = []
        damage = []
        for i in range(sample_count):
            temp_damage = self.computer_damage(red_deployment, single_warship_formation)
            # print(temp_damage)
            damage.append(temp_damage)
            if temp_damage > DAMAGE_THRESHOLD_VALUE:
                win_result.append(0)
            else:
                win_result.append(1)
        win_rate = sum(win_result) / len(win_result)  # 表示蓝方的胜利率，即造成的总损伤小于阈值条件
        return win_rate, damage

    # 测试同样的部署对不同阵型的胜利率
    def computer_win_rate_all_formation(self, red_deployment):
        win_result = []
        for id, unit in self.warships_formation.items():
            win_rate = self.generate_single_data_output(red_deployment, SAMPLE_NUMBER, unit)
            win_result.append(win_rate)
        return win_result

    # 将红方部署(列表形式)转化为10*4的数据
    def _transform_deployment_to_input(self, deployment):
        red_information = np.zeros((10, 4))
        base_position = self.bases_position.tolist()
        for i in range(red_information.shape[0]):
            for j in range(1, red_information.shape[1] + 1):
                if [j - 4, i] in base_position:
                    red_information[i, j] = deployment[base_position.index([j - 4, i])]
        return red_information

    # 将输入信息的10*4数据形式转换为红方部署列表
    def _transform_array_to_deployment(self, red_input_information):
        red_deployment = []
        for i in range(red_input_information.shape[0]):
            for j in range(red_input_information.shape[1]):
                if red_input_information[i, j] != 0:
                    red_deployment.append(red_input_information[i, j])
        return red_deployment

    # 根据红方部署随机产生蓝方可观测的部分信息
    def sample_incomplete_red_deployment(self, red_deployment):
        detected_red_deployment = []
        for i in range(len(red_deployment)):
            if red_deployment[i] > 2:
                detected_red_deployment.append(random.randint(red_deployment[i] - 2, red_deployment[i]))
            else:
                detected_red_deployment.append(random.randint(1, red_deployment[i]))
        return detected_red_deployment

    # 生成不完全信息的蓝方训练样本
    def generate_incomplete_information_data(self, each_formation_sample_number, random_detected_sample_count):
        self.file_name_incomplete = open("sample_data.npy", "wb")
        count = 0
        for id, unit in self.warships_formation.items():
            print("-------{0} th formation-------".format(id))
            for i in range(each_formation_sample_number):
                red_deployment = self.get_red_random_deployment()
                print("red_deployment:", red_deployment)
                label_win_rate, damage = self.generate_single_data_output(red_deployment, SAMPLE_NUMBER, unit)
                print("{0}th deployment win rate:".format(i), label_win_rate)
                for k in range(random_detected_sample_count):
                    detected_red_deployment = self.sample_incomplete_red_deployment(red_deployment)
                    print("{0} th detected red deployment:".format(k), detected_red_deployment)
                    red_input = self._transform_deployment_to_input(detected_red_deployment)
                    np.save(self.file_name_incomplete, [red_input, unit, np.array(label_win_rate)])
            count += 1
        self.file_name_incomplete.close()

    # 测试某种阵型下应该随机采样多少红方部署
    def test_deployment_each_formation(self, data_count, formation_index):
        f = open("red_deployment.txt", "w")
        win_rate = []
        for i in range(data_count):
            print("{0} th iter".format(i))
            red_deployment = self.get_red_random_deployment()
            print(red_deployment)
            f.write(str(red_deployment) + '\n')
            temp_win_rate, damage = self.generate_single_data_output(red_deployment, 5000,  \
                                    self.warships_formation[formation_index])
            print(temp_win_rate)
            win_rate.append(temp_win_rate)
        f.close()
        bins = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9,0.95, 1]
        plt.hist(win_rate, bins, histtype='bar', rwidth=1)
        plt.xlabel("blue_win_rate")
        plt.ylabel("random_deployment_count")
        plt.show()

    # 测试所有阵型下的不同部署的蓝方胜利率分布
    def test_deployment_win_rate(self, data_count):

        for id, formation in self.warships_formation.items():
            print("----{0} th formation----".format(id))
            win_rate = []
            for i in range(data_count):
                print("{0} th iter".format(i))
                red_deployment = self.get_red_random_deployment()
                print(red_deployment)
                temp_win_rate, damage = self.generate_single_data_output(red_deployment, 100, formation)
                win_rate.append(temp_win_rate)
            bins = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
            plt.subplot(2,5,id+1)
            plt.hist(win_rate, bins, histtype='bar', rwidth=1, color= 'g')
        plt.show()

    # 测试固定红方部署，蓝方不同阵型的毁伤
    def test_computer_damage(self, formation_index):
        red_deployment = [2] * settings.bases_num
        red_deployment = self.get_red_random_deployment()
        print("red_deployment:", red_deployment)
        win_rate, damage = self.generate_single_data_output(red_deployment, 5000, self.warships_formation[formation_index])
        average_damage = sum(damage) / len(damage)
        print("average_damage:", average_damage)
        print("win_rate:", win_rate)
        bins = [4, 4.2, 4.4, 4.6, 4.8, 5.0, 5.2, 5.4, 5.6, 5.8, 6.0, 6.2, 6.4, 6.6, 6.8, 7.0]
        plt.hist(damage, bins, histtype='bar', rwidth=1, color= 'g')
        plt.show()


# 测试代码
if __name__ == '__main__':
    generate_data = NewGenerateData()

    random.seed(int(time.time()))
    # generate_data.test_computer_damage()

    generate_data.test_deployment_each_formation(100,0)

# generate_data.test_deployment_win_rate(500)