import numpy as np
import math
import random

import generate_warship_formation

# THRESHOLD
# F :F is the set of formation,each formation is a 10*60 matrix in numpy form
BASE_POSITION = np.array([[-1, 0], [-1, 1], [-1, 2], [-1, 3], [-1, 4], [-1, 5], [-1, 6], [-1, 7], [-1, 8], [-1, 9]])
CITY_POSITION = np.array([[61, 0], [61, 1], [61, 2], [61, 3], [61, 4], [61, 5], [61, 6], [61, 7], [61, 8], [61, 9]])

#MAX_ITERATION_1 : max number of interation for method 1
MAX_INTERATION_1 = 50
#MAX_ITERATION_2 : max number of interation for method 2
MAX_INTERATION_2 = 5
#MAX_ITERATION_3 : max number of interation for method 3
MAX_INTERATION_3 = 30
#THRESHOLD ：the threshold of total damage ,when damage reaches the threshold,stop the iteration
THRESHOLD = 8
#EPSLON: when the difference between two iteration less than EPSILON, we think it come into a local optimal
EPSILON = 0.001
#NUMBER_OF_BASE: the number of base in the map
NUMBER_OF_BASE = 10
#NUMBER_OF_WARSHIPS: the number of warship in the map
NUMBER_OF_WARSHIPS = 30
#NUMBER_OF_CITY: the number of city in the map
NUMBER_OF_CITY = 10
#NUMBER_OF_MISSILES: the number of missiles
NUMBER_OF_MISSILES = 20
#INTERCEPTION_RATE: the successful rate of each interception
INTERCEPTION_RATE = 0.1
#MEAN: for routes with an effective distance of l,the number of interception is L/MEAN
MEAN = 1.5
#RADIUS: the radius of Missile interception range
RADIUS = 1
#MAX_MISSILE :max number of missile in each base
MAX_MISSILE = 4

# F = generate_warship_formation.generate_warships_formation()
F = generate_warship_formation.generate_warships_formation()
print(F[0].shape[0])


# print(F[0])
# print('..................')
# print(F[1])


def initialStrategy():
    strategy = np.zeros([NUMBER_OF_BASE, MAX_MISSILE, NUMBER_OF_CITY])
    for i in range(NUMBER_OF_BASE):
        for j in range(MAX_MISSILE):
            if (j == 0 or j == 1):
                k = random.randint(0, 9)
                strategy[i][j][k] = 1
    return strategy

# 将阵型10*60的矩阵，转换成10*2的矩阵(表示位置信息)
def transformFormationIntoPosition(formation):
    k = 0
    positions = np.zeros([NUMBER_OF_WARSHIPS, 2])
    for i in range(formation.shape[0]):
        for j in range(formation.shape[1]):
            if formation[i][j] != 0:
                positions[k] = [j, i]
                k = k + 1

    return positions

# 计算单条线路穿过某搜船的拦截距离
def computeDistance(base_pos, city_pos, warship_pos):
    molecular = abs((base_pos[1] - city_pos[1]) * (warship_pos[0] - city_pos[0]) + (city_pos[0] - base_pos[0]) * (
                warship_pos[1] - city_pos[1]))
    denominator = math.sqrt((base_pos[1] - city_pos[1]) ** 2 + (city_pos[0] - base_pos[0]) ** 2)
    length = molecular / denominator
    # print(length)
    if length < RADIUS:
        distance = 2 * math.sqrt(RADIUS ** 2 - length ** 2)
    else:
        distance = 0

    return distance

# 计算单条线路的TF概率
def computeProbability(sum_of_distance):
    number_of_interception = sum_of_distance / MEAN
    expect_number_of_missiles = (1 - INTERCEPTION_RATE) ** number_of_interception
    return expect_number_of_missiles

# 计算在某种阵型和策略下，每条线路实际穿过的期望DD数
def computeExpectedNumberOfMissiles(formation, strategy, base_position, target_position):
    expected_number_of_missiles = np.zeros_like(strategy)
    positions = transformFormationIntoPosition(formation)

    for i, base in enumerate(strategy):
        for j, missiles in enumerate(base):
            if sum(strategy[i][j]) == 0:
                continue
            else:
                k = np.nonzero(missiles)[0][0]
                distance = np.zeros(len(positions))
                for m, position in enumerate(positions):
                    distance[m] = computeDistance(base_position[i], target_position[k], position)
                sum_of_distance = sum(distance)
                probability = computeProbability(sum_of_distance)
                expected_number_of_missiles[i][j][k] = probability
    return expected_number_of_missiles

# 计算在某种阵型和策略下，实际打到每个目标的期望DD数
def getSumOfExpectedNumberOfMissiles(expected_number_of_missiles):

    expected_number_of_missiles_in_each_city = np.sum(np.sum(expected_number_of_missiles, axis=0), axis=0)
    return expected_number_of_missiles_in_each_city

# 获得红方的分配方案
def getSumOfNumber(strategy):
    number_of_missiles = np.sum(np.sum(strategy, axis=0), axis=0)

    return number_of_missiles

# 毁伤函数，计算每个目标的期望毁伤
def damageFunction(expected_number_of_missiles_in_each_city):
    damage_of_each_city = np.zeros(10)

    for i, number in enumerate(expected_number_of_missiles_in_each_city):
        if (number > 3):
            damage_of_each_city[i] = 1
        elif (number >= 0 and number <= 1):
            damage_of_each_city[i] = 0.0167 * number ** 3 - 0.15 * number ** 2 + 19 / 30 * number
        elif (number > 1 and number <= 2):
            damage_of_each_city[i] = 0.0167 * (number - 1) ** 3 - 0.1 * (number - 1) ** 2 + 23 / 60 * (number - 1) + 0.5
        elif (number > 2 and number <= 3):
            damage_of_each_city[i] = 0.0167 * (number - 2) ** 3 - 0.05 * (number - 2) ** 2 + 7 / 30 * (number - 2) + 0.8
        # print(damage_of_each_city[i])
    return damage_of_each_city

# 根据目标的期望毁伤计算每条线路的毁伤
def getDamageOfEachRoute(damage_of_each_city, expected_number_of_missiles, expected_number_of_missiles_in_each_city):
    damage_of_each_route = np.zeros_like(expected_number_of_missiles)
    for i, ii in enumerate(damage_of_each_route):
        for j, jj in enumerate(ii):
            for k, kk in enumerate(jj):
                if expected_number_of_missiles_in_each_city[k] != 0:
                    damage_of_each_route[i][j][k] = expected_number_of_missiles[i][j][k] / \
                                                    expected_number_of_missiles_in_each_city[k] * damage_of_each_city[k]
                else:
                    damage_of_each_route[i][j][k] = 0
    return damage_of_each_route

# 计算某种阵型和策略下每条线路造成的毁伤
def computeDamageOfEachRoute(formation, strategy, base_position, target_position):
    damage_of_each_route = np.zeros_like(strategy)

    expected_number_of_missiles = computeExpectedNumberOfMissiles(formation, strategy,base_position,target_position)
    expected_number_of_missiles_in_each_city = getSumOfExpectedNumberOfMissiles(expected_number_of_missiles)
    number_of_missiles = getSumOfNumber(strategy)

    damage_of_each_city = damageFunction(expected_number_of_missiles_in_each_city)
    damage_of_each_route = getDamageOfEachRoute(damage_of_each_city, expected_number_of_missiles,
                                                expected_number_of_missiles_in_each_city)

    return damage_of_each_route

# 计算满足阵型概率分布的条件下每条线路的期望毁伤
def computeAverageDamageOfEachRoute(pro, strategy, base_position, target_position):
    damage_of_each_route = np.zeros_like(strategy)
    average_damage_of_each_route = np.zeros_like(strategy)

    for i in range(len(F)):
        damage_of_each_route = computeDamageOfEachRoute(F[i], strategy,base_position,target_position)
        average_damage_of_each_route = average_damage_of_each_route + pro[i] * damage_of_each_route

    return average_damage_of_each_route

# 获得期望毁伤最小的线路的index
def getWorstRoute(average_damage_of_each_route):
    index = [0] * 3
    min_damage = 100
    for i in range(len(average_damage_of_each_route)):
        for j in range(len(average_damage_of_each_route[i])):
            for k in range(len(average_damage_of_each_route[i][j])):
                if j != 0 and average_damage_of_each_route[i][j][k] != 0 and average_damage_of_each_route[i][j][
                    k] < min_damage:
                    min_damage = average_damage_of_each_route[i][j][k]
                    index = [i, j, k]

    return index

def moveMissile(strategy, index):
    strategy_ = strategy.copy()
    strategy_[index[0]][index[1]][index[2]] = 0
    return strategy_

def getPossibleStrategy(strategy_, base_position, target_position):
    possible_strategy = np.zeros([len(base_position) * len(target_position), len(base_position), MAX_MISSILE, len(target_position)])
    for i in range(len(base_position) * len(target_position)):
        strategy = strategy_.copy()
        for j, city in enumerate(strategy_[int(i / len(base_position))]):
            if sum(abs(city)) == 0:
                strategy[int(i / len(base_position))][j][i % len(target_position)] = 1
                possible_strategy[i] = strategy.copy()
                break
    return possible_strategy

def getBestStrategy(PossibleStrategy, pro,base_position, target_position):
    best_strategy = np.zeros_like(PossibleStrategy[0])
    max_damage = 0
    for strategy in PossibleStrategy:
        damage = computeTotalDamage(strategy,pro,base_position,target_position)
        if damage > max_damage:
            max_damage = damage
            best_strategy = strategy.copy()

    return best_strategy

# 计算总的期望毁伤
def computeTotalDamage(strategy,pro, base_position, target_position):
    average_damage_of_each_route = computeAverageDamageOfEachRoute(pro, strategy, base_position,target_position)
    total_damage = sum(sum(sum(average_damage_of_each_route)))
    return total_damage
