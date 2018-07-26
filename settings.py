import numpy as np
# how many ships will be detected by the satellite of the red side
# it will be used as numpy.random.randint(a,b)
# e.g. (5,6) means 5, (5,16) means random number from 5 tp 15
detected__warships_num = (10, 16)

# total number of warships in the formation of the blue side
warships_num = 30

# volume of training set for the red network
train_num = 100000

# volume of testing set for the red network
test_num = 20000

# volume of both training and testing set for the blue network
# must corresponds to the .npy file under ./data
data_amount = 100000

# number of data that will be used to train the network of the blue side
train_amount = 80000

# total number of formations
formation_num = 10

# gpu setup, give the gpu number which you can check out by using nvidia-smi in the terminal
gpus = [0, 1]

# path to training/testing data
data_root = './data/'

# path to saved models
model_path = './models/'

# batch_size per GPU, if use GPU mode batch size = batch_size * num of GPUs
batch_size = 100

# workers for processing dataset
num_workers = 8

# initial learning rate
init_lr = 0.01

# learning rate decay rate
lr_decay = 0.8

# momentum
momentum = 0.9

weight_decay = 0.000

nesterov = True

# max training epoch
max_epoch = 100

# save frequency
save_freq_epoch = 1

ocean_grid = (10,60)

base_grid = (10, 4)

city_grid = (10, 4)

# this is used to normalize
MAX_MISSILE = 4

BASE_POSITION = np.array([[-1, 0], [-1, 1], [-1, 2], [-1, 3], [-1, 4], [-1, 5], [-1, 6], [-1, 7], [-1, 8], [-1, 9]])

CITY_POSITION = np.array([[61, 0], [61, 1], [61, 2], [61, 3], [61, 4], [61, 5], [61, 6], [61, 7], [61, 8], [61, 9]])