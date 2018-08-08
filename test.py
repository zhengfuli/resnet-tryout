from torch import nn
from utils import tester
from network.resnet import resnet18, resnet34, resnet101
from settings import *
import sys
from New_Generate_data import *
from tqdm import trange
import torchvision.transforms as T
from matplotlib import pyplot as plt
from PIL import Image
import torch
from torch.autograd import Variable
from utils.tester import *
import torch.nn.functional as F

def testRed(num, visualize):
    # Set Test parameters
    params = TestParams()
    params.gpus = []  # set 'params.gpus=[]' to use CPU model. if len(params.gpus)>1, default to use params.gpus[0] to test
    # this model corresponds to the model trained in the 60th epoch shown in the two training results under ./architecture:
    # red_*_train_1e5_test_2e4_10_kinds_3min_per_epoch_resnet18.png
    params.ckpt = './models/formation_prediction.pth'

    # models
    # model = resnet34(pretrained=False, num_classes=1000)  # batch_size=120, 1GPU Memory < 7000M
    # model.fc = nn.Linear(512, 6)
    model = resnet18(pretrained=False,num_classes=1000)  # batch_size=60, 1GPU Memory > 9000M
    model.fc = nn.Linear(512, formation_num)

    # Test
    tester = RedTester(model, params)
    tester.test(num, visualize)

def testBlue(total_data, visualize=True):
    dataset = [total_data[i] for i in np.random.choice(range(100000), 10, replace=False)] if visualize else total_data
    images = []
    summary = [[] for i in range(formation_num)]

    model = resnet18(pretrained=False, num_classes=1000)
    model.fc = nn.Linear(512, 1)
    model.eval()
    m = torch.load("./models/state_evaluation.pth")
    new_dict = model.state_dict().copy()

    for i in range(len(m)):
        new_dict[list(model.state_dict().keys())[i]] = m[list(m.keys())[i]]

    model.load_state_dict(new_dict)

    city_position = [[255] * city_grid[1] for i in range(city_grid[0])]
    for pos in CITY_POSITION:
        city_position[pos[1]][pos[0] - 1 - ocean_grid[1]] = 0

    for data in dataset:
        base_position = data[0]

        row, col = base_position.shape

        for i in range(row):
            for j in range(col):
                if base_position[i][j]:
                    assert (base_position[i][j] <= MAX_MISSILE)
                    base_position[i][j] = (MAX_MISSILE - base_position[i][j]) * int(255 / MAX_MISSILE)
                else:
                    base_position[i][j] = 255

        img = np.hstack((base_position, 255 - 255 * data[1], np.array(city_position)))

        h, w = img.shape
        if (224 - h) % 2 != 0 or (224 - w) % 2 != 0:
            padding = (int((224 - w) / 2), 224 - int((224 - w) / 2) - w,
                       int((224 - h) / 2), 224 - int((224 - h) / 2) - h)
        else:
            padding = (int((224 - w) / 2), int((224 - h) / 2))

        transforms1 = T.Compose([
            T.Pad(padding, fill=255),
            T.ToTensor()
        ])

        transforms2 = T.Compose([
            T.Pad((0, 29), fill=255),
        ])

        img = Image.fromarray(img.astype('uint8')).convert('L')
        eval_value = model(Variable(torch.unsqueeze(transforms1(img),  0)))
        eval_value = round(float(eval_value), 4)

        if visualize:
            image = transforms2(img)
            images.append([image, str(eval_value) + "/" + str(data[2])])
        else:
            # print(data[1], type(data[1]))
            forms = [f.tolist() for f in list(formations.values())]
            i = forms.index(data[1].tolist())
            summary[i].append(abs(eval_value - float(data[2])))

    # print(images)

    if visualize:
        plt.figure("State Evaluation")
        for i in range(1, len(images)+1):
            plt.subplot(2, 5, i)
            plt.title(images[i-1][1])
            plt.imshow(images[i-1][0])
            # plt.axis('off')
        plt.show()

        if input() == " ":
            testBlue(total_data, visualize)
    else:
        avg, min_val, max_val = [], [], []
        total = 0
        for diff in summary:
            total += sum(diff)
            avg.append(round(sum(diff) / len(diff),4))
            max_val.append(round(max(diff), 4))
            min_val.append(round(min(diff), 4))
        print(round(total/10000), 4)
        index = [1,2,3,4,5,6,7,8,9,10]
        plt.bar(left=index, height=min_val, label="Min Absolute Difference")
        plt.bar(left=index, height=avg, bottom=min_val, label="Avg Absolute Difference")
        plt.bar(left=index, height=max_val, bottom=avg, label="Max Absolute Difference")
        plt.xlabel('Formation')
        plt.ylabel('Absolute Difference Between Evaluation and Ground Truth')
        plt.legend(loc='best')
        plt.show()

if __name__ == "__main__":
    # testRed(10, True)
    formations = generate_formations.generate_warships_formation()
    # print(list(formations.values()))

    file_incomplete_data = open("./data/sample_data0801.npy", "rb")
    total_data = []

    np.random.seed(int(time.time()))

    for i in trange(10000):
        data = np.load(file_incomplete_data)
        total_data.append(data)
    file_incomplete_data.close()
    np.random.shuffle(total_data)

    testBlue(total_data, False)