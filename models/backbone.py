import numpy as np
from collections import OrderedDict
import paddle.fluid as fluid
from FCN_Paddle.models.nn import ReLU, Dropout2d


class VGG16(fluid.dygraph.Layer):

    def __init__(self):
        super(VGG16, self).__init__()
        # conv1
        self.conv1_1 = fluid.dygraph.Conv2D(3, 64, 3, padding=100)
        self.relu1_1 = ReLU()
        self.conv1_2 = fluid.dygraph.Conv2D(64, 64, 3, padding=1)
        self.relu1_2 = ReLU()
        self.pool1 = fluid.dygraph.Pool2D(2, pool_stride=2, ceil_mode=True)  # 1/2

        # conv2
        self.conv2_1 = fluid.dygraph.Conv2D(64, 128, 3, padding=1)
        self.relu2_1 = ReLU()
        self.conv2_2 = fluid.dygraph.Conv2D(128, 128, 3, padding=1)
        self.relu2_2 = ReLU()
        self.pool2 = fluid.dygraph.Pool2D(2, pool_stride=2, ceil_mode=True)  # 1/4

        # conv3
        self.conv3_1 = fluid.dygraph.Conv2D(128, 256, 3, padding=1)
        self.relu3_1 = ReLU()
        self.conv3_2 = fluid.dygraph.Conv2D(256, 256, 3, padding=1)
        self.relu3_2 = ReLU()
        self.conv3_3 = fluid.dygraph.Conv2D(256, 256, 3, padding=1)
        self.relu3_3 = ReLU()
        self.pool3 = fluid.dygraph.Pool2D(2, pool_stride=2, ceil_mode=True)  # 1/8

        # conv4
        self.conv4_1 = fluid.dygraph.Conv2D(256, 512, 3, padding=1)
        self.relu4_1 = ReLU()
        self.conv4_2 = fluid.dygraph.Conv2D(512, 512, 3, padding=1)
        self.relu4_2 = ReLU()
        self.conv4_3 = fluid.dygraph.Conv2D(512, 512, 3, padding=1)
        self.relu4_3 = ReLU()
        self.pool4 = fluid.dygraph.Pool2D(2, pool_stride=2, ceil_mode=True)  # 1/16

        # conv5
        self.conv5_1 = fluid.dygraph.Conv2D(512, 512, 3, padding=1)
        self.relu5_1 = ReLU()
        self.conv5_2 = fluid.dygraph.Conv2D(512, 512, 3, padding=1)
        self.relu5_2 = ReLU()
        self.conv5_3 = fluid.dygraph.Conv2D(512, 512, 3, padding=1)
        self.relu5_3 = ReLU()
        self.pool5 = fluid.dygraph.Pool2D(2, pool_stride=2, ceil_mode=True)  # 1/32

        # fc6
        self.fc6 = fluid.dygraph.Conv2D(512, 4096, 7)
        self.relu6 = ReLU()
        self.drop6 = Dropout2d()

        # fc7
        self.fc7 = fluid.dygraph.Conv2D(4096, 4096, 1)
        self.relu7 = ReLU()
        self.drop7 = Dropout2d()

    def forward(self, x):
        h = x
        h = self.relu1_1(self.conv1_1(h))
        h = self.relu1_2(self.conv1_2(h))
        h = self.pool1(h)  # 1/2

        h = self.relu2_1(self.conv2_1(h))
        h = self.relu2_2(self.conv2_2(h))
        h = self.pool2(h)  # 1/4

        h = self.relu3_1(self.conv3_1(h))
        h = self.relu3_2(self.conv3_2(h))
        h = self.relu3_3(self.conv3_3(h))
        h = self.pool3(h)  # 1/8

        h = self.relu4_1(self.conv4_1(h))
        h = self.relu4_2(self.conv4_2(h))
        h = self.relu4_3(self.conv4_3(h))
        h = self.pool4(h)  # 1/16
        x4 = h

        h = self.relu5_1(self.conv5_1(h))
        h = self.relu5_2(self.conv5_2(h))
        h = self.relu5_3(self.conv5_3(h))
        h = self.pool5(h)
        x5 = h  # 1/32

        h = self.relu6(self.fc6(h))
        h = self.drop6(h)
        x6 = h

        h = self.relu7(self.fc7(h))
        h = self.drop7(h)
        x7 = h

        return x4, x5, x6, x7

    def copy_params_from_vgg16(self, vgg_path):
        state_dict = fluid.dygraph.load_dygraph(model_path=vgg_path)[0]
        print(state_dict.keys())
        print(self.state_dict().keys())
        Dict = OrderedDict()
        for l1, l2 in zip(list(self.state_dict().keys()), list(state_dict.keys())[:-2]):
            if self.state_dict()[l1].shape == state_dict[l2].shape:
                Dict[l1] = state_dict[l2]
            else:
                Dict[l1] = np.reshape(state_dict[l2], self.state_dict()[l1].shape)
        self.load_dict(Dict)


if __name__ == '__main__':
    with fluid.dygraph.guard():
        model = VGG16()
        model.copy_params_from_vgg16(r"E:\Code\Python\PaddleSeg\PaddleVision\models\vgg16")
        print(model.state_dict()["conv1_1.bias"])
        x = fluid.dygraph.to_variable(np.ones(shape=[1, 3, 512, 512], dtype="float32"))
        outs = model(x)
        for out in outs:
            print(out.shape)

