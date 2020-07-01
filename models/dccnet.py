import paddle.fluid as fluid
# from .backbone import vgg
from DCCNet.models.nn import Dropout2d, ReLU
from DCCNet.models.backbone import VGG16


class DCCNet(fluid.dygraph.Layer):
    def __init__(self, vgg_path=None, nclass=21):
        super(DCCNet, self).__init__()
        self.backbone = VGG16()
        if vgg_path is not None:
            self.backbone.copy_params_from_vgg16(vgg_path)

        self.head = DCCNet_Head(out_channels=nclass)

    def forward(self, x):
        _, _, h, w = x.shape
        x4, x5, x6, x7 = self.backbone(x)
        print("X4 ", x4.shape)
        print("x5 ", x5.shape)
        print("x6 ", x6.shape)
        print("x7 ", x7.shape)
        conv_1, conv_2, conv_3, conv_4 = self.head(*[x4, x5, x6, x7])
        fusion = conv_1[:, :, 32:32+h, 32:32+w] + conv_2[:, :, 45:45+h, 45:45+w] + conv_3[:, :, 5:5+h, 5:5+w] + conv_4[:, :, 10:10+h, 10:10+w]
        print(fusion.shape)


class DCCNet_Head(fluid.dygraph.Layer):
    def __init__(self, in_channels=(512, 512, 4096, 4096), out_channels=21):
        super(DCCNet_Head, self).__init__()
        self.conv_1 = fluid.dygraph.Sequential(
            fluid.dygraph.Conv2D(num_channels=in_channels[0], num_filters=64, filter_size=3, padding=1, bias_attr=False),
            ReLU(),
            fluid.dygraph.BatchNorm(64),
            fluid.dygraph.Conv2DTranspose(num_channels=64, num_filters=out_channels, filter_size=32, stride=16, bias_attr=False))
        self.conv_2 = fluid.dygraph.Sequential(
            fluid.dygraph.Conv2D(num_channels=in_channels[1], num_filters=64, filter_size=3, padding=1, bias_attr=False),
            ReLU(),
            fluid.dygraph.BatchNorm(64),
            fluid.dygraph.Conv2DTranspose(num_channels=64, num_filters=out_channels, filter_size=64, stride=32, bias_attr=False))
        self.conv_3 = fluid.dygraph.Sequential(
            fluid.dygraph.Conv2D(num_channels=in_channels[2], num_filters=64, filter_size=3, padding=1, bias_attr=False),
            ReLU(),
            fluid.dygraph.BatchNorm(64),
            fluid.dygraph.Conv2DTranspose(num_channels=64, num_filters=out_channels, filter_size=64, stride=32, bias_attr=False))
        self.conv_4 = fluid.dygraph.Sequential(
            fluid.dygraph.Conv2D(num_channels=in_channels[3], num_filters=64, filter_size=3, padding=1, bias_attr=False),
            ReLU(),
            fluid.dygraph.BatchNorm(64),
            fluid.dygraph.Conv2DTranspose(num_channels=64, num_filters=out_channels, filter_size=64, stride=32, bias_attr=False))

    def forward(self, *inputs):
        conv_1 = self.conv_1(inputs[0])
        conv_2 = self.conv_2(inputs[1])
        conv_3 = self.conv_3(inputs[2])
        conv_4 = self.conv_4(inputs[3])
        print(conv_1.shape)
        print(conv_2.shape)
        print(conv_3.shape)
        print(conv_4.shape)
        # return self.conv_1(inputs[0])+self.conv_2(inputs[1])+self.conv_3(inputs[2])
        return conv_1, conv_2, conv_3, conv_4


if __name__ == '__main__':
    with fluid.dygraph.guard():
        import numpy as np
        # x1 = fluid.dygraph.to_variable(np.ones(shape=(1, 512, 32, 32), dtype="float32"))
        # x2 = fluid.dygraph.to_variable(np.ones(shape=(1, 512, 16, 16), dtype="float32"))
        # x3 = fluid.dygraph.to_variable(np.ones(shape=(1, 4096, 8, 8), dtype="float32"))
        # x4 = fluid.dygraph.to_variable(np.ones(shape=(1, 4096, 8, 8), dtype="float32"))
        # model = DCCNet_Head()
        # out = model(*[x1, x2, x3, x4])

        x = fluid.dygraph.to_variable(np.ones(shape=(1, 3, 512, 512), dtype="float32"))
        model = DCCNet()
        out = model(x)
