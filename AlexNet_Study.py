import os
import sys
import json
import torch
import torch.nn as nn
from torchvision import transforms, datasets
import torch.optim as optim
from tqdm import tqdm


# 模型构建
class AlexNet(nn.Module):  # 经典面向对象编程思想，把一个对象给抽象描述出来
    def __init__(self, num_classes=1000, init_weights=False):  # 定义模型的结构，模型的前半部分，卷积和池化，特征提取
        super(AlexNet, self).__init__()  # 继承的思想，继承nn.module
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),  # input[3, 224, 224]  output[96, 55, 55]
            nn.ReLU(inplace=True),  # 节省内存，后一次的结果覆盖前面的结果
            nn.MaxPool2d(kernel_size=3, stride=2),  # output[96, 27, 27]

            nn.Conv2d(96, 256, kernel_size=5, padding=2),  # output[256, 27, 27]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # output[256, 13, 13]

            nn.Conv2d(256, 384, kernel_size=3, padding=1),  # output[384, 13, 13]
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 384, kernel_size=3, padding=1),  # output[384, 13, 13]
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),  # output[256, 13, 13]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # output[256, 6, 6]
        )
        self.classifier = nn.Sequential(  # 分类识别
            nn.Dropout(p=0.5),  # 第一层
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),

            nn.Dropout(p=0.5),  # 防止过拟合，每层结果以50%随机丢失；第二层
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),

            nn.Linear(4096, num_classes),  # 第三层
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):  # 定义模型的功能
        x = self.features(x)  # 特征提取
        x = torch.flatten(x, start_dim=1)  # 打平
        x = self.classifier(x)  # 分类识别
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def alexnet(num_classes):
    model = AlexNet(num_classes=num_classes)
    return model


def main():
    # 判断可用设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # 数据集路径
    data_path = 'C:\\Dataset\\flower'
    assert os.path.exists(data_path), "{} path does not exists.".format(data_path)

    # 数据预处理与增强
    """
    Totensor()能够把灰度范围从0-255变换到0-1之间的张量.
    transform.Normalize()则把0-1变换到(-1,1).具体地说，对每个通道而言，Normalization执行以下操作：image=（image-mean)/std
    其中mean和std分别通过(0.5,0.5,0.5)和(0.5,0.5,0.5)进行指定。原来的0-1最小值0则变成(0-0.5)/0.5=-1；而最大值1则变成(1-0.5)/0.5=1.
    也就是一个均值为0，方差为1的正态分布.这样的数据输入格式可以使神经网络更快收敛。
    """
    data_transform = {
        "train": transforms.Compose([transforms.Resize(224),  # 图片缩放
                                     transforms.CenterCrop(224),  # 图片增强，中心裁剪，数据扩充
                                     transforms.ToTensor(),  # 图片转成张量的格式，pytorch
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),  # 归一化
        "val": transforms.Compose([transforms.Resize((224, 224)),  # val不需要任何数据增强
                                   transforms.ToTensor(),  # 转化成Tensor格式
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])  # 归一化
    }

    # 使用ImageFolder加载数据集中的图像，并使用指定的预处理操作来处理图像，ImageFolder会同时返回图像和对应的标签(image_path,class_index) tuples
    train_dataset = datasets.ImageFolder(root=os.path.join(data_path, "train"), transform=data_transform["train"])
    validate_dataset = datasets.ImageFolder(root=os.path.join(data_path, "val"), transform=data_transform["val"])
    train_num = len(train_dataset)
    val_num = len(validate_dataset)

    batch_size = 64  # 将batch_size大小，是超参，可调，如果模型跑不起来，尝试调小batch_size，选取批量数据进行训练

    # 使用DataLoder 将ImageFolder加载的数据集处理成批量（batch）加载模式
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # 数据打包
    validate_loader = torch.utils.data.DataLoader(validate_dataset, batch_size=4, shuffle=False)  # 注意验证集不需要shuffle
    print("using {} images for training, {} images for validation.".format(train_num, val_num))

    # 实例化模型，并送进设备

    net = AlexNet(num_classes=5)
    net.to(device)

    # 指定损失函数用于计算损失；指定优化器用于更新模型参数；指定训练迭代的轮数，训练权重的存储地址
    loss_function = nn.CrossEntropyLoss()  # 损失函数
    optimizer = optim.Adam(net.parameters(), lr=0.0002)  # 实例化优化器，pytorch封装好的，模型更新权重和偏执，底层实习了梯度下降
    epochs = 70  # 重复学习70次
    save_path = os.path.abspath(os.path.join(os.getcwd(), './alexnet'))  # 指定训练好的模型权重保存的路径
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for epoch in range(epochs):  # 循环epochs次
        #####train#####训练模型
        net.train()
        acc_num = torch.zeros(1).to(device)  # 初始化，用于计算训练过程中预测正确的数量
        sample_num = 0  # 初始化，用于记录当前迭代中，已经计算了多少个样本
        # tqdm 是一个进度条显示器，可以在终端打印出现的训练进度
        train_bar = tqdm(train_loader, file=sys.stdout, ncols=100)  # 进度条，对数据集没有任何影响
        for data in train_bar:
            images, labels = data
            sample_num += images.shape[0]  # [64,3,244,244]
            optimizer.zero_grad()  # 初始化
            outputs = net(images.to(device))  # output_shape:[batch_size, num_classes]；图片送到设备里面，然后再把设备送给模型，模型给出一个预测结果
            # 会有5个预测结果，把最大的结果取出来
            pred_class = torch.max(outputs, dim=1)[1]  # torch.max 返回值是一个tuple，第一个元素是max值，第二个元素是max值的索引
            acc_num += torch.eq(pred_class, labels.to(device)).sum()  # 预测值与真值比较是否相等，相等数进行累加
            loss = loss_function(outputs, labels.to(device))  # 求损失
            loss.backward()  # 自动求导
            optimizer.step()  # 梯度下降

            # print statistics
            train_acc = acc_num.item() / sample_num  # 预测正确的num除样本num得准确率
            # .desc是进度条tqdm中的成员变量，作用是描述信息
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, epochs, loss)

        # validate#验证
        net.eval()  # 模型置于验证状态下
        acc_num = 0.0  # 初始化当前验证集上识别正确的数量
        with torch.no_grad():  # 验证且不希望改变梯度
            for val_data in validate_loader:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc_num += torch.eq(predict_y, val_labels.to(device)).sum().item()

        val_accurate = acc_num / val_num
        print('[epoch %d] train_loss: %.3f train_acc: %.3f val_accuracy: %.3f' % (
        epoch + 1, loss, train_acc, val_accurate))
        torch.save(net.state_dict(), os.path.join(save_path, "AlexNet.pth"))

        # 每次迭代后清空这些指标，重新计算
        train_acc = 0.0
        val_accurate = 0.0

    print('Finished Training')


main()