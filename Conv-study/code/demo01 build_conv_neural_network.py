'''
@time 2022.1.12
@author xuan
@email 1920425406@qq.com

吴恩达老师卷积神经网路课程课后作业-搭建卷积神经网络

来源: https://blog.csdn.net/u013733326/article/details/80086090
'''

import numpy as np
import h5py
import matplotlib.pyplot as plt

# 边界padding填充函数
def zero_pad(X, pad):
    """
    功能:把数据集X的图像边界全部使用O来扩充pad个宽度和高度, 可以通过np.pad()函数来实现

    X - 图像数据集， 维度 (样本数n， 图像高度h， 图像宽度w， 图像通道数c)
    pad - 整数，每个图像在垂直和水平方向上的填充量p

    返回:
        X_paded - 扩充后的数据集，维度(n，h+2p, w+2p, c)

    """

    X_paded = np.pad(X, (
                        (0, 0),      # 样本数，不进行填充
                        (pad, pad),  # 图像高度，在上面和下面同时填充pad个0
                        (pad, pad),  # 图像宽度， 在左面和右面填充pad个0
                        (0, 0)),     # 通道数，不填充
                        'constant', constant_values=0)    #连续填充，填充值为0

    return X_paded

# 测试填充边界
def test():
    """
    功能: 对padding函数进行测试

    """
    np.random.seed(1)    # 设置种子
    x = np.random.randn(4, 3, 3, 2)
    x_paded = zero_pad(x, 2)

    # 展示信息
    print("x.shape", x.shape)
    print("x_paded.shape", x_paded.shape)


    # 画图
    fig, axarr = plt.subplots(1, 2)
    axarr[0].set_title('x')
    axarr[0].imshow(x[0, :, :, 0])
    axarr[1].set_title('x_paded')
    axarr[1].imshow(x_paded[0, :, :, 0])

    plt.show()


# 单步卷积
def conv_single_step(a_slice_prev, w, b):
    """
    
    功能: 在前一层的激活输出的一个片段上应用一个由参数w定义的过滤器
    切片尺寸与过滤器尺寸f大小相同

    参数:
        a_slice_prev  输入数据的一个片段， 维度(f, f, 前一层激活输出的通道数)
        w  --权重参数， 包含在一个矩阵内. 维度(f, f, 前一层激活输出的通道数)
        b  --偏置参数， 同样包含在一个矩阵内， 维度(1, 1, 1)

    """

    z = np.sum(np.multiply(a_slice_prev, w)) + b

    return z


# 测试单步卷积
def test_conv_single_step():

    np.random.seed(1)

    a_slice_prev = np.random.randn(4, 4, 3)
    w = np.random.randn(4, 4, 3)
    b = np.random.randn(1, 1, 1)

    z =conv_single_step(a_slice_prev, w, b)

    print(z)


if __name__ == "__main__":

    # test()
    test_conv_single_step()