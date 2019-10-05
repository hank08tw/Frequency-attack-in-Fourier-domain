# coding:utf8
import warnings


class DefaultConfig(object):
    env = 'default'  # visdom 环境
    #model = 'ResNet34'#'AlexNet'  # 使用的模型，名字必须与models/__init__.py中的名字一致
    test_root='./data/test_data/'
    check_root='./data/check_data/'
    #load_model_path = './checkpoints/resnet34_0110_10:38:29.pth'  # 加载预训练的模型的路径，为None代表不加载
    batch_size = 1  # batch size
    use_gpu = True  # user GPU or not
    num_workers = 4  # how many workers for loading data
    print_freq = 25  # print info every N batch

    debug_file = '/tmp/debug'  # if os.path.exists(debug_file): enter ipdb
    result_file = 'result.csv'

    max_epoch = 100
    lr = 0.001  # initial learning rate
    lr_decay = 0.5  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 0e-5  # 损失函数


def parse(self, kwargs):
    """
    根据字典kwargs 更新 config参数
    """
    for k, v in kwargs.items():
        if not hasattr(self, k):
            warnings.warn("Warning: opt has not attribut %s" % k)
        setattr(self, k, v)

    print('user config:')
    for k, v in self.__class__.__dict__.items():
        if not k.startswith('__'):
            print(k, getattr(self, k))


DefaultConfig.parse = parse
opt = DefaultConfig()
opt.parse = parse
