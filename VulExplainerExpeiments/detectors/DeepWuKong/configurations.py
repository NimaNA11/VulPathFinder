import random
from global_defines import num_classes, vul_types, device
### DeepWuKong Configuration


class ModelParser:
    def __init__(self):
        self.vector_size = 128  # 图结点的向量维度
        self.hidden_size = 128  # GNN隐层向量维度
        self.layer_num = 3  # GNN层数
        self.rnn_layer_num = 1  # RNN层数
        self.num_classes = num_classes
        self.with_sym = True
        self.model_dir = "{}/models/{}/model/"
        self.device = device
        self.model_name = 'gcn'
        self.pretrain_word2vec_model = "{}/models/{}/word/w2v_slice.model"
        self.detector = 'dwk'


class DataParser:
    def __init__(self):
        self.dataset_dir = "{}/datasets/{}"
        self.shuffle_data = True  # 是否随机打乱数据集
        self.num_workers = 8
        self.random_split = True
        self.batch_size = 64
        self.test_batch_size = 64
        self.device = device
        self.num_classes = 2


class TrainParser:
    def __init__(self):
        self.max_epochs = 100
        self.early_stopping = 10
        self.save_epoch = 5
        self.learning_rate = 0.002
        self.weight_decay = 1.3e-6


random.seed(2)
model_args = ModelParser()
data_args = DataParser()
train_args = TrainParser()