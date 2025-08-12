import torch
from global_defines import vul_types, device, num_classes, cur_vul_type_idx

### IVDetect Configuration
# type = vul_types[cur_vul_type_idx]


class ModelParser:
    def __init__(self):
        self.pretrain_word2vec_model = '{}/models/{}/word/w2v_ivdetect.model'
        self.hidden_size = 128  # GNN隐层向量维度
        self.feature_representation_size = 128
        self.num_node_features = 5
        self.num_classes = num_classes
        self.num_layers = 3  # GNN层数
        self.dropout_rate = 0.3
        self.model_dir = "{}/models/{}/model/"
        self.device = device
        self.model_name = 'gcn'
        self.detector = "ivdetect"


class DataParser:
    def __init__(self):
        self.dataset_dir = '{}/datasets/{}/'
        self.shuffle_data = True  # 是否随机打乱数据集
        self.num_workers = 8
        self.random_split = True
        self.seed = 2
        self.batch_size = 64
        self.device = device
        self.num_classes = 2


class TrainParser:
    def __init__(self):
        self.max_epochs = 100
        self.early_stopping = 3
        self.save_epoch = 10
        self.learning_rate = 0.0001
        self.weight_decay = 0.0
        self.batch_size = 64
        self.test_batch_size = 64


model_args = ModelParser()
data_args = DataParser()
train_args = TrainParser()