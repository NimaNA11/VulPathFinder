import random
from global_defines import vul_types, device, num_classes, cur_vul_type_idx
### Devign Configuration

# type = vul_types[cur_vul_type_idx]

type_map = {
    'AndExpression': 1, 'Sizeof': 2, 'Identifier': 3, 'ForInit': 4, 'ReturnStatement': 5, 'SizeofOperand': 6,
    'InclusiveOrExpression': 7, 'PtrMemberAccess': 8, 'AssignmentExpr': 9, 'ParameterList': 10,
    'IdentifierDeclType': 11, 'SizeofExpr': 12, 'SwitchStatement': 13, 'IncDec': 14, 'Function': 15,
    'BitAndExpression': 16, 'UnaryOp': 17, 'DoStatement': 18, 'GotoStatement': 19, 'Callee': 20,
    'OrExpression': 21, 'ShiftExpression': 22, 'Decl': 23, 'CFGErrorNode': 24, 'WhileStatement': 25,
    'InfiniteForNode': 26, 'RelationalExpression': 27, 'CFGExitNode': 28, 'Condition': 29, 'BreakStatement': 30,
    'CompoundStatement': 31, 'UnaryOperator': 32, 'CallExpression': 33, 'CastExpression': 34,
    'ConditionalExpression': 35, 'ArrayIndexing': 36, 'PostfixExpression': 37, 'Label': 38,
    'ArgumentList': 39, 'EqualityExpression': 40, 'ReturnType': 41, 'Parameter': 42, 'Argument': 43, 'Symbol': 44,
    'ParameterType': 45, 'Statement': 46, 'AdditiveExpression': 47, 'PrimaryExpression': 48, 'DeclStmt': 49,
    'CastTarget': 50, 'IdentifierDeclStatement': 51, 'IdentifierDecl': 52, 'CFGEntryNode': 53, 'TryStatement': 54,
    'Expression': 55, 'ExclusiveOrExpression': 56, 'ClassDef': 57, 'ClassStaticIdentifier': 58, 'ForRangeInit': 59,
    'ClassDefStatement': 60, 'FunctionDef': 61, 'IfStatement': 62, 'MultiplicativeExpression': 63,
    'ContinueStatement': 64, 'MemberAccess': 65, 'ExpressionStatement': 66, 'ForStatement': 67, 'InitializerList': 68,
    'ElseStatement': 69, 'ThrowExpression': 70, 'IncDecOp': 71, 'NewExpression': 72, 'DeleteExpression': 73, 'BoolExpression': 74,
    'CharExpression': 75, 'DoubleExpression': 76, 'IntegerExpression': 77, 'PointerExpression': 78, 'StringExpression': 79,
    'ExpressionHolderStatement': 80
}

class ModelParser:
    def __init__(self):
        self.pretrain_word2vec_model = "{}/models/{}/word/w2v.model"
        self.vector_size = 100  # 图结点的向量维度
        self.hidden_size = 256  # GNN隐层向量维度
        self.layer_num = 3  # GNN层数
        self.num_classes = num_classes
        self.model_dir = "{}/models/{}/model/"
        self.device = device
        self.model_name = 'ggnn'
        self.detector = 'devign'


class DataParser:
    def __init__(self):
        self.dataset_dir = '{}/datasets/{}'
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
        self.early_stopping = 5
        self.save_epoch = 5
        self.learning_rate = 1e-4
        self.weight_decay = 1.3e-6


random.seed(2)
model_args = ModelParser()
data_args = DataParser()
train_args = TrainParser()