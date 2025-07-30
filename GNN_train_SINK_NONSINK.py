import json
import torch
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from gensim.models import Word2Vec
import random
import os
import logging
import seaborn as sns
import matplotlib.pyplot as plt

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler('train_gnn.log'), logging.StreamHandler()])

# Device configuration
device = 'cpu'#torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Node type mapping
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
    'ElseStatement': 69, 'ThrowExpression': 70, 'IncDecOp': 71, 'NewExpression': 72, 'DeleteExpression': 73,
    'BoolExpression': 74, 'CharExpression': 75, 'DoubleExpression': 76, 'IntegerExpression': 77,
    'PointerExpression': 78, 'StringExpression': 79, 'ExpressionHolderStatement': 80
}

# Parser classes
class ModelParser:
    def __init__(self):
        self.vector_size = 128
        self.hidden_size = 256
        self.layer_num = 6
        self.num_classes = 2  # sink, none
        self.device = device
        self.model_name = 'ggnn'
        self.detector = 'devign'

class DataParser:
    def __init__(self):
        self.shuffle_data = True
        self.num_workers = 4
        self.batch_size = 32
        self.val_batch_size = 32
        self.test_batch_size = 32
        self.device = device
        self.num_classes = 2

class TrainParser:
    def __init__(self):
        self.max_epochs = 100
        self.early_stopping = 10
        self.save_epoch = 5
        self.learning_rate = 5e-4
        self.weight_decay = 1e-5

# Initialize parsers
random.seed(42)
torch.manual_seed(42)
np.random.seed(42)
model_args = ModelParser()
data_args = DataParser()
train_args = TrainParser()
data_args.num_classes = model_args.num_classes

class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, layer_num=6):
        super(GNN, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        self.layers.append(GCNConv(in_channels, hidden_channels))
        self.norms.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(layer_num - 2):
            self.layers.append(GCNConv(hidden_channels, hidden_channels))
            self.norms.append(torch.nn.BatchNorm1d(hidden_channels))
        self.layers.append(GCNConv(hidden_channels, out_channels))
        self.dropout = torch.nn.Dropout(0.5)
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for i, (layer, norm) in enumerate(zip(self.layers[:-1], self.norms)):
            x_new = layer(x, edge_index)
            x_new = norm(x_new)
            x_new = torch.relu(x_new)
            x = x_new + x if i > 0 else x_new
            x = self.dropout(x)
        x = self.layers[-1](x, edge_index)
        return x

def random_walk(sample, start_node, walk_length=6, minority_nodes=None):
    walk = [start_node]
    edges = []
    for edge_type in ["ddgEdges", "cdgEdges", "cfgEdges"]:
        for edge in sample[edge_type]:
            src, dst, *_ = json.loads(edge)
            edges.append((src, dst))
    
    current = start_node
    for _ in range(walk_length - 1):
        neighbors = [dst for src, dst in edges if src == current]
        if not neighbors:
            break
        if minority_nodes and current in minority_nodes:
            minority_neighbors = [n for n in neighbors if n in minority_nodes]
            if minority_neighbors and random.random() < 0.8:
                current = random.choice(minority_neighbors)
            else:
                current = random.choice(neighbors)
        else:
            current = random.choice(neighbors)
        walk.append(current)
    return walk

def generate_word2vec_corpus(dataset, walk_per_node=20, walk_length=6):
    corpus = []
    role_to_idx = {"none": 0, "sink": 1}
    for sample in tqdm(dataset, desc="Generating Word2Vec corpus"):
        nodes = sample["nodes"]
        minority_nodes = []
        for idx, node in enumerate(nodes):
            node_dict = json.loads(node)
            role = node_dict.get("role", "none")
            if role == "sink":
                minority_nodes.append(idx)
        
        for node_idx in range(len(nodes)):
            walks = walk_per_node if node_idx not in minority_nodes else walk_per_node * 4
            for _ in range(walks):
                walk = random_walk(sample, node_idx, walk_length, minority_nodes)
                walk_types = []
                for idx in walk:
                    node_dict = json.loads(nodes[idx])
                    node_type = node_dict["contents"][0][0]
                    node_content = node_dict["contents"][0][1].replace(' ', '_')[:50]
                    walk_types.append(f"{node_type}_{node_content}")
                if walk_types:
                    corpus.append(walk_types)
    return corpus

def train_word2vec(corpus, embedding_dim=128, window=5, min_count=1):
    model = Word2Vec(
        sentences=corpus,
        vector_size=embedding_dim,
        window=window,
        min_count=min_count,
        workers=data_args.num_workers,
        sg=1,
        epochs=10
    )
    return model

def oversample_minority(dataset):
    role_to_idx = {"none": 0, "sink": 1}
    sink_samples = []
    for sample in dataset:
        nodes = sample["nodes"]
        has_sink = any(json.loads(node).get("role", "none") == "sink" for node in nodes)
        if has_sink:
            sink_samples.append(sample)
    
    oversampled = dataset + sink_samples * 2
    random.shuffle(oversampled)
    return oversampled

def load_dataset(dataset_path, word2vec_model):
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    indices = list(range(len(dataset)))
    if data_args.shuffle_data:
        random.shuffle(indices)
    
    train_split = int(0.7 * len(dataset))
    val_split = int(0.8 * len(dataset))
    train_indices = indices[:train_split]
    val_indices = indices[train_split:val_split]
    test_indices = indices[val_split:]
    
    splits = {
        'train_indices': train_indices,
        'val_indices': val_indices,
        'test_indices': test_indices
    }
    with open('splits.json', 'w') as f:
        json.dump(splits, f)
    
    train_dataset = oversample_minority([dataset[i] for i in train_indices])
    val_dataset = [dataset[i] for i in val_indices]
    test_dataset = [dataset[i] for i in test_indices]
    
    embedding_dim = word2vec_model.vector_size
    num_features = embedding_dim
    default_embedding = np.zeros(embedding_dim)
    
    def process_dataset(data_subset, desc):
        data_list = []
        for sample in tqdm(data_subset, desc=desc):
            nodes = sample["nodes"]
            x = []
            for node in nodes:
                node_dict = json.loads(node)
                node_type = node_dict["contents"][0][0]
                node_content = node_dict["contents"][0][1].replace(' ', '_')[:50]
                token = f"{node_type}_{node_content}"
                embedding = word2vec_model.wv[token] if token in word2vec_model.wv else default_embedding
                x.append(embedding)
            x = torch.tensor(np.array(x), dtype=torch.float).to(device)
            
            node_roles = {i: json.loads(node).get("role", "none") for i, node in enumerate(nodes)}
            role_to_idx = {"none": 0, "sink": 1}
            y = torch.tensor([role_to_idx[role] for role in node_roles.values()], dtype=torch.long).to(device)
            
            edge_index = []
            for edge_type in ["ddgEdges", "cdgEdges", "cfgEdges"]:
                for edge in sample[edge_type]:
                    edge_data = json.loads(edge)
                    src, dst = edge_data[0], edge_data[1]
                    edge_index.append([src, dst])
                    edge_index.append([dst, src])
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(device)
            
            data = Data(x=x, edge_index=edge_index, y=y)
            data_list.append(data)
        return data_list
    
    train_data_list = process_dataset(train_dataset, "Processing train samples")
    val_data_list = process_dataset(val_dataset, "Processing validation samples")
    test_data_list = process_dataset(test_dataset, "Processing test samples")
    return train_data_list, val_data_list, test_data_list, num_features, dataset, splits

def plot_confusion_matrix(labels, preds, classes, epoch, split='test'):
    cm = confusion_matrix(labels, preds, labels=range(len(classes)))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(f'Confusion Matrix - {split.capitalize()} Set - Epoch {epoch+1}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'confusion_matrix_{split}_epoch_{epoch+1}.png')
    plt.close()

def train_gnn(dataset_path):
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
    except Exception as e:
        logging.error(f"Failed to load dataset: {e}")
        return
    
    logging.info("Generating Word2Vec corpus...")
    corpus = generate_word2vec_corpus(dataset)
    word2vec_model = train_word2vec(corpus, embedding_dim=model_args.vector_size)
    word2vec_model.save("word2vec_model.model")
    logging.info("Word2Vec model saved to word2vec_model.model")
    
    train_data_list, val_data_list, test_data_list, num_features, dataset, splits = load_dataset(dataset_path, word2vec_model)
    
    class_counts = torch.zeros(model_args.num_classes)
    for data in train_data_list:
        class_counts += torch.bincount(data.y, minlength=model_args.num_classes)
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights[1] *= 1.5
    class_weights = class_weights / class_weights.sum() * model_args.num_classes
    class_weights = class_weights.to(device)
    logging.info(f"Class weights: {class_weights.tolist()}")
    
    train_loader = DataLoader(
        train_data_list,
        batch_size=data_args.batch_size,
        shuffle=True,
        num_workers=data_args.num_workers
    )
    val_loader = DataLoader(
        val_data_list,
        batch_size=data_args.val_batch_size,
        shuffle=False,
        num_workers=data_args.num_workers
    )
    test_loader = DataLoader(
        test_data_list,
        batch_size=data_args.test_batch_size,
        shuffle=False,
        num_workers=data_args.num_workers
    )
    
    model = GNN(
        in_channels=num_features,
        hidden_channels=model_args.hidden_size,
        out_channels=model_args.num_classes,
        layer_num=model_args.layer_num
    ).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_args.learning_rate,
        weight_decay=train_args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_args.max_epochs)
    
    best_val_f1 = 0
    epochs_no_improve = 0
    classes = ["none", "sink"]
    
    for epoch in range(train_args.max_epochs):
        model.train()
        total_loss = 0
        for data in train_loader:
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for data in val_loader:
                out = model(data)
                preds = out.argmax(dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(data.y.cpu().numpy())
        
        val_accuracy = accuracy_score(val_labels, val_preds)
        val_f1_macro = f1_score(val_labels, val_preds, average='macro')
        val_f1_per_class = f1_score(val_labels, val_preds, average=None, labels=range(len(classes)))
        val_precision = precision_score(val_labels, val_preds, average='macro', zero_division=0)
        val_recall = recall_score(val_labels, val_preds, average='macro', zero_division=0)
        
        logging.info(f"Epoch {epoch+1}, Train Loss: {total_loss / len(train_loader):.4f}, "
                     f"Val Accuracy: {val_accuracy:.4f}, Val F1-Macro: {val_f1_macro:.4f}, "
                     f"Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}")
        logging.info(f"Val Per-class F1: {dict(zip(classes, val_f1_per_class.round(4)))}")
        
        if (epoch + 1) % 5 == 0:
            plot_confusion_matrix(val_labels, val_preds, classes, epoch, split='val')
        
        if val_f1_macro > best_val_f1:
            best_val_f1 = val_f1_macro
            torch.save(model.state_dict(), "gnn_word2vec_model.pth")
            logging.info("Model saved to gnn_word2vec_model.pth")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        scheduler.step()
        
        if epochs_no_improve >= train_args.early_stopping:
            logging.info(f"Early stopping at epoch {epoch+1}")
            break
    
    model.load_state_dict(torch.load("gnn_word2vec_model.pth"))
    model.eval()
    test_preds, test_labels = [], []
    with torch.no_grad():
        for data in test_loader:
            out = model(data)
            preds = out.argmax(dim=1)
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(data.y.cpu().numpy())
    
    test_accuracy = accuracy_score(test_labels, test_preds)
    test_f1_macro = f1_score(test_labels, test_preds, average='macro')
    test_f1_per_class = f1_score(test_labels, test_preds, average=None, labels=range(len(classes)))
    test_precision = precision_score(test_labels, test_preds, average='macro', zero_division=0)
    test_recall = recall_score(test_labels, test_preds, average='macro', zero_division=0)
    cm = confusion_matrix(test_labels, test_preds, labels=range(len(classes)))
    
    logging.info(f"\nFinal Test Accuracy: {test_accuracy:.4f}")
    logging.info(f"Final Test F1-Macro: {test_f1_macro:.4f}")
    logging.info(f"Final Test Precision: {test_precision:.4f}")
    logging.info(f"Final Test Recall: {test_recall:.4f}")
    logging.info(f"Final Test Per-class F1: {dict(zip(classes, test_f1_per_class.round(4)))}")
    logging.info(f"Test Confusion Matrix:\n{cm}")
    
    plot_confusion_matrix(test_labels, test_preds, classes, epoch, split='test')
    
    logging.info("\nSample Predictions:")
    idx_to_role = {0: "none", 1: "sink"}
    for i, data in enumerate(test_data_list[:3]):
        sample_idx = splits['test_indices'][i]
        sample = dataset[sample_idx]
        with torch.no_grad():
            out = model(data)
            preds = out.argmax(dim=1).cpu().numpy()
            labels = data.y.cpu().numpy()
            logging.info(f"\nSample {sample_idx} ({sample.get('fileName', 'unknown')}):")
            for j in range(min(5, len(preds))):
                node_dict = json.loads(sample["nodes"][j])
                node_content = node_dict["contents"][0][1][:50]
                logging.info(f"  Node {j}: Predicted={idx_to_role[preds[j]]}, True={idx_to_role[labels[j]]}, Content={node_content}")

if __name__ == "__main__":
    dataset_path = "/home/nimana11/Thesis/codes/VulExplainerExp-84ED_2/runs/run_full_bufferoverflow_sink_none/labeled_dataset.json"
    train_gnn(dataset_path)