import json
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from gensim.models import Word2Vec
import numpy as np
import logging
from vul_explainer.basic_visitor import extract_vars

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler('vul_explainer.log'), logging.StreamHandler()])

# Device configuration
device = 'cpu'#torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
            x = x_new + x if i > 0 else x_new  # Residual connection
            x = self.dropout(x)
        x = self.layers[-1](x, edge_index)
        return x

def gnn_identify_sink_points(self, cpg, word2vec_model_path="word2vec_model.model", gnn_model_path="gnn_word2vec_model.pth"):
    """
    Identify sink points in the CPG using a trained GNN model.
    
    Args:
        cpg: CPG object containing statements, DDGEdges, CDGEdges, and CFGEdges.
        word2vec_model_path: Path to the trained Word2Vec model.
        gnn_model_path: Path to the trained GNN model.
    
    Returns:
        sink_idxs: List of node indices identified as sinks.
        cdg_prec: Dictionary mapping node indices to their control dependency predecessors.
        ddg_prec: Dictionary mapping node indices to their data dependency predecessors.
    """
    # Load Word2Vec model
    try:
        word2vec_model = Word2Vec.load(word2vec_model_path)
        # logging.info("Word2Vec model loaded successfully")
    except Exception as e:
        logging.error(f"Failed to load Word2Vec model: {e}")
        return [], {}, {}
    
    # Load GNN model
    try:
        model = GNN(
            in_channels=word2vec_model.vector_size,
            hidden_channels=256,
            out_channels=4,  # none, source, sink, sanitizer
            layer_num=6
        ).to(device)
        model.load_state_dict(torch.load(gnn_model_path))
        model.eval()
        logging.info("GNN model loaded successfully")
    except Exception as e:
        logging.error(f"Failed to load GNN model: {e}")
        return [], {}, {}
    
    # Process CPG into a Data object
    embedding_dim = word2vec_model.vector_size
    default_embedding = np.zeros(embedding_dim)
    x = []
    
    # Generate node features using Word2Vec
    for stmt in cpg.statements:
        # Assume statements can be converted to a format similar to the dataset JSON
        # Using a simplified representation based on statement string or type
        try:
            # Convert statement to string and extract type/content
            stmt_str = str(stmt)
            node_type = stmt.__class__.__name__  # e.g., Identifier, CallExpression
            node_content = stmt_str.replace(' ', '_')[:50]
            token = f"{node_type}_{node_content}"
            embedding = word2vec_model.wv[token] if token in word2vec_model.wv else default_embedding
        except:
            # Fallback for statements with no clear string representation
            node_type = stmt.__class__.__name__
            node_content = "unknown"
            token = f"{node_type}_{node_content}"
            embedding = word2vec_model.wv[token] if token in word2vec_model.wv else default_embedding
        x.append(embedding)
    
    x = torch.tensor(np.array(x), dtype=torch.float).to(device)
    
    # Generate edge indices (bidirectional edges for DDG, CDG, CFG)
    edge_index = []
    for edge_type in [cpg.DDGEdges, cpg.CDGEdges, cpg.CFGEdges]:
        for edge in edge_type:
            src, dst = edge.source, edge.destination
            edge_index.append([src, dst])
            edge_index.append([dst, src])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(device)
    
    # Create Data object
    data = Data(x=x, edge_index=edge_index).to(device)
    
    # Predict node roles using GNN
    with torch.no_grad():
        out = model(data)
        preds = out.argmax(dim=1).cpu().numpy()
    
    # Identify sink points (class index 2)
    sink_idxs = [i for i, pred in enumerate(preds) if pred == 2]
    logging.info(f"Identified sink points: {sink_idxs}")
    
    # Compute control dependency predecessors
    cdg_prec = {}
    key_vars = {}
    for i, stmt in enumerate(cpg.statements):
        try:
            vars_in_stmt = extract_vars(stmt)
            key_vars[i] = vars_in_stmt
        except:
            key_vars[i] = set()
    
    for edge in cpg.CDGEdges:
        if edge.destination in key_vars:
            vars = key_vars[edge.destination]
            flag = False
            for var in vars:
                source_vars = extract_vars(cpg.statements[edge.source])
                if var in source_vars:
                    flag = True
            if not flag:
                continue
        if edge.destination in cdg_prec:
            cdg_prec[edge.destination].append(edge.source)
        else:
            cdg_prec[edge.destination] = [edge.source]
    
    # Compute data dependency predecessors
    ddg_prec = {}
    for edge in cpg.DDGEdges:
        if edge.destination in key_vars:
            vars = key_vars[edge.destination]
            flag = False
            if self.slice_level:
                source_vars = extract_vars(cpg.statements[edge.source])
                for var in vars:
                    if var in source_vars:
                        flag = True
            else:
                for var in vars:
                    if var in edge.property:
                        flag = True
            if not flag:
                continue
        if edge.destination in ddg_prec:
            ddg_prec[edge.destination].append(edge.source)
        else:
            ddg_prec[edge.destination] = [edge.source]
    
    logging.info(f"cdg_prec: {cdg_prec}")
    logging.info(f"ddg_prec: {ddg_prec}")
    return sink_idxs, cdg_prec, ddg_prec