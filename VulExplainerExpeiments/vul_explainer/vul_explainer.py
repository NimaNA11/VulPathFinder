from CppCodeAnalyzer.mainTool.CPG import CPG
from CppCodeAnalyzer.mainTool.ast.expressions.expressionHolders import Condition
from CppCodeAnalyzer.mainTool.ast.declarations.simpleDecls import ForInit
from vul_explainer.basic_visitor import SinkVisitor, checkisREexpr, checkDependence, extract_vars
from vul_explainer.PSP_visitor import BackwardLeakVisitor
# from vulexplainer import slice_level
from typing import List, Dict, Tuple

from typing import List, Dict, Set
from global_defines import sparsity_value

## 773 401 forward, 789 400, backward
# detector = "deepwukong"


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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

class VulExplainer:
    def __init__(self, visitor: SinkVisitor, slice_level: bool):
        self.visitor: SinkVisitor = visitor
        self.slice_level = slice_level
        self.slices: List[List[int]] = list()
        self.add_cdg: bool = True
        self.limits: int = 5

    # Identify sink points
    def identify_sink_points(self, cpg: CPG):
        sink_idxs: List[int] = list()
        cdg_prec: Dict[int, List[int]] = dict()  # Control dependency predecessors

        # Compute control dependency information
        for edge in cpg.CDGEdges:
            # print("Edge: ", edge)
            if edge.destination in cdg_prec.keys():
                cdg_prec[edge.destination].append(edge.source)
            else:
                cdg_prec[edge.destination] = [edge.source]

        # print("sink_idxs:")
        # print(sink_idxs)
        # print('cdg_prec:')
        # print(cdg_prec)
        key_vars: Dict[int, Set[int]] = dict()

        for i, stmt in enumerate(cpg.statements):
            # print(f"Statement {i}: {stmt}")
            self.visitor.reset()
            stmt.accept(self.visitor)
            if self.visitor.isSink:
                sink_idxs.append(i)
                key_vars[i] = self.visitor.key_vars
                # print("SINK: ", sink_idxs)
                # print("KEY VARS: ", key_vars)
            elif self.visitor.potential:
                # Check if the result is restricted
                if not checkDependence(i, self.visitor.potential_var, self.visitor.check_upper, self.visitor.check_lower,
                                   cdg_prec, cpg.statements):
                    sink_idxs.append(i)
                    key_vars[i] = self.visitor.key_vars
            elif self.visitor.isCond:
                if isinstance(self.visitor, BackwardLeakVisitor):
                    key_vars[i] = self.visitor.key_vars
                    # Relevant variables cannot be constants and must be in a for loop
                    if len(key_vars) > 0 and isinstance(cpg.statements[i - 1], ForInit):
                        sink_idxs.append(i)

        cdg_prec: Dict[int, List[int]] = dict()  # Control dependency predecessors
        # Compute control dependency information
        for edge in cpg.CDGEdges:                        
            # print("Keyvars")
            # print(key_vars.keys())
            # print("edgedestination")
            # print(edge.destination)
            if edge.destination in key_vars.keys():

                vars = key_vars[edge.destination]
                flag = False # Check if the variables corresponding to data dependency might be tainted
                for var in vars:
                    source_vars = extract_vars(cpg.statements[edge.source])
                    if var in source_vars:
                        flag = True
                if not flag:
                    continue
            if edge.destination in cdg_prec.keys():
                cdg_prec[edge.destination].append(edge.source)
            else:
                cdg_prec[edge.destination] = [edge.source]

        # Compute data dependency information
        ddg_prec: Dict[int, List[int]] = dict()  # Data dependency predecessors
        for edge in cpg.DDGEdges:
            # print("Keyvars")
            # print(key_vars.keys())
            # print("dataedgedestination")
            # print(edge.destination)
            # print("edgeproperty")
            # print(edge.property)
            # Sink points
            if edge.destination in key_vars.keys():
                vars = key_vars[edge.destination]
                flag = False # Check if the variables corresponding to data dependency might be tainted
                for var in vars:
                    # If you are explaining deepwukong, use the following code
                    if self.slice_level:
                        source_vars = extract_vars(cpg.statements[edge.source])
                        if var in source_vars:
                            flag = True
                    # If you are explaining other 3 function-level detectors, use the following code
                    else:
                        if var in edge.property:
                            flag = True
                if not flag:
                    continue

            if edge.destination in ddg_prec.keys():
                ddg_prec[edge.destination].append(edge.source)
            else:
                ddg_prec[edge.destination] = [edge.source]
            # print("sink_idxs:")
            # print(sink_idxs)
            # print("cdg_prec:")
            # print(cdg_prec)
            # print("ddg_prec:")
            # print(ddg_prec)
        return sink_idxs, cdg_prec, ddg_prec
    

    def identify_vul_lines(self, cpg: CPG, vul_idxs):
        sink_idxs: List[int] = list()
        cdg_prec: Dict[int, List[int]] = dict()  # Control dependency predecessors

        # Compute control dependency information
        for edge in cpg.CDGEdges:
            # print("Edge: ", edge)
            if edge.destination in cdg_prec.keys():
                cdg_prec[edge.destination].append(edge.source)
            else:
                cdg_prec[edge.destination] = [edge.source]

        # print("sink_idxs:")
        # print(sink_idxs)
        # print('cdg_prec:')
        # print(cdg_prec)
        key_vars: Dict[int, Set[int]] = dict()

        for i, stmt in enumerate(cpg.statements):
            # print(f"Statement {i}: {stmt}")
            self.visitor.reset()
            stmt.accept(self.visitor)
            if self.visitor.isSink or i in vul_idxs:
                sink_idxs.append(i)
                key_vars[i] = self.visitor.key_vars
                # print("SINK: ", sink_idxs)
                # print("KEY VARS: ", key_vars)
            elif self.visitor.potential:
                # Check if the result is restricted
                if not checkDependence(i, self.visitor.potential_var, self.visitor.check_upper, self.visitor.check_lower,
                                   cdg_prec, cpg.statements):
                    sink_idxs.append(i)
                    key_vars[i] = self.visitor.key_vars
            elif self.visitor.isCond:
                if isinstance(self.visitor, BackwardLeakVisitor):
                    key_vars[i] = self.visitor.key_vars
                    # Relevant variables cannot be constants and must be in a for loop
                    if len(key_vars) > 0 and isinstance(cpg.statements[i - 1], ForInit):
                        sink_idxs.append(i)

        cdg_prec: Dict[int, List[int]] = dict()  # Control dependency predecessors
        # Compute control dependency information
        for edge in cpg.CDGEdges:                        
            # print("Keyvars")
            # print(key_vars.keys())
            # print("edgedestination")
            # print(edge.destination)
            if edge.destination in key_vars.keys():

                vars = key_vars[edge.destination]
                flag = False # Check if the variables corresponding to data dependency might be tainted
                for var in vars:
                    source_vars = extract_vars(cpg.statements[edge.source])
                    if var in source_vars:
                        flag = True
                if not flag:
                    continue
            if edge.destination in cdg_prec.keys():
                cdg_prec[edge.destination].append(edge.source)
            else:
                cdg_prec[edge.destination] = [edge.source]

        # Compute data dependency information
        ddg_prec: Dict[int, List[int]] = dict()  # Data dependency predecessors
        for edge in cpg.DDGEdges:
            # print("Keyvars")
            # print(key_vars.keys())
            # print("dataedgedestination")
            # print(edge.destination)
            # print("edgeproperty")
            # print(edge.property)
            # Sink points
            if edge.destination in key_vars.keys():
                vars = key_vars[edge.destination]
                flag = False # Check if the variables corresponding to data dependency might be tainted
                for var in vars:
                    # If you are explaining deepwukong, use the following code
                    if self.slice_level:
                        source_vars = extract_vars(cpg.statements[edge.source])
                        if var in source_vars:
                            flag = True
                    # If you are explaining other 3 function-level detectors, use the following code
                    else:
                        if var in edge.property:
                            flag = True
                if not flag:
                    continue

            if edge.destination in ddg_prec.keys():
                ddg_prec[edge.destination].append(edge.source)
            else:
                ddg_prec[edge.destination] = [edge.source]
            # print("sink_idxs:")
            # print(sink_idxs)
            # print("cdg_prec:")
            # print(cdg_prec)
            # print("ddg_prec:")
            # print(ddg_prec)
        return sink_idxs, cdg_prec, ddg_prec



    # def gnn_identify_sink_points(self, cpg, word2vec_model_path="word2vec_model.model", gnn_model_path="gnn_word2vec_model.pth"):
    #     """
    #     Identify sink points in the CPG using a trained GNN model.
        
    #     Args:
    #         cpg: CPG object containing statements, DDGEdges, CDGEdges, and CFGEdges.
    #         word2vec_model_path: Path to the trained Word2Vec model.
    #         gnn_model_path: Path to the trained GNN model.
        
    #     Returns:
    #         sink_idxs: List of node indices identified as sinks.
    #         cdg_prec: Dictionary mapping node indices to their control dependency predecessors.
    #         ddg_prec: Dictionary mapping node indices to their data dependency predecessors.
    #     """
    #     # Load Word2Vec model
    #     try:
    #         word2vec_model = Word2Vec.load(word2vec_model_path)
    #         # logging.info("Word2Vec model loaded successfully")
    #     except Exception as e:
    #         logging.error(f"Failed to load Word2Vec model: {e}")
    #         return [], {}, {}
        
    #     # Load GNN model
    #     try:
    #         model = GNN(
    #             in_channels=word2vec_model.vector_size,
    #             hidden_channels=256,
    #             out_channels=4,  # none, source, sink, sanitizer
    #             layer_num=6
    #         ).to(device)
    #         model.load_state_dict(torch.load(gnn_model_path))
    #         model.eval()
    #         # logging.info("GNN model loaded successfully")
    #     except Exception as e:
    #         # logging.error(f"Failed to load GNN model: {e}")
    #         return [], {}, {}
        
    #     # Process CPG into a Data object
    #     embedding_dim = word2vec_model.vector_size
    #     default_embedding = np.zeros(embedding_dim)
    #     x = []
        
    #     # Generate node features using Word2Vec
    #     for stmt in cpg.statements:
    #         # Assume statements can be converted to a format similar to the dataset JSON
    #         # Using a simplified representation based on statement string or type
    #         try:
    #             # Convert statement to string and extract type/content
    #             stmt_str = str(stmt)
    #             node_type = stmt.__class__.__name__  # e.g., Identifier, CallExpression
    #             node_content = stmt_str.replace(' ', '_')[:50]
    #             token = f"{node_type}_{node_content}"
    #             embedding = word2vec_model.wv[token] if token in word2vec_model.wv else default_embedding
    #         except:
    #             # Fallback for statements with no clear string representation
    #             node_type = stmt.__class__.__name__
    #             node_content = "unknown"
    #             token = f"{node_type}_{node_content}"
    #             embedding = word2vec_model.wv[token] if token in word2vec_model.wv else default_embedding
    #         x.append(embedding)
        
    #     x = torch.tensor(np.array(x), dtype=torch.float).to(device)
        
    #     # Generate edge indices (bidirectional edges for DDG, CDG, CFG)
    #     edge_index = []
    #     for edge_type in [cpg.DDGEdges, cpg.CDGEdges, cpg.CFGEdges]:
    #         for edge in edge_type:
    #             src, dst = edge.source, edge.destination
    #             edge_index.append([src, dst])
    #             edge_index.append([dst, src])
    #     edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(device)
        
    #     # Create Data object
    #     data = Data(x=x, edge_index=edge_index).to(device)
        
    #     # Predict node roles using GNN
    #     with torch.no_grad():
    #         out = model(data)
    #         preds = out.argmax(dim=1).cpu().numpy()
        
    #     # Identify sink points (class index 2)
    #     sink_idxs = [i for i, pred in enumerate(preds) if pred == 2]
    #     # logging.info(f"Identified sink points: {sink_idxs}")
        
    #     # Compute control dependency predecessors
    #     cdg_prec = {}
    #     key_vars = {}
    #     for i, stmt in enumerate(cpg.statements):
    #         try:
    #             vars_in_stmt = extract_vars(stmt)
    #             key_vars[i] = vars_in_stmt
    #         except:
    #             key_vars[i] = set()
        
    #     for edge in cpg.CDGEdges:
    #         if edge.destination in key_vars:
    #             vars = key_vars[edge.destination]
    #             flag = False
    #             for var in vars:
    #                 source_vars = extract_vars(cpg.statements[edge.source])
    #                 if var in source_vars:
    #                     flag = True
    #             if not flag:
    #                 continue
    #         if edge.destination in cdg_prec:
    #             cdg_prec[edge.destination].append(edge.source)
    #         else:
    #             cdg_prec[edge.destination] = [edge.source]
        
    #     # Compute data dependency predecessors
    #     ddg_prec = {}
    #     for edge in cpg.DDGEdges:
    #         if edge.destination in key_vars:
    #             vars = key_vars[edge.destination]
    #             flag = False
    #             if self.slice_level:
    #                 source_vars = extract_vars(cpg.statements[edge.source])
    #                 for var in vars:
    #                     if var in source_vars:
    #                         flag = True
    #             else:
    #                 for var in vars:
    #                     if var in edge.property:
    #                         flag = True
    #             if not flag:
    #                 continue
    #         if edge.destination in ddg_prec:
    #             ddg_prec[edge.destination].append(edge.source)
    #         else:
    #             ddg_prec[edge.destination] = [edge.source]
        
    #     # logging.info(f"cdg_prec: {cdg_prec}")
    #     # logging.info(f"ddg_prec: {ddg_prec}")
    #     return sink_idxs, cdg_prec, ddg_prec


    def gnn_identify_sink_points(self, cpg, word2vec_model_path: str = "word2vec_model.model", gnn_model_path: str = "gnn_word2vec_model.pth") -> Tuple[List[int], Dict[int, List[int]], Dict[int, List[int]]]:
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
            logging.info("Word2Vec model loaded successfully")
        except Exception as e:
            logging.error(f"Failed to load Word2Vec model: {e}")
            return [], {}, {}
        
        # Load GNN model
        try:
            model = GNN(
                in_channels=word2vec_model.vector_size,
                hidden_channels=256,
                out_channels=2,  # none, sink
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
        
        # Identify sink points (class index 1)
        sink_idxs = [i for i, pred in enumerate(preds) if pred == 1]
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
                source_vars = extract_vars(cpg.statements[edge.source])
                for var in vars:
                    if var in source_vars:
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


    def dfs(self, stack: List[int], num: int, ddg_prec: Dict[int, List[int]]
            , cdg_prec: Dict[int, List[int]], max_num: int):
        # Reached length limit
        if num == max_num:
            self.slices.append(stack.copy())
            return
        cur = stack[-1]
        added_node: Set[int] = set()
        for prec_idx in ddg_prec.get(cur, []):
            added_node.add(prec_idx)
        if self.add_cdg:
            for prec_idx in cdg_prec.get(cur, []):
                if prec_idx in self.conds:
                    added_node.add(prec_idx)

        # Traversed to the end
        if len(added_node) == 0:
            self.slices.append(stack.copy())
            return

        for node_idx in added_node:
            stack.append(node_idx)
            self.slices.append(stack.copy())
            self.dfs(stack, num + 1, ddg_prec, cdg_prec, max_num)
            stack.pop()
        # print("SLICES")
        # print(self.slices)


        


    # Conduct backward traversal
    def generate_backward_slices(self, cpg: CPG, sink_idxs: List[int]
                                 , cdg_prec: Dict[int, List[int]], ddg_prec: Dict[int, List[int]]):
        slices: List[List[int]] = list()
        self.conds: List[int] = list()

        # Traverse condition nodes in the CPG
        for i, stmt in enumerate(cpg.statements):
            if isinstance(stmt, Condition):
                if checkisREexpr(stmt):
                    self.conds.append(i)
                else:
                    for prec_idx in ddg_prec.get(i, []):
                        if checkisREexpr(cpg.statements[prec_idx]):
                            self.conds.append(i)
                            break


        max_num: int = min(int(sparsity_value * len(cpg.statements)), self.limits)
        for idx in sink_idxs:
            stack: List[int] = [idx]
            self.dfs(stack, 1, ddg_prec, cdg_prec, max_num)

        return slices