from vul_explainer.vul_explainer import VulExplainer
from vul_explainer.basic_visitor import SinkVisitor
from vul_explainer.PSP_visitor import BufferOverflowVisitor, IncorrectCalculationVisitor, BackwardLeakVisitor, PathTraversalVisitor, CommandInjectionVisitor, UncontrolledFormatVisitor

from CppCodeAnalyzer.mainTool.CPG import CPG, CodeEdge
from CppCodeAnalyzer.mainTool.ast.builders import json2astNode
from CppCodeAnalyzer.mainTool.ast.astNode import ASTNode

from global_defines import device
import json
from typing import Dict, List
import os
import sys
from tqdm import tqdm, trange

import numpy as np
from gensim.models.word2vec import Word2Vec
import torch
from torch_geometric.data import Data, Batch

# from vulexplainer import data_path, cur_detector
# Reveal
from detectors.Reveal.model import ClassifyModel
from detectors.Reveal.util import RevealUtil
from detectors.Reveal.configurations import model_args as reveal_model_args, \
    data_args as reveal_data_args

# Devign
from detectors.Devign.model import DevignModel
from detectors.Devign.util import DevignUtil
from detectors.Devign.configurations import model_args as devign_model_args, \
    data_args as devign_data_args

#IVDetect
from detectors.IVDetect.model import IVDetectModel
from detectors.IVDetect.util import IVDetectUtil
from detectors.IVDetect.configurations import model_args as ivdetect_model_args, \
        data_args as ivdetect_data_args

from .GNN_identify_sink_points import *

import html
import numpy.core.multiarray
import numpy
data_path = {
    1: "function/explain_reveal.json",
    3: "function/explain_devign.json",
    2: "function/explain_ivdetect.json",
}



def compare_path_and_edge_mask(path: List[int], edge_index: torch.Tensor, binary_edge_mask: torch.Tensor) -> float:
    """
    Compare VulExplainer path (node indices) with GNNExplainer binary edge mask.
    
    Args:
        path (List[int]): List of node indices from VulExplainer's selected path.
        edge_index (torch.Tensor): Tensor of shape [2, num_edges] defining graph edges.
        binary_edge_mask (torch.Tensor): Binary tensor of shape [num_edges] with 1 for important edges.
    
    Returns:
        float: Jaccard similarity between path edges and GNNExplainer important edges.
    """
    path_edges = set()
    for i in range(len(path) - 1):
        src, dst = path[i], path[i + 1]
        mask = (edge_index[0] == src) & (edge_index[1] == dst)
        matching_indices = torch.where(mask)[0]
        if matching_indices.numel() > 0:
            edge_idx = matching_indices[0].item()  # Take the first matching edge index
            path_edges.add((src, dst, edge_idx))
    
    gnn_edges = set()
    important_indices = torch.where(binary_edge_mask == 1)[0]
    for idx in important_indices:
        src, dst = edge_index[0, idx].item(), edge_index[1, idx].item()
        gnn_edges.add((src, dst, idx.item()))
    
    intersection = len(path_edges & gnn_edges)
    union = len(path_edges | gnn_edges)
    jaccard_similarity = intersection / union if union > 0 else 0.0
    return jaccard_similarity


def compute_gnn_recall(vul_idxs: List[int], edge_index: torch.Tensor, binary_edge_mask: torch.Tensor) -> float:
    """
    Compute recall for GNNExplainer by identifying nodes incident to important edges.
    
    Args:
        vul_idxs (List[int]): List of vulnerable node indices.
        edge_index (torch.Tensor): Tensor of shape [2, num_edges] defining graph edges.
        binary_edge_mask (torch.Tensor): Binary tensor of shape [num_edges] with 1 for important edges.
    
    Returns:
        float: Fraction of vul_idxs covered by nodes incident to important edges.
    """
    if not vul_idxs or binary_edge_mask is None:
        return 0.0
    important_indices = torch.where(binary_edge_mask == 1)[0]
    gnn_nodes = set()
    for idx in important_indices:
        src, dst = edge_index[0, idx].item(), edge_index[1, idx].item()
        gnn_nodes.add(src)
        gnn_nodes.add(dst)
    print("GNN Path: ")
    print(gnn_nodes)
    intersection = len(set(vul_idxs) & gnn_nodes)
    return intersection / len(vul_idxs) if vul_idxs else 0.0


class VulExplainerTester:
    def __init__(self, cur_vul_type_idx, vul_type, abspath, limit, detector_idx):
        # self.explainer1: VulForwardExplainer = None
        self.vul_type = vul_type
        self.abspath = abspath
        self.detector_idx = detector_idx
        if cur_vul_type_idx == 0:
            self.visitor: SinkVisitor = BufferOverflowVisitor()
        elif cur_vul_type_idx == 1:
            self.visitor: SinkVisitor = IncorrectCalculationVisitor()
        elif cur_vul_type_idx == 2:
            self.visitor: SinkVisitor = BackwardLeakVisitor()
        elif cur_vul_type_idx == 3:
            self.visitor: SinkVisitor = PathTraversalVisitor()
        elif cur_vul_type_idx == 4:
            self.visitor: SinkVisitor = CommandInjectionVisitor()
        else:
            self.visitor: SinkVisitor = UncontrolledFormatVisitor()

        # Devign, Reveal, IVDetect
        slice_level = False
        self.explainer: VulExplainer = VulExplainer(self.visitor, slice_level)
        self.explainer.limits = limit
        # Devign doesn't use control dependence
        if detector_idx == 3:
            self.explainer.add_cdg = False

    def fromSerJson(self, serJsonData: Dict):
        cfgEdges: List[list] = [json.loads(serEdge) for serEdge in serJsonData["cfgEdges"]]
        cdgEdges: List[list] = [json.loads(serEdge) for serEdge in serJsonData["cdgEdges"]]
        ddgEdges: List[list] = [json.loads(serEdge) for serEdge in serJsonData["ddgEdges"]]
        jsonStatements: List[dict] = [json.loads(serStmt) for serStmt in serJsonData["nodes"]]
        json_data: Dict = {
            "fileName": serJsonData["fileName"],
            "functionName": serJsonData["functionName"],
            "nodes": jsonStatements,
            "cfgEdges": cfgEdges,
            "cdgEdges": cdgEdges,
            "ddgEdges": ddgEdges
        }

        return CPG.fromJson(json_data)

    def constructCPGfromXFG(self, xfg_data: Dict) -> CPG:
        cpg: CPG = CPG()
        stmts: List[ASTNode] = list()
        # load nodes
        for node_info in xfg_data["line-nodes"]:
            node_content = json.loads(node_info)
            astNode: ASTNode = json2astNode(node_content)
            stmts.append(astNode)
        cpg.statements.extend(stmts)
        # load edge
        # cdg
        cpg.CDGEdges.extend(list(map(lambda e: CodeEdge.fromJson(json.loads(e)),
                                     xfg_data["control-dependences"])))
        # ddg
        cpg.DDGEdges.extend(list(map(lambda e: CodeEdge.fromJson(json.loads(e)),
                                     xfg_data["data-dependences"])))
        return cpg


    def explain_single(self, model, cpg: CPG, data: Data):
        sink_points, cdg_prec, ddg_prec = self.explainer.identify_sink_points(cpg)
        self.explainer.slices = list()
        self.explainer.generate_backward_slices(cpg, sink_points, cdg_prec, ddg_prec)
        cur_slices: List[List[int]] = self.explainer.slices

        # if cur_vul_type_idx != 2:
        # cur_slices = list(filter(lambda slice: len(slice) > 1, cur_slices))
        sub_data_list: List[Data] = list()
        for slice in cur_slices:
            # fidelity+
            mask = torch.FloatTensor(
                [1 if i not in slice else 0 for i in range(len(data.x))]).unsqueeze(
                dim=1).to(device)
            new_data: Data = Data(x=data.x * mask, edge_index=data.edge_index)
            sub_data_list.append(new_data)
        sub_probs = torch.softmax(model(data=Batch.from_data_list(sub_data_list).to(device)), dim=1)
        sub_probs = sub_probs.cpu().tolist()
        data_dicts = [(i, value[1]) for i, value in enumerate(sub_probs)]
        # each item is a tuple (path_idx, prob)
        sorted_dicts = sorted(data_dicts, key=lambda x: x[1], reverse=False)
        selected_path_idx = sorted_dicts[0][0]
        path = cur_slices[selected_path_idx]

        # print(list(reversed(path)))
        return list(reversed(path))


    def set_random_seed(self, seed: int = 42):
        """Set random seeds for reproducibility."""
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For CUDA devices
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)

    def explain(self, model, vul_idxs_list: List[List[int]], cpgs: List[CPG], all_datas: List[Data], model_idx:int=3):
        import os
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        self.set_random_seed(42)
        from typing import List
        from dig.xgraph.method import GNNExplainer
        from gensim.models import Word2Vec
        length = len(vul_idxs_list)
        recalls = []  # VulExplainer recalls
        gnn_recalls = []  # GNNExplainer recalls
        jaccard_similarities = []  # Jaccard similarities for edge comparison
        vul_lines_identified = []

        for idx in trange(length, desc="explaining", file=sys.stdout):
            cpg = cpgs[idx]
            data: Data = all_datas[idx]
            vul_idxs = vul_idxs_list[idx]
            # print("data: ")
            # print(data)
            # print("cpg: ")
            # print(cpg)
            print("vul_idx: ")
            print(vul_idxs)
            # print("edgeIndex: ")
            # print(data.edge_index)

            # sink_points, cdg_prec, ddg_prec = self.explainer.identify_sink_points(cpg)

            sink_points, cdg_prec, ddg_prec = self.explainer.gnn_identify_sink_points(cpg=cpg, 
                                                                       word2vec_model_path="../..runs/run_full_bufferoverflow_sink_none/word2vec_model.model",
                                                                       gnn_model_path="/../../runs/run_full_bufferoverflow_sink_none/gnn_word2vec_model.pth")
            print("sink points")
            print(sink_points)
            

            self.explainer.slices = list()
            if len(sink_points) == 0:
                recalls.append(0)
                gnn_recalls.append(0)
                jaccard_similarities.append(0.0)
                vul_lines_identified.append({'sink_idx': sink_points, 'path': [], 'jaccard_similarity': 0.0, 'recall': 0.0, 'gnn_recall': 0.0})
                continue

            self.explainer.generate_backward_slices(cpg, sink_points, cdg_prec, ddg_prec)
            cur_slices: List[List[int]] = self.explainer.slices



            # Compute sub_probs for VulExplainer
            sub_data_list: List[Data] = list()
            for slice in cur_slices:
                # fidelity+
                mask = torch.FloatTensor(
                    [1 if i not in slice else 0 for i in range(len(data.x))]).unsqueeze(
                    dim=1).to(device)
                new_data: Data = Data(x=data.x * mask, edge_index=data.edge_index)
                sub_data_list.append(new_data)
            sub_probs = torch.softmax(model(data=Batch.from_data_list(sub_data_list).to(device)), dim=1)
            sub_probs = sub_probs.cpu().tolist()

            data_dicts = [(i, value[1]) for i, value in enumerate(sub_probs)]
            # print("data dicts:")
            # print(data_dicts)
            # each item is a tuple (path_idx, prob)
            sorted_dicts = sorted(data_dicts, key=lambda x: x[1], reverse=False)
            selected_path_idx = sorted_dicts[0][0]
            path = cur_slices[selected_path_idx]
            print("PATH:")
            print(path)
            graph_data = data



            # Compute recalls
            recall = len(set(vul_idxs) & set(path)) / len(vul_idxs) if vul_idxs else 0.0
            recalls.append(recall)

            ################################################################################################
            x = graph_data.x
            # Normalize inputs to prevent numerical instability
            x = (x - x.mean()) / (x.std() + 1e-8)
            # print("X: ", x)
            # print(f"edge_index: {edge_index}")
            # print("##########################")
            # print("x shape: ", x.shape)
            # print("edge_index shape: ", edge_index.shape)
            # print("##########################")
            edge_index = graph_data.edge_index
            # print("Edge index")
            # print(edge_index)
            target_label = graph_data.y

            # Debug: Check model output
            with torch.no_grad():
                try:
                    preds = model(x=x, edge_index=edge_index)
                    # print("DevignModel predictions:", preds)
                    if torch.isnan(preds).any() or torch.isinf(preds).any():
                        print("Warning: Model predictions contain NaN or Inf values")
                except Exception as e:
                    print(f"Error in model prediction: {e}")
                    preds = None

            # Debug: Check if DevignModel supports edge_weight
            supports_edge_weight = True
            with torch.no_grad():
                try:
                    test_mask = torch.ones(edge_index.size(1), device=x.device)
                    pred = model(x=x, edge_index=edge_index, edge_weight=test_mask)
                    # print("DevignModel with edge_weight:", pred)
                except Exception as e:
                    print(f"DevignModel does not support edge_weight: {e}")
                    supports_edge_weight = False

            # ⚙️ 3) Instantiate YOUR explainer
            explainer = GNNExplainer(
                model=model,  # Fixed: Use devign_model
                epochs=100,  # Increased for better convergence
                lr=0.002,  # Reduced for stability
                explain_graph=True
            )
            explainer.coeffs = {
                'edge_size': 0.001,
                'edge_ent': 0.001,
                'node_feat_size': 1.0,
                'node_feat_ent': 0.1,
            }

            # ⚙️ 4) Run your custom explainer with error handling
            try:
                _, edge_masks, _ = explainer(
                    x=data.x,
                    edge_index=data.edge_index,
                    mask_features=False,
                    num_classes=2,
                    sparsity=0.7,
                )
                # Clamp edge masks to prevent inf/-inf
                # edge_masks = [torch.clamp(mask, min=-10, max=10) for mask in edge_masks] if edge_masks else None
                # print("Edge masks (continuous):", edge_masks)
                # Convert to binary masks
                def to_binary_mask(mask):
                    """
                    Convert a continuous edge mask with inf/-inf to a binary mask (1 for inf, 0 for -inf).
                    
                    Args:
                        mask (torch.Tensor): Continuous mask with inf/-inf values.
                    
                    Returns:
                        torch.Tensor: Binary mask with 1s for inf and 0s for -inf.
                    """
                    if mask is None:
                        return None
                    binary_mask = torch.where(mask == float('inf'), torch.tensor(1.0, device=mask.device),
                                            torch.tensor(0.0, device=mask.device))
                    return binary_mask
                binary_edge_masks = [to_binary_mask(mask) for mask in edge_masks] if edge_masks else None
                # print("Edge masks (binary):", binary_edge_masks)
            except Exception as e:
                print(f"Error in GNNExplainer: {e}")
                edge_masks = None
                binary_edge_masks = None

            # Manually compute related_preds to avoid NoneType error
            # if edge_masks:
            #     related_preds = []
            #     with torch.no_grad():
            #         self_loop_edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
            #         for label, edge_mask in enumerate(edge_masks):
            #             try:
            #                 # Original prediction (no edge mask)
            #                 orig_pred = devign_model(x=x, edge_index=edge_index)
            #                 # No edge mask (all edges with weight 1)
            #                 no_mask_pred = devign_model(x=x, edge_index=self_loop_edge_index)
            #                 # With edge mask
            #                 if supports_edge_weight and binary_edge_masks and binary_edge_masks[label] is not None:
            #                     mask_pred = devign_model(x=x, edge_index=self_loop_edge_index, edge_weight=binary_edge_masks[label])
            #                 else:
            #                     mask_pred = devign_model(x=x, edge_index=self_loop_edge_index)
            #                 # With complement of edge mask
            #                 if supports_edge_weight and binary_edge_masks and binary_edge_masks[label] is not None:
            #                     maskout_pred = devign_model(x=x, edge_index=self_loop_edge_index, edge_weight=(1 - binary_edge_masks[label]))
            #                 else:
            #                     maskout_pred = devign_model(x=x, edge_index=self_loop_edge_index)
            #                 related_preds.append({
            #                     'label': label,
            #                     'origin': orig_pred.softmax(dim=-1)[0][label].item(),
            #                     'zero': no_mask_pred.softmax(dim=-1)[0][label].item(),
            #                     'masked': mask_pred.softmax(dim=-1)[0][label].item(),
            #                     'maskout': maskout_pred.softmax(dim=-1)[0][label].item()
            #                 })
            #             except Exception as e:
            #                 print(f"Error computing related_preds for label {label}: {e}")
            #                 related_preds.append(None)
            # else:
            #     related_preds = None

            # print("Related predictions:", related_preds)

           
            # Compare VulExplainer path with GNNExplainer edge mask (class 1)
            jaccard_similarity = 0.0
            if binary_edge_masks and binary_edge_masks[1] is not None and path:
                # print("edge index")
                # print(edge_index)
                # print("binary_edge_masks")
                # print(binary_edge_masks)

                jaccard_similarity = compare_path_and_edge_mask(path, edge_index, binary_edge_masks[1])
                print(f"Jaccard Similarity (VulExplainer path vs. GNNExplainer edges for class 1): {jaccard_similarity:.3f}")
            else:
                print("Skipping Jaccard similarity: No valid path or edge mask")
            jaccard_similarities.append(jaccard_similarity)


            ################################################################################################
            
            gnn_recall = compute_gnn_recall(vul_idxs, edge_index, binary_edge_masks[1] if binary_edge_masks else None)
            gnn_recalls.append(gnn_recall)
            print(f"VulPathFinder Recall: {recall:.3f}")
            print(f"SliceLocator Recall: {recall:.3f}")
            print(f"GNNExplainer Recall: {gnn_recall:.3f}")
            print(f"Recall Difference (VulExplainer - GNNExplainer): {(recall - gnn_recall):.3f}")

            cur_slices_unique = [list(x) for x in set(tuple(slice) for slice in cur_slices)]
            vul_lines_identified.append({
                'sink_idx': sink_points,
                'path': cur_slices_unique,
                'jaccard_similarity': jaccard_similarity,
                'recall': recall,
                'gnn_recall': gnn_recall,
                'recall_difference': recall - gnn_recall
            })

        with open('vul_paths.json', 'w') as f:
            json.dump(vul_lines_identified, f, indent=4)
        print(f"Triggering line coverage (VulExplainer): {np.nanmean(recalls):.3f}")
        print(f"Triggering line coverage (GNNExplainer): {np.nanmean(gnn_recalls):.3f}")
        print(f"Average Jaccard Similarity (VulExplainer vs. GNNExplainer): {np.nanmean(jaccard_similarities):.3f}")
        print("======================")


    def process_ivdetect(self):
        test_datas: List[Dict] = json.load(
            open(os.path.join(ivdetect_data_args.dataset_dir.format(self.abspath, self.vul_type), data_path[self.detector_idx]),
                 'r', encoding='utf-8'))
        print(len(test_datas))
        print("=================")

        pretrain_model = Word2Vec.load(ivdetect_model_args.pretrain_word2vec_model.format(self.abspath, self.vul_type))

        # checkpoint = torch.load(
        #     os.path.join(ivdetect_model_args.model_dir.format(self.abspath, self.vul_type),
        #                  f'{ivdetect_model_args.model_name}_{ivdetect_model_args.detector}_best.pth'))

        model: IVDetectModel = IVDetectModel()
        # model.load_state_dict(checkpoint['net'])

        ##############################################
        
        import torch.serialization
        import numpy.core.multiarray

        # Allow the specific numpy global that's being blocked
        with torch.serialization.safe_globals([numpy.core.multiarray.scalar]):
            checkpoint = torch.load(
            os.path.join(ivdetect_model_args.model_dir.format(self.abspath, self.vul_type),
                 f'{ivdetect_model_args.model_name}_{ivdetect_model_args.detector}_best.pth'),
            weights_only=False
            )

        # Create a mapping from old to new keys
        key_mapping = {
            "convs.0.weight": "convs.0.lin.weight",
            "convs.1.weight": "convs.1.lin.weight", 
            "convs.2.weight": "convs.2.lin.weight"
        }

        # Rename the keys
        new_state_dict = {}
        for key, value in checkpoint['net'].items():
            new_key = key_mapping.get(key, key)
            new_state_dict[new_key] = value

        # Load the modified state dict
        model.load_state_dict(new_state_dict)
        ##############################################
        model.to(device)
        model.eval()


        ivdetect_util: IVDetectUtil = IVDetectUtil(pretrain_model)

        vul_features = [ivdetect_util.generate_all_features(sample) for sample in
                             tqdm(test_datas, desc="generating feature for vul sample", file=sys.stdout)]
        all_datas: List[Data] = [model.vectorize_graph(feature) for feature in tqdm(vul_features, desc="vectorizing data",
                                                                                    file=sys.stdout)]
        cpgs: List[CPG] = [self.fromSerJson(sample) for sample in
                           tqdm(test_datas, desc="restoring CPG", file=sys.stdout)]
        vul_idxs_list: List[List[int]] = [sample["vul_idxs"] for sample in test_datas]
        self.explain(model, vul_idxs_list, cpgs, all_datas, model_idx=2)

    def process_reveal(self):
        test_datas: List[Dict] = json.load(
            open(os.path.join(reveal_data_args.dataset_dir.format(self.abspath, self.vul_type), data_path[self.detector_idx]),
                 'r', encoding='utf-8'))
        print(len(test_datas))
        print("=================")

        # load model
        pretrain_model = Word2Vec.load(reveal_model_args.pretrain_word2vec_model.format(self.abspath, self.vul_type))
        model: ClassifyModel = ClassifyModel()
        model.to(device)
        model.eval()

        # checkpoint = torch.load(
        #     os.path.join(reveal_model_args.model_dir.format(self.abspath, self.vul_type),
        #                  f'{reveal_model_args.model_name}_{reveal_model_args.detector}_best.pth'))
        ##############################################
        
        import torch.serialization
        import numpy.core.multiarray

        # Allow the specific numpy global that's being blocked
        with torch.serialization.safe_globals([numpy.core.multiarray.scalar]):
            checkpoint = torch.load(
            os.path.join(reveal_model_args.model_dir.format(self.abspath, self.vul_type),
                 f'{reveal_model_args.model_name}_{reveal_model_args.detector}_best.pth'),
            weights_only=False
            )

        # Create a mapping from old to new keys
        key_mapping = {
            "convs.0.weight": "convs.0.lin.weight",
            "convs.1.weight": "convs.1.lin.weight", 
            "convs.2.weight": "convs.2.lin.weight"
        }

        # Rename the keys
        new_state_dict = {}
        for key, value in checkpoint['net'].items():
            new_key = key_mapping.get(key, key)
            new_state_dict[new_key] = value

        # Load the modified state dict
        model.load_state_dict(new_state_dict)
        ##############################################

        # model.load_state_dict(checkpoint['net'])
        reveal_util: RevealUtil = RevealUtil(pretrain_model, model)

        graph_infos: List[tuple] = [reveal_util.generate_initial_training_datas(sample) for sample in tqdm(test_datas,
                                                                        desc="parsing raw datas", file=sys.stdout)]
        all_datas: List[Data] = [reveal_util.generate_initial_graph_embedding(graph_info)
                                 for graph_info in tqdm(graph_infos, desc="embedding datas", file=sys.stdout)]
        cpgs: List[CPG] = [self.fromSerJson(sample) for sample in tqdm(test_datas, desc="restoring CPG", file=sys.stdout)]
        vul_idxs_list: List[List[int]] = [sample["vul_idxs"] for sample in test_datas]
        self.explain(model, vul_idxs_list, cpgs, all_datas, model_idx=1)

    def process_devign(self):
        test_datas: List[Dict] = json.load(
            open(os.path.join(devign_data_args.dataset_dir.format(self.abspath, self.vul_type), data_path[self.detector_idx]),
                 'r', encoding='utf-8'))
        
        # load model

        # checkpoint = torch.load(
        # os.path.join(devign_model_args.model_dir.format(self.abspath, self.vul_type),
        #                 f'{devign_model_args.model_name}_{devign_model_args.detector}_best.pth'))
    
        pretrain_model = Word2Vec.load(devign_model_args.pretrain_word2vec_model.format(self.abspath, self.vul_type))
        model: DevignModel = DevignModel()
        model.to(devign_model_args.device)
        # model.load_state_dict(checkpoint['net'])
        ##############################################
        
        import torch.serialization
        import numpy.core.multiarray

        # Allow the specific numpy global that's being blocked
        with torch.serialization.safe_globals([numpy.core.multiarray.scalar]):
            checkpoint = torch.load(
            os.path.join(devign_model_args.model_dir.format(self.abspath, self.vul_type),
             f'{devign_model_args.model_name}_{devign_model_args.detector}_best.pth'),
            weights_only=False
            )

        # Create a mapping from old to new keys
        key_mapping = {
            "convs.0.weight": "convs.0.lin.weight",
            "convs.1.weight": "convs.1.lin.weight", 
            "convs.2.weight": "convs.2.lin.weight"
        }

        # Rename the keys
        new_state_dict = {}
        for key, value in checkpoint['net'].items():
            new_key = key_mapping.get(key, key)
            new_state_dict[new_key] = value

        # Load the modified state dict
        model.load_state_dict(new_state_dict)
        ##############################################
        

        devign_util = DevignUtil(pretrain_model, model)
        graph_infos: List[tuple] = [devign_util.generate_initial_training_datas(sample) for sample in tqdm(test_datas,
                                                                                                           desc="parsing raw datas",
                                                                                                           file=sys.stdout)]
        # print('graph_infos:\n\n')
        # print(graph_infos)
        
        all_datas: List[Data] = [devign_util.generate_initial_graph_embedding(graph_info)
                                 for graph_info in tqdm(graph_infos, desc="embedding datas", file=sys.stdout)]
        cpgs: List[CPG] = [self.fromSerJson(sample) for sample in
                           tqdm(test_datas, desc="restoring CPG", file=sys.stdout)]
        vul_idxs_list: List[List[int]] = [sample["vul_idxs"] for sample in test_datas]
        # print('graph_infos:\n\n')
        # print(graph_infos)
        # print('\n\nall_datas\n\n')
        # print(all_datas)
        # print('\n\ncpgs\n\n')
        # print(cpgs)
        # print('\n\nvul_idxs_list\n\n')
        # print(vul_idxs_list)

        self.explain(model, vul_idxs_list, cpgs, all_datas, model_idx=3)


    def generate_html_visualization(self, samples, sample_indices, trad_paths_info, gnn_paths_info, vul_idxs_list, metrics):
        """Generate an HTML file to compare sink points, paths, ground truth vulnerabilities, and overall accuracies."""
        base_dir = "/datasets/buffer overflow/function/testcases"
        
        # Ensure paths_info lists match the number of samples
        trad_paths_info = trad_paths_info + [{'sink_points': [], 'paths': [], 'vul_idxs': [], 'recall': 0.0} for _ in range(len(samples) - len(trad_paths_info))]
        gnn_paths_info = gnn_paths_info + [{'sink_points': [], 'paths': [], 'vul_idxs': [], 'recall': 0.0} for _ in range(len(samples) - len(gnn_paths_info))]
        
        select_options = ""
        for sample_idx, sample in zip(sample_indices, samples):
            filename = html.escape(sample.get('fileName', 'unknown'))
            select_options += f'<option value="{sample_idx}">{sample_idx}: {filename}</option>\n'
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Sink Points and Vulnerabilities Comparison</title>
            <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
            <style>
                html, body {{ width: 100%; margin: 0; padding: 20px; box-sizing: border-box; overflow: auto; }}
                .sample {{ display: none; margin-bottom: 40px; }}
                .sample.active {{ display: block; }}
                .container {{ display: flex; width: 100%; max-width: 100%; flex-wrap: wrap; }}
                .code {{ flex: 1; margin: 10px; min-width: 300px; max-width: 33%; }}
                .graph {{ flex: 1; margin: 10px; min-width: 300px; }}
                .network {{ width: 100%; height: 400px; border: 1px solid #ccc; }}
                .legend {{ margin-top: 20px; }}
                .legend div {{ margin: 5px; }}
                select {{ margin-bottom: 20px; padding: 5px; font-size: 16px; width: 100%; max-width: 500px; }}
                pre {{ white-space: pre-wrap; word-wrap: break-word; }}
                .method {{ margin-bottom: 20px; }}
                .paths {{ margin-top: 10px; }}
                .sink {{ background-color: #ccccff; }}
                .vul {{ background-color: #ffccff; }}
                .metrics {{ margin-top: 20px; }}
            </style>
        </head>
        <body>
            <h1>Sink Points and Vulnerabilities Comparison</h1>
            <div>
                <label for="sampleSelect">Select Sample:</label>
                <select id="sampleSelect" onchange="showSample(this.value)">
                    {select_options}
                </select>
            </div>
        """
        
        for idx, (sample, sample_idx) in enumerate(zip(samples, sample_indices)):
            nodes = sample["nodes"]
            
            filename = sample.get('fileName', '')
            if not filename:
                logging.error(f"No filename found for sample {sample_idx}")
                code_lines = ["/* Error: No filename specified */"]
            else:
                cwe_category = filename.split('__')[0]
                cwe_dir = os.path.join(base_dir, cwe_category)
                code_lines = None
                try:
                    for subfolder in os.listdir(cwe_dir):
                        if subfolder.startswith('s') and subfolder[1:].isdigit():
                            file_path = os.path.join(cwe_dir, subfolder, filename)
                            if os.path.exists(file_path):
                                with open(file_path, 'r', encoding='utf-8') as f:
                                    code_lines = f.readlines()
                                break
                    if code_lines is None:
                        logging.error(f"File {filename} not found in any subfolder of {cwe_dir}")
                        code_lines = [f"/* Error: File {filename} not found */"]
                except Exception as e:
                    logging.error(f"Failed to search for file {filename} in {cwe_dir}: {e}")
                    code_lines = [f"/* Error: Could not access file {filename} */"]
            
            trad_info = trad_paths_info[idx]
            gnn_info = gnn_paths_info[idx]
            trad_sink_points = trad_info.get('sink_points', [])
            gnn_sink_points = gnn_info.get('sink_points', [])
            trad_paths = trad_info.get('paths', [])
            gnn_paths = gnn_info.get('paths', [])
            vul_idxs = vul_idxs_list[idx] if idx < len(vul_idxs_list) else []
            
            # Map node indices to line numbers
            node_to_line = {}
            for node_idx, node in enumerate(nodes):
                node_dict = json.loads(node)
                line_num = node_dict["line"]
                node_to_line[node_idx] = line_num
            
            trad_sink_lines = [node_to_line.get(idx, -1) for idx in trad_sink_points if idx in node_to_line]
            gnn_sink_lines = [node_to_line.get(idx, -1) for idx in gnn_sink_points if idx in node_to_line]
            vul_lines = [node_to_line.get(idx, -1) for idx in vul_idxs if idx in node_to_line]
            
            # Generate code HTML for traditional method
            trad_code_html = "<pre style='font-family: monospace; line-height: 1.2;'>"
            for line_num, line in enumerate(code_lines, 1):
                line = line.rstrip('\n')
                if not line.strip():
                    trad_code_html += f"<div>{html.escape(f'{line_num:>4} | ')}</div>"
                    continue
                style = " class='sink'" if line_num in trad_sink_lines else ""
                trad_code_html += f"<div{style}>{html.escape(f'{line_num:>4} | {line}')}</div>"
            trad_code_html += "</pre>"
            
            # Generate code HTML for GNN method
            gnn_code_html = "<pre style='font-family: monospace; line-height: 1.2;'>"
            for line_num, line in enumerate(code_lines, 1):
                line = line.rstrip('\n')
                if not line.strip():
                    gnn_code_html += f"<div>{html.escape(f'{line_num:>4} | ')}</div>"
                    continue
                style = " class='sink'" if line_num in gnn_sink_lines else ""
                gnn_code_html += f"<div{style}>{html.escape(f'{line_num:>4} | {line}')}</div>"
            gnn_code_html += "</pre>"
            
            # Generate code HTML for ground truth
            vul_code_html = "<pre style='font-family: monospace; line-height: 1.2;'>"
            for line_num, line in enumerate(code_lines, 1):
                line = line.rstrip('\n')
                if not line.strip():
                    vul_code_html += f"<div>{html.escape(f'{line_num:>4} | ')}</div>"
                    continue
                style = " class='vul'" if line_num in vul_lines else ""
                vul_code_html += f"<div{style}>{html.escape(f'{line_num:>4} | {line}')}</div>"
            vul_code_html += "</pre>"
            
            # Generate paths HTML with importance scores
            trad_paths_html = "<div class='paths'><strong>Paths (Importance Score):</strong><ul>"
            for path_info in trad_paths:
                path = path_info['path']
                score = path_info['score']
                path_lines = [node_to_line.get(idx, -1) for idx in path if idx in node_to_line]
                trad_paths_html += f"<li>{html.escape(str(path_lines))} (Score: {score:.4f})</li>"
            trad_paths_html += "</ul></div>"
            
            gnn_paths_html = "<div class='paths'><strong>Paths (Importance Score):</strong><ul>"
            for path_info in gnn_paths:
                path = path_info['path']
                score = path_info['score']
                path_lines = [node_to_line.get(idx, -1) for idx in path if idx in node_to_line]
                gnn_paths_html += f"<li>{html.escape(str(path_lines))} (Score: {score:.4f})</li>"
            gnn_paths_html += "</ul></div>"
            
            # Generate graph data for Vis.js
            vis_nodes = []
            vis_edges = []
            for i, node in enumerate(nodes):
                node_dict = json.loads(node)
                node_type = node_dict["contents"][0][0]
                node_content = node_dict["contents"][0][1][:20]
                is_trad_sink = i in trad_sink_points
                is_gnn_sink = i in gnn_sink_points
                is_vul = i in vul_idxs
                if is_vul:
                    color = "#ffccff"  # Ground truth vulnerability
                    label = "Vulnerability"
                elif is_trad_sink and is_gnn_sink:
                    color = "#6666ff"  # Both methods
                    label = "Sink (Both)"
                elif is_trad_sink:
                    color = "#ff6666"  # Traditional only
                    label = "Sink (Traditional)"
                elif is_gnn_sink:
                    color = "#66ff66"  # GNN only
                    label = "Sink (GNN)"
                else:
                    color = "#cccccc"  # Non-sink
                    label = "Non-Sink"
                vis_nodes.append({
                    "id": i,
                    "label": f"{node_type}\n{node_content}",
                    "color": color,
                    "title": label
                })
            
            for edge_type in ["cfgEdges", "ddgEdges", "cdgEdges"]:
                color = {"cfgEdges": "#0000ff", "ddgEdges": "#ff0000", "cdgEdges": "#00ff00"}[edge_type]
                for edge in sample[edge_type]:
                    edge_data = json.loads(edge)
                    src, dst = edge_data[0], edge_data[1]
                    vis_edges.append({
                        "from": src,
                        "to": dst,
                        "color": color,
                        "arrows": "to",
                        "label": edge_type[:3].upper()
                    })
            
            html_content += f"""
            <div class="sample" id="sample_{sample_idx}">
                <h2>Sample {sample_idx} ({html.escape(filename)})</h2>
                <div class="container">
                    <div class="method code">
                        <h3>Traditional Method</h3>
                        {trad_code_html}
                        {trad_paths_html}
                    </div>
                    <div class="method code">
                        <h3>GNN Method</h3>
                        {gnn_code_html}
                        {gnn_paths_html}
                    </div>
                    <div class="method code">
                        <h3>Ground Truth</h3>
                        {vul_code_html}
                    </div>
                    <div class="graph">
                        <h3>Graph Visualization</h3>
                        <div id="network_{sample_idx}" class="network"></div>
                    </div>
                </div>
                <script>
                    var nodes_{sample_idx} = new vis.DataSet({json.dumps(vis_nodes)});
                    var edges_{sample_idx} = new vis.DataSet({json.dumps(vis_edges)});
                    var container = document.getElementById('network_{sample_idx}');
                    var data = {{ nodes: nodes_{sample_idx}, edges: edges_{sample_idx} }};
                    var options = {{
                        nodes: {{ shape: 'dot', size: 20, font: {{ size: 12 }} }},
                        edges: {{ font: {{ size: 10 }} }},
                        physics: {{ hierarchicalRepulsion: {{ nodeDistance: 120 }} }},
                        layout: {{ hierarchical: {{ direction: 'UD', sortMethod: 'directed' }} }}
                    }};
                    var network = new vis.Network(container, data, options);
                </script>
            </div>
            """
        
        html_content += f"""
            <div class="legend">
                <h3>Legend</h3>
                <div style="background-color: #ffccff;">Ground Truth Vulnerability</div>
                <div style="background-color: #6666ff;">Sink (Both Methods)</div>
                <div style="background-color: #ff666;">Sink (Traditional Only)</div>
                <div style="background-color: #66ff66;">Sink (GNN Only)</div>
                <div style="background-color: #cccccc;">Non-Sink</div>
                <div>Edges: Blue (CFG), Red (DDG), Green (CDG)</div>
            </div>
            <div class="metrics">
                <h3>Overall Metrics</h3>
                <p>Traditional Method Triggering Line Coverage: {metrics['traditional_accuracy']:.4f}</p>
                <p>GNN Method Triggering Line Coverage: {metrics['gnn_accuracy']:.4f}</p>
            </div>
            <script>
                function showSample(sampleId) {{
                    document.querySelectorAll('.sample').forEach(sample => {{
                        sample.classList.remove('active');
                    }});
                    var selectedSample = document.getElementById('sample_' + sampleId);
                    if (selectedSample) {{
                        selectedSample.classList.add('active');
                    }}
                }}
                showSample('{sample_indices[0] if sample_indices else ''}');
            </script>
        </body>
        </html>
        """
        
        os.makedirs("visualizations", exist_ok=True)
        filename = "visualizations/sink_vul_comparison.html"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(html_content)
        logging.info(f"Generated visualization: {filename}")

