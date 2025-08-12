from gensim.models import Word2Vec
import numpy as np
import json
from typing import Dict, List, Tuple, Set
from detectors.IVDetect.configurations import model_args
import torch
from .treelstm import calculate_evaluation_orders

from CppCodeAnalyzer.mainTool.ast.astNode import ASTNode
from CppCodeAnalyzer.mainTool.ast.builders import json2astNode
from CppCodeAnalyzer.extraTools.vuldetect.ivdetect import lexical_parse, generate_feature3, find_control, find_data

class IVDetectUtil(object):
    def __init__(self, w2v_model: Word2Vec):
        self.pretrain_model = w2v_model

    def generate_feature4(self, cpg: Dict, limit: int = 1) -> List[List[List[str]]]:
        edges = [json.loads(edge) for edge in cpg["cdgEdges"]]
        cdg_edge_idxs: Dict[int, int] = {edge[1]: edge[0] for edge in edges}
        nodes_infos: List[Dict] = cpg["nodes"]
        #  每个statement的控制依赖结点
        cd_idxs_for_stmt: List[List[int]] = list()
        for stmt_idx in range(len(nodes_infos)):
            seq: List[int] = list()
            find_control(stmt_idx, cdg_edge_idxs, seq, 1, limit)
            cd_idxs_for_stmt.append(seq)

        feature4_for_stmts: List[List[List[str]]] = list()
        for cd_idxs in cd_idxs_for_stmt:
            sub_tokens_in_stmts: List[List[str]] = [lexical_parse(nodes_infos[idx]["contents"][0][1])
                                                    for idx in cd_idxs]
            feature4_for_stmts.append(sub_tokens_in_stmts)

        return feature4_for_stmts

    def generate_feature5(self, cpg: Dict, limit: int = 1) -> List[List[List[str]]]:
        edges = [json.loads(edge) for edge in cpg["ddgEdges"]]
        ddg_edge_idxs: Dict[int, Set[int]] = dict()
        for edge in edges:
            # source -> destination
            if edge[1] not in ddg_edge_idxs.keys():
                ddg_edge_idxs[edge[1]] = {edge[0]}
            else:
                ddg_edge_idxs[edge[1]].add(edge[0])

            # destination -> source
            if edge[0] not in ddg_edge_idxs.keys():
                ddg_edge_idxs[edge[0]] = {edge[1]}
            else:
                ddg_edge_idxs[edge[0]].add(edge[1])

        nodes_infos: List[Dict] = cpg["nodes"]
        #  每个statement的控制依赖结点
        dd_idxs_for_stmt: List[List[int]] = list()
        for stmt_idx in range(len(nodes_infos)):
            seq: List[int] = list()
            find_data(stmt_idx, ddg_edge_idxs, seq, 1, limit)
            dd_idxs_for_stmt.append(seq)

        feature5_for_stmts: List[List[List[str]]] = list()
        for dd_idxs in dd_idxs_for_stmt:
            sub_tokens_in_stmts: List[List[str]] = [lexical_parse(nodes_infos[idx]["contents"][0][1])
                                                    for idx in dd_idxs]
            feature5_for_stmts.append(sub_tokens_in_stmts)
        return feature5_for_stmts

    def generate_all_features(self, data: Dict) -> Tuple[List[torch.Tensor], List[Tuple], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor],
                                        torch.LongTensor, int]:
        ## generate feature 1, sub token list
        nodes_infos: List[Dict] = [json.loads(sample) if isinstance(sample, str) else sample for sample in data["nodes"]]
        temp = data["nodes"]
        data["nodes"] = nodes_infos
        sub_tokens_list: List[List[str]] = [lexical_parse(node_infos["contents"][0][1]) for node_infos in nodes_infos]
        # vectorizing sub token list
        # 所有statement的token list
        feature1 = []
        for i, sub_tokens in enumerate(sub_tokens_list):
            stmt_vector = []
            for j, sub_token in enumerate(sub_tokens):
                if sub_token in self.pretrain_model.wv.key_to_index:
                    stmt_vector.append(self.pretrain_model.wv[sub_token])  # Updated from self.pretrain_model
                else:
                    stmt_vector.append(np.zeros(shape=(model_args.feature_representation_size)))
            if len(stmt_vector) == 0:
                stmt_vector.append(np.zeros(shape=(model_args.feature_representation_size)))
            stmt_vector = np.stack(stmt_vector)
            feature1.append(torch.from_numpy(stmt_vector).to(model_args.device))

        ## generate feature2, AST subtrees
        # each ast subtree is parsed into node feature, edge
        feature2: List[Tuple] = list()
        for ast in nodes_infos:
            edges = ast["edges"]
            if len(edges) == 0:
                tokens: List[str] = lexical_parse(ast["contents"][0][1])
                if len(tokens) == 0:
                    feature2.append((torch.zeros(size=(model_args.feature_representation_size,)).to(model_args.device),))
                    continue
                vecs = [self.pretrain_model.wv[token] if token in self.pretrain_model.wv.key_to_index  # Updated from self.pretrain_model
                        else np.zeros(shape=(model_args.feature_representation_size)) for token in tokens]
                vecs = np.stack(vecs).mean(axis=0)
                stmt_vector = torch.from_numpy(vecs).to(model_args.device)
                feature2.append((stmt_vector,))
                continue

            edges = torch.LongTensor(edges)
            stmt_vectors = []
            for node in ast["contents"]:
                tokens: List[str] = lexical_parse(node[1])
                if len(tokens) == 0:
                    stmt_vector = np.zeros(shape=(model_args.feature_representation_size))
                else:
                    stmt_vector = np.array([self.pretrain_model.wv[token] if token in self.pretrain_model.wv.key_to_index  # Updated from self.pretrain_model
                                            else np.zeros(shape=(model_args.feature_representation_size)) for token in tokens]).mean(axis=0)
                stmt_vectors.append(stmt_vector)
            features = torch.from_numpy(np.stack(stmt_vectors))
            node_order, edge_order = calculate_evaluation_orders(edges, len(features))
            feature2.append((features.to(model_args.device), edges.to(model_args.device), node_order, edge_order))

        ## generate feature3, variable list
        astNodes: List[ASTNode] = [json2astNode(node_infos) for node_infos in nodes_infos]
        varLists: List[List[str]] = generate_feature3(astNodes)
        # vectorizing sub token list
        feature3 = []
        for i, varList in enumerate(varLists):
            stmt_vector = []
            for j, var in enumerate(varList):
                if var in self.pretrain_model.wv.key_to_index:
                    stmt_vector.append(self.pretrain_model.wv[var])  # Updated from self.pretrain_model
                else:
                    stmt_vector.append(np.zeros(shape=(model_args.feature_representation_size)))
            if len(stmt_vector) == 0:

                stmt_vector.append(np.zeros(shape=(model_args.feature_representation_size)))
            stmt_vector = np.stack(stmt_vector)
            feature3.append(torch.from_numpy(stmt_vector).to(model_args.device))

        ## generate feature4, control dependence list
        feature4_for_stmts: List[List[List[str]]] = self.generate_feature4(data, 1)

        # List[List[str]]
        feature4 = []
        for feature4_for_stmt in feature4_for_stmts:
            if len(feature4_for_stmt) == 0:
                feature4.append(torch.zeros(size=(1, model_args.feature_representation_size)).to(model_args.device))
                continue
            # List[str]
            vectors = []
            for context in feature4_for_stmt:
                # 一维向量
                if len(context) == 0:
                    stmt_vector = np.zeros(shape=(model_args.feature_representation_size))
                else:
                    stmt_vector = np.array([self.pretrain_model.wv[token] if token in self.pretrain_model.wv.key_to_index  # Updated from self.pretrain_model
                                            else np.zeros(shape=(model_args.feature_representation_size)) for token in context]).mean(axis=0)
                vectors.append(stmt_vector)
            vectors = np.stack(vectors)
            feature4.append(torch.from_numpy(vectors).to(model_args.device))

        # generate feature 5, data dependence list
        feature5_for_stmts: List[List[List[str]]] = self.generate_feature5(data, 1)

        # List[List[str]]
        feature5 = []
        for feature5_for_stmt in feature5_for_stmts:
            if len(feature5_for_stmt) == 0:
                feature5.append(torch.zeros(size=(1, model_args.feature_representation_size)).to(model_args.device))
                continue
            # vector
            vectors = []
            # List[str]
            for context in feature5_for_stmt:
                # 一维向量
                if len(context) == 0:
                    stmt_vector = np.zeros(shape=(model_args.feature_representation_size))
                else:
                    stmt_vector = np.array([self.pretrain_model.wv[token] if token in self.pretrain_model.wv.key_to_index  # Updated from self.pretrain_model
                                            else np.zeros(shape=(model_args.feature_representation_size)) for token in context]).mean(axis=0)
                vectors.append(stmt_vector)
            vectors = np.stack(vectors)
            feature5.append(torch.from_numpy(vectors).to(model_args.device))

        # edge indexes
        edges = [json.loads(edge)[:2] for edge in data["ddgEdges"]] + [json.loads(edge) for edge in data["cdgEdges"]]
        edge_index: torch.LongTensor = torch.LongTensor(edges).t().to(model_args.device)
        # label
        data["nodes"] = temp
        return (feature1, feature2, feature3, feature4, feature5, edge_index, data["target"])

if __name__ == '__main__':
    pretrain_model = Word2Vec.load(model_args.pretrain_word2vec_model)
    ivdetect_util = IVDetectUtil(pretrain_model)