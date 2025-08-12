from gensim.models import Word2Vec
from torch_geometric.data import Batch, Data
import torch
import os

from detectors.Reveal.configurations import model_args as reveal_model_args
from detectors.Reveal.model import ClassifyModel
from detectors.Reveal.util import RevealUtil

from detectors.Devign.model import DevignModel
from detectors.Devign.util import DevignUtil
from detectors.Devign.configurations import model_args as devign_model_args

from detectors.IVDetect.model import IVDetectModel
from detectors.IVDetect.configurations import model_args as ivdetect_model_args
from detectors.IVDetect.util import IVDetectUtil

from eval_utils.evaluate_vulexplainer import VulExplainerTester
from CppCodeAnalyzer.mainTool.CPG import CPG
from CppCodeAnalyzer.mainTool.CPG import initialCalleeInfos, CFGToUDGConverter, ASTDefUseAnalyzer, CFGAndUDGToDefUseCFG, DDGCreator, fileParse, astNodeToJson

from typing import Dict, List, Tuple
import json


def toJsonCPG(cpg: CPG):
    jsonStatements: List[str] = [json.dumps(astNodeToJson(statement)) for statement in cpg.statements]
    serializedCfgEdges: List[str] = [json.dumps(edge.toJson()) for edge in cpg.CFGEdges]
    serializedCdgEdges: List[str] = [json.dumps(edge.toJson()) for edge in cpg.CDGEdges]
    serializedDdgEdges: List[str] = [json.dumps(edge.toJson()) for edge in cpg.DDGEdges]
    lines: List[int] = [statement.location.startLine for statement in cpg.statements]

    return {
        "fileName": cpg.file,
        "functionName": cpg.name,
        "nodes": jsonStatements,
        "cfgEdges": serializedCfgEdges,
        "cdgEdges": serializedCdgEdges,
        "ddgEdges": serializedDdgEdges,
        "lines": lines,
        "target": 0
    }


def evaluate_single_file_util(file: str, calleeInfoFile: str, vul_type: str, detector_idx: int, vul_idx: int, limit: int):
    calleeInfs = json.load(open(calleeInfoFile, 'r', encoding='utf-8'))
    calleeInfos = initialCalleeInfos(calleeInfs)

    converter: CFGToUDGConverter = CFGToUDGConverter()
    astAnalyzer: ASTDefUseAnalyzer = ASTDefUseAnalyzer()
    astAnalyzer.calleeInfos = calleeInfos
    converter.astAnalyzer = astAnalyzer

    defUseConverter: CFGAndUDGToDefUseCFG = CFGAndUDGToDefUseCFG()
    ddgCreator: DDGCreator = DDGCreator()

    cpgs: List[CPG] = fileParse(file, converter, defUseConverter, ddgCreator)
    json_cpgs: List[Dict] = [toJsonCPG(cpg) for cpg in cpgs]

    print(f"detected vulnerability type: {vul_type}")

    if detector_idx == 1:
        evaluate_with_Reveal(vul_idx, vul_type, os.getcwd(), json_cpgs, limit)
    elif detector_idx == 2:
        evaluate_IVDetect(vul_idx, vul_type, os.getcwd(), json_cpgs, limit)
    elif detector_idx == 3:
        evaluate_Devign(vul_idx, vul_type, os.getcwd(), json_cpgs, limit)


def evaluate_with_Reveal(vul_idx: int, vul_type: str, abspath: str, datas: List[Dict], limit: int):
    pretrain_model = Word2Vec.load(reveal_model_args.pretrain_word2vec_model.format(abspath, vul_type))
    model: ClassifyModel = ClassifyModel()
    model.to(reveal_model_args.device)
    model.eval()
    checkpoint = torch.load(
        os.path.join(reveal_model_args.model_dir.format(abspath, vul_type),
                     f'{reveal_model_args.model_name}_{reveal_model_args.detector}_best.pth'))
    model.load_state_dict(checkpoint['net'])
    reveal_util: RevealUtil = RevealUtil(pretrain_model, model)
    vulExpTester: VulExplainerTester = VulExplainerTester(vul_idx, vul_type, abspath, limit, 1)

    for data in datas:
        graph_info: Tuple[int, List[Data], torch.LongTensor] = reveal_util.generate_initial_training_datas(data)
        graph_data: Data = reveal_util.generate_initial_graph_embedding(graph_info)
        probs = torch.softmax(model(data = Batch.from_data_list([graph_data])), dim=1)
        vul_prob = probs.cpu()[0][1]
        print(f"vulnerability of function {data['functionName']} being vulnerable: {vul_prob}")
        if vul_prob > 0.5:
            cpg: CPG = vulExpTester.fromSerJson(data)
            path = vulExpTester.explain_single(model, cpg, graph_data)
            print(f"vul line path in function {data['functionName']} is:")
            print("line:", "--->".join([str(data["lines"][node]) for node in path]))



def evaluate_IVDetect(vul_idx: int, vul_type: str, abspath: str, datas: List[Dict], limit: int):
    pretrain_model = Word2Vec.load(ivdetect_model_args.pretrain_word2vec_model.format(abspath, vul_type))
    checkpoint = torch.load(
        os.path.join(ivdetect_model_args.model_dir.format(abspath, vul_type),
                     f'{ivdetect_model_args.model_name}_{ivdetect_model_args.detector}_best.pth'))
    model: IVDetectModel = IVDetectModel()
    model.load_state_dict(checkpoint['net'])
    model.to(ivdetect_model_args.device)
    model.eval()
    ivdetect_util: IVDetectUtil = IVDetectUtil(pretrain_model)
    vulExpTester: VulExplainerTester = VulExplainerTester(vul_idx, vul_type, abspath, limit, 2)

    for data in datas:
        feature: Tuple[List[torch.Tensor], List[Tuple], List[torch.Tensor], List[torch.Tensor],
                       List[torch.Tensor], torch.LongTensor, int] = ivdetect_util.generate_all_features(data)
        graph_data: Data = model.vectorize_graph(feature)
        probs = torch.softmax(model(data=Batch.from_data_list([graph_data])), dim=1)
        vul_prob = probs.cpu()[0][1]
        print(f"vulnerability of function {data['functionName']} being vulnerable: {vul_prob}")

        if vul_prob > 0.5:
            cpg: CPG = vulExpTester.fromSerJson(data)
            path = vulExpTester.explain_single(model, cpg, graph_data)
            print(f"vul line path in function {data['functionName']} is:")
            print("line:", "--->".join([str(data["lines"][node]) for node in path]))



def evaluate_Devign(vul_idx: int, vul_type: str, abspath: str, datas: List[Dict], limit: int):
    checkpoint = torch.load(
        os.path.join(devign_model_args.model_dir.format(abspath, vul_type),
                     f'{devign_model_args.model_name}_{devign_model_args.detector}_best.pth'))
    pretrain_model = Word2Vec.load(devign_model_args.pretrain_word2vec_model.format(abspath, vul_type))
    model: DevignModel = DevignModel()
    model.to(devign_model_args.device)
    model.load_state_dict(checkpoint['net'])
    devign_util = DevignUtil(pretrain_model, model)
    vulExpTester: VulExplainerTester = VulExplainerTester(vul_idx, vul_type, abspath, limit, 3)

    for data in datas:
        graph_info: Tuple[int, List[Data], torch.LongTensor] = devign_util.generate_initial_training_datas(data)
        graph_data: Data = devign_util.generate_initial_graph_embedding(graph_info)
        probs = torch.softmax(model(data=Batch.from_data_list([graph_data])), dim=1)
        vul_prob = probs.cpu()[0][1]
        print(f"vulnerability of function {data['functionName']} being vulnerable: {vul_prob}")

        if vul_prob > 0.5:
            cpg: CPG = vulExpTester.fromSerJson(data)
            path = vulExpTester.explain_single(model, cpg, graph_data)
            print(f"vul line path in function {data['functionName']} is:")
            print("line:", "--->".join([str(data["lines"][node]) for node in path]))