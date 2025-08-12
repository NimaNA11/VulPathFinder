import argparse
import os
import sys

from global_defines import vul_types

from eval_utils.train_detectors import train_Reveal, train_Devign, train_IVDetect
from eval_utils.evaluate_detectors import evaluate_Reveal, evaluate_Devign, evaluate_IVDetect
from eval_utils.evaluate_vulexplainer import VulExplainerTester
from eval_utils.eval_single_data import evaluate_single_file_util

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_detector", action='store_true', default=False)
    parser.add_argument("--run_vulexplainer", action='store_true', default=False)
    parser.add_argument("--train_detector", action='store_true', default=False,
                        help="whether to train detectors, please do not use with run_detector")
    parser.add_argument("--detector_idx", type = int, default = 1,
                        help="detector idx, 1 --> Reveal, 2 --> IVDetect, 3 --> Devign")
    parser.add_argument("--vul_idx", type=int, default=0,
                        help="vulnerability idx, 0 --> buffer overflow, 1 --> incorrect calculation, 2 --> memory leak, 3 --> path traversal, 4 --> command injection")
    parser.add_argument("--limit", type=int, default=5,
                        help="path limit for vul explainer")
    parser.add_argument("--file", type=str, help="specify single file to analyze, can support function-level detector now")

    args = parser.parse_args()
    run_detector: bool = args.run_detector
    run_vulexplainer: bool = args.run_vulexplainer
    train_detector: bool = args.train_detector
    detector_idx: int = args.detector_idx
    vul_idx: int = args.vul_idx
    limit: int = args.limit
    file_to_analyze: str = args.file

    if train_detector and (run_detector or run_vulexplainer):
        print("please do not train and evaluate at the same time")
        sys.exit(-1)

    if train_detector:
        # train Reveal
        if detector_idx == 1:
            train_Reveal(vul_types[vul_idx], os.getcwd())

        # train IVDetect
        elif detector_idx == 2:
            train_IVDetect(vul_types[vul_idx], os.getcwd())

        # train Devign
        elif detector_idx == 3:
            train_Devign(vul_types[vul_idx], os.getcwd())

    ## Evaluate the performance of detectors: DeepWuKong, Reveal, IVDetect, Devign
    if run_detector:
        # evaluate Reveal
        if detector_idx == 1:
            evaluate_Reveal(vul_types[vul_idx], os.getcwd())

        # evaluate IVDetect
        elif detector_idx == 2:
            evaluate_IVDetect(vul_types[vul_idx], os.getcwd())

        # evaluate Devign
        elif detector_idx == 3:
            evaluate_Devign(vul_types[vul_idx], os.getcwd())

    ## Evaluate vul explainer
    if run_vulexplainer:
        tester: VulExplainerTester = VulExplainerTester(vul_idx, vul_types[vul_idx], os.getcwd(), limit, detector_idx)
        # evaluate VulExplainer with Reveal
        if detector_idx == 1:
            tester.process_reveal()
        # evaluate VulExplainer with IVDetect
        elif detector_idx == 2:
            tester.process_ivdetect()
        # evaluate VulExplainer with Devign
        elif detector_idx == 3:
            tester.process_devign()
            # tester.process_ensemble()

    # evaluate single file with function-level detector
    if file_to_analyze is not None:
        calleeInfoFile = "resources/calleeInfos.json"  
        evaluate_single_file_util(file_to_analyze, calleeInfoFile, vul_types[vul_idx], detector_idx, vul_idx, limit)



if __name__ == '__main__':
    main()