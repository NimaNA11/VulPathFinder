from gensim.models import Word2Vec

from detectors.Reveal.configurations import model_args as reveal_model_args
from detectors.Reveal.model import ClassifyModel
from detectors.Reveal.train import TrainUtil as RevealTrainUtil

from detectors.Devign.model import DevignModel
from detectors.Devign.train import TrainUtil as DevignTrainUtil
from detectors.Devign.configurations import model_args as devign_model_args

from detectors.IVDetect.model import IVDetectModel
from detectors.IVDetect.configurations import model_args as ivdetect_model_args
from detectors.IVDetect.train import TrainUtil as IVDetectTrainUtil


def train_Reveal(vul_type: str, abspath: str):
    pretrain_model = Word2Vec.load(reveal_model_args.pretrain_word2vec_model.format(abspath, vul_type))
    reveal_model: ClassifyModel = ClassifyModel().to(reveal_model_args.device)
    train_util: RevealTrainUtil = RevealTrainUtil(pretrain_model, reveal_model, vul_type, abspath)
    train_util.train()


def train_IVDetect(vul_type: str, abspath: str):
    pretrain_model = Word2Vec.load(ivdetect_model_args.pretrain_word2vec_model.format(abspath, vul_type))
    ivdetect_model: IVDetectModel = IVDetectModel()
    ivdetect_model.to(ivdetect_model_args.device)
    train_util = IVDetectTrainUtil(pretrain_model, ivdetect_model, vul_type, abspath)
    train_util.train()


def train_Devign(vul_type: str, abspath: str):
    pretrain_model = Word2Vec.load(devign_model_args.pretrain_word2vec_model.format(abspath, vul_type))
    devign_model: DevignModel = DevignModel()
    devign_model.to(devign_model_args.device)
    train_util = DevignTrainUtil(pretrain_model, devign_model, vul_type, abspath)
    train_util.train()