from models.Resnet50 import Resnet50_model
from models.MobileNetv2 import MobileNetv2_model
from models.DenseNet import DenseNet121_model
from models.VIT import VIT_model
from models.ShuffleNetV2 import ShuffleNetV2_model
from models.HarDNet import HarDNet_model
from models.EfficientNet import Efficientnet_b3_model
from models.GENet_Res50 import GENet_Res50_model

MODELS = [
    "Resnet50_model",
    "MobileNetv2_model",
    "ShuffleNetV2_model",
    "HarDNet_model",
    "DenseNet121_model",
    "Efficientnet_b3_model",
    "GENet_Res50_model",
    "VIT_model",
]


def choose_models(model_name):
    if model_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(model_name))
    return globals()[model_name]
