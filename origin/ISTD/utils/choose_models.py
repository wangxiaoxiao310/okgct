# model
from model.model_UNet import UNet
from model.model_PSPNet import PSPNet

MODELS = [
    "UNet",
    "PSPNet",
]


# print(globals())
def choose_models(model_name):
    if model_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(model_name))
    return globals()[model_name]
