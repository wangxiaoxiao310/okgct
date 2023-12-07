"""
迁移程序关于MindSpore的辅助映射类
"""


class X2Mindspore:
    def __init__(self, source_framework: str, para_dict: dict):
        self.source_framework = source_framework
        self.para_dict = para_dict
        self.mapping_function_dict = {
            "mindspore.nn.SequentialCell": "mindspore_nn_SequentialCell",
            "mindspore.nn.Conv2d": "mindspore_nn_Conv2d",
            "mindspore.nn.MaxPool2d": "mindspore_nn_MaxPool2d",
            "mindspore.dataset.transforms.Compose": "mindspore_dataset_transforms_Compose",
            "mindspore.dataset.Cifar10Dataset": "mindspore_dataset_Cifar10Dataset",
            "callback.mindspore.dataset.Cifar10Dataset": "callback_mindspore_dataset_Cifar10Dataset",
            "callback": "callback",
        }  # 利用反射找到对应的辅助映射函数

    """
    True代表需要迁移、False代表跳过迁移
    """

    def mindspore_None(self):
        """
        :return:
        """
        special_para = {}
        return True, special_para

    def mindspore_nn_SequentialCell(self):
        """
        如果目标算子名称是mindspore.nn.SequentialCell，则将变量的处理推迟到 以后
        :return:
        """
        res = None
        return False, res

    def mindspore_nn_Conv2d(self):
        special_para = {}
        if self.source_framework == "PyTorch":
            padding = self.para_dict["padding"]["value"]
            padding_mode = self.para_dict["padding_mode"]["value"]
            bias = self.para_dict["bias"]["value"]

            if isinstance(padding, int) and padding == 0:
                special_para["padding"] = 0
                special_para["pad_mode"] = "0valid"  # 字面量
            elif isinstance(padding, int) and padding > 0:
                special_para["pad_mode"] = "0pad"  # 字面量
        return True, special_para

    def mindspore_nn_MaxPool2d(self):
        special_para = {}
        if self.source_framework == "PyTorch":
            padding = self.para_dict["padding"]["value"]

            if isinstance(padding, int) and padding == 0:
                special_para["padding"] = 0
                special_para["pad_mode"] = "0valid"  # 字面量
            elif isinstance(padding, int) and padding > 0:
                special_para["pad_mode"] = "0same"  # 字面量
            elif padding == "Null":
                special_para["padding"] = 0
                special_para["pad_mode"] = "0valid"  # 字面量
        return True, special_para

    def mindspore_dataset_transforms_Compose(self):
        """
        如果目标算子名称是mindspore.dataset.transforms.Compose，则将变量的处理推迟到 以后
        :return:
        """
        special_para = None
        return False, special_para

    def callback(self):
        special_para = {}
        return True, special_para

    def mindspore_dataset_Cifar10Dataset(self):
        special_para = {}
        if self.para_dict["train"]["value"]:
            special_para["usage"] = "0train"  # 字面量
        else:
            special_para["usage"] = "0test"
        special_para["expect_operator"] = "torch.utils.data.DataLoader"

        return True, special_para

    def callback_mindspore_dataset_Cifar10Dataset(self):
        """
        负责mindspore.dataset.Cifar10Dataset的回调
        :return:
        """
        special_para = {}
        if self.para_dict["shuffle"]["value"]:
            special_para["buffer_size"] = 4
        return True, special_para
