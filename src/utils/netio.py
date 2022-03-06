# @author: shaoeric
# @create: 2022.2.24
import torch
import torch.nn as nn
import os

class NetIO:
    def __init__(self, config) -> None:
        self.config = config
        
        self.__save_dir = os.path.join(config.output["ckpt_root"], config.output["save_dir"])
        self.__save_freq = config.output["save_freq"]
        self.__strict = config.model["strict"]
        self.__key_metric = config.train["metric"]["key_metric_name"]
        self.__strategy = config.train["metric"]["strategy"]
        
        if self.__strategy == "max":
            self.__best_score = -float("inf")
        elif self.__strategy == "min":
            self.__best_score = float("inf")
        elif self.__strategy == "none":
            self.__best_score = None
        else:
            raise NotImplementedError("Strategy not defined")

        os.makedirs(self.__save_dir, exist_ok=True)

    def load_file(self, net: nn.Module, weight_path: str):
        serial = torch.load(weight_path, map_location="cpu")
        state_dict = serial["state_dict"]
        net.load_state_dict(state_dict, strict=self.__strict)
        return net

    def save_file(self, net: nn.Module, epoch: int, metric: float, *args, **kwargs):
        if isinstance(net, nn.DataParallel):
            state_dict = net.module.state_dict()
        else:
            state_dict = net.state_dict()
        dic = {
            "state_dict": state_dict,
            "epoch": epoch,
            self.__key_metric: metric
        }
        # save checkpoint
        if self.__save_freq <= 0:
            torch.save(dic, os.path.join(self.__save_dir, "last.pth"))
        elif epoch % self.__save_freq == 0:
            torch.save(dic, os.path.join(self.__save_dir, "{}.pth".format(epoch)))

        # strategy
        if (self.__strategy == "max" and metric > self.__best_score) or \
            (self.__strategy == "min" and metric < self.__best_score):
            torch.save(dic, os.path.join(self.__save_dir, "best.pth"))
            self.__best_score = metric
    
    def get_best_score(self):
        return self.__best_score