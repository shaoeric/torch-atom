import logging
import os

from tensorboardX import SummaryWriter


class LoggerBuilder:
    def __init__(self, config) -> None:
        self.config = config
        self.log_dir = os.path.join(config.output["log_dir"], config.output["save_dir"])
        self.log_file = os.path.join(self.log_dir, "train.log")
    
    def load(self):
        logger = self.__load_logger()
        writer = self.__load_tensorboard()
        return logger, writer
        
    def __load_logger(self) -> logging.Logger:
        logger = logging.getLogger(self.config.name)
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(funcName)s - %(lineno)d - %(message)s',datefmt='%Y-%m-%d %H:%M:%S'
        )
        os.makedirs(self.log_dir, exist_ok=True)
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        return logger

    def __load_tensorboard(self):
        summary_writer = None
        if self.config.output["tensorboard"]:
            summary_writer = SummaryWriter(self.log_dir)
        return summary_writer