# @Author:LiuZhQ

from .base_setting import BaseSetting


class SearchSetting(BaseSetting):
    """
        This class include the setting of arguments and the tools of experimental recorder.
    """

    def __init__(self, config_path):
        super(SearchSetting, self).__init__(config_path)
        # Dataset
        self.train_portion = self.con.dataset['train_portion']  # Ratio used for training
        # Training strategy
        self.unrolled = self.con.strategy['unrolled']  # Use one-step unrolled validation loss(2nd order)
        # Optimizer
        self.alpha_lr = self.con.optimizer.alpha['learning_rate']  # Learning rate for alpha encoding
        self.alpha_weight_decay = self.con.optimizer.alpha['weight_decay']  # Weight decay for alpha encoding
