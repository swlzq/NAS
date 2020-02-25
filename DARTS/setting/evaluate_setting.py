# @Author:LiuZhQ

from .base_setting import BaseSetting


class EvaluateSetting(BaseSetting):
    """
        This class include the setting of arguments and the tools of experimental recorder.
    """

    def __init__(self, config_path):
        super(EvaluateSetting, self).__init__(config_path)
        # Training strategy
        self.arch = self.con.strategy.arch  # Searched architecture name
        self.auxiliary = self.con.strategy.auxiliary  # Use auxiliary tower
        self.auxiliary_weight = self.con.strategy.auxiliary_weight  # Weight for auxiliary loss
        self.drop_path_prob = self.con.strategy.drop_path_prob  # Drop path probability
