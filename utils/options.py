import argparse

class BaseOptions():
    """
    Argument Parser object    
    """
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--dataroot', type=str, default="./data/",
                                 help='path to folder (should contains past_data.csv, prediction_data.csv, bu_feat.csv, data_knn.json)')
        self.parser.add_argument('--device', type=int, default=0,
                                 help='choose between cpu=0(default) and gpu=1')
        self.parser.add_argument('--bu', type=int, default=95,
                                 help='business unit number for which you will plot the prediction')
        self.parser.add_argument('--dep', type=int, default=73,
                                 help='department number for which you will plot the prediction')
        self.parser.add_argument('--plot', type=int, default=0,
                                 help='plot the prediction for a given bu and dep (yes = 1 | no = 0)')
        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        return self.opt