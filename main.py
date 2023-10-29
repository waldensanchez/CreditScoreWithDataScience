import numpy as np
import pandas as pd
from cleanData import clean
from prediction import setScore

class modelWAM():

    def __init__(self, datos) -> None:
        self.df = datos

    def cleanData(self):
        self.df = clean(self.df)

    def prediction(self):
        self.df = setScore(self.df)

    def accuracy(self):
        evaluation = (self.df['Score_Predicted_in_bracket']-self.df['Score']).apply(lambda x: 0 if x != 0 else 1)
        return evaluation.sum()/len(evaluation)
