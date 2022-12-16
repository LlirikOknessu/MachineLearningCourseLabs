import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Input
from keras.models import Model
from pathlib import Path
import pandas as pd

class SubClassing(Model):
    def __init__(self, hidden_laeyer=3, neurons_cnt=128):
        super(SubClassing, self).__init__()
        self.hidden_layers = hidden_laeyer
        