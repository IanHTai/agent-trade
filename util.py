import numpy as np
from keras.models import model_from_config, Sequential, Model
from keras.optimizers import Optimizer
import keras.backend as K

def clone_model(model, custom_objects={}):
    # Requires Keras 1.0.7 since get_config has breaking changes.
    config = {
        'class_name': model.__class__.__name__,
        'config': model.get_config(),
    }
    clone = model_from_config(config, custom_objects=custom_objects)
    clone.set_weights(model.get_weights())
    return clone

def get_target_updates(target_model, source_model, tau):
    target_weights = target_model.trainable_weights + sum([l.non_trainable_weights for l in target_model.layers], [])
    source_weights = source_model.trainable_weights + sum([l.non_trainable_weights for l in source_model.layers], [])
    assert len(target_weights) == len(source_weights)

    # Create updates.
    updates = []
    for tw, sw in zip(target_weights, source_weights):
        updates.append((tw, tau * sw + (1. - tau) * tw))

    return updates


class TargetUpdateOptimizer(Optimizer):
    def __init__(self, updates):
        super(TargetUpdateOptimizer, self).__init__()
        self.targetUpdates = updates

    def get_updates(self, params, loss):
        self.updates = self.targetUpdates
        return self.updates