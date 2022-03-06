# @author shaoeric
# @create: 2022.2.23

from .loss_builder import LossBuilder
from .loss_wrapper import LossWrapper

__all__ = ['LossBuilder', 'LossWrapper']


# In src/losses, define the customized loss function with .py script
# In src/losses/loss_wrapper.py, initialize all loss functions with `LossLoader` in loss_loader.py and their corresponding loss weights, and implement how to compute the total loss in `forward`
