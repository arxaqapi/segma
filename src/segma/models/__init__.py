from segma.models.models import Whisperidou, WhisperiMax
from segma.models.pyannet import PyanNet, PyanNetSlim

# from segma.models.utils import ConvolutionSettings


# improve with str.lowercase
Models = {
    "Whisperidou": Whisperidou,
    "whisperidou": Whisperidou,
    "WhisperiMax": WhisperiMax,
    "whisperimax": WhisperiMax,
    "pyannet": PyanNet,
    "PyanNet": PyanNet,
    "pyannet_slim": PyanNetSlim,
    "PyanNet_slim": PyanNetSlim,
}


__all__ = [
    "Whisperidou",
    "WhisperiMax",
    "PyanNet",
    "Models",
    # "ConvolutionSettings",
]
