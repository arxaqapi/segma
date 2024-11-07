from segma.models.models import Whisperidou, WhisperiMax
from segma.models.pyannet import PyanNet

# from segma.models.utils import ConvolutionSettings


# improve with str.lowercase
Models = {
    "Whisperidou": Whisperidou,
    "whisperidou": Whisperidou,
    "WhisperiMax": WhisperiMax,
    "whisperimax": WhisperiMax,
    "pyannet": PyanNet,
    "PyanNet": PyanNet,
}


__all__ = [
    "Whisperidou",
    "WhisperiMax",
    "PyanNet",
    "Models",
    # "ConvolutionSettings",
]
