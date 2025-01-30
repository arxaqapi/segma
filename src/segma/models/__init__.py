from segma.models.hydra import HydraWhisper
from segma.models.pyannet import PyanNet, PyanNetSlim
from segma.models.surgical import SurgicalWhisper
from segma.models.whiser_based import Whisperidou, WhisperiMax

Models = {
    "Whisperidou": Whisperidou,
    "whisperidou": Whisperidou,
    "WhisperiMax": WhisperiMax,
    "whisperimax": WhisperiMax,
    "pyannet": PyanNet,
    "PyanNet": PyanNet,
    "pyannet_slim": PyanNetSlim,
    "PyanNet_slim": PyanNetSlim,
    "surgical_whisper": SurgicalWhisper,
    "surgicalwhisper": SurgicalWhisper,
    "hydra_whisper": HydraWhisper,
    "HydraWhisper": HydraWhisper,
}


__all__ = [
    "Whisperidou",
    "WhisperiMax",
    "PyanNet",
    "PyanNetSlim",
    "SurgicalWhisper",
    "HydraWhisper",
    "Models",
]
