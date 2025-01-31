from segma.models.hydra import HydraWhisper
from segma.models.pyannet import PyanNet, PyanNetSlim
from segma.models.surgical import SurgicalWhisper
from segma.models.whiser_based import Whisperidou, WhisperiMax

Models = {
    "whisperidou": Whisperidou,
    "whisperimax": WhisperiMax,
    "pyannet": PyanNet,
    "pyannet_slim": PyanNetSlim,
    "surgical_whisper": SurgicalWhisper,
    "hydra_whisper": HydraWhisper,
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
