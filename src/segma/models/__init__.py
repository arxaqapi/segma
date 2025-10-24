from segma.models.pyannet import PyanNet, PyanNetSlim

from .hubert.surgical_hydra import SurgicalHydraHubert
from .whisper.hydra import HydraWhisper
from .whisper.surgical import SurgicalWhisper
from .whisper.surgical_hydra import SurgicalHydra
from .whisper.whisperidou import Whisperidou
from .whisper.whisperimax import WhisperiMax

Models = {
    "whisperidou": Whisperidou,
    "whisperimax": WhisperiMax,
    "pyannet": PyanNet,
    "pyannet_slim": PyanNetSlim,
    "surgical_whisper": SurgicalWhisper,
    "hydra_whisper": HydraWhisper,
    "surgical_hydra": SurgicalHydra,
    "surgical_hubert_hydra": SurgicalHydraHubert,
}


__all__ = [
    "Whisperidou",
    "WhisperiMax",
    "PyanNet",
    "PyanNetSlim",
    "SurgicalWhisper",
    "HydraWhisper",
    "SurgicalHydra",
    "SurgicalHydraHubert",
    "Models",
]
