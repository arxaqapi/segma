from segma.models.pyannet import PyanNet, PyanNetSlim

from .whisper.hydra import HydraWhisper
from .whisper.surgical import SurgicalWhisper
from .whisper.surgical_hydra import SurgicalHydra
from .whisper.whisperidou import Whisperidou
from .whisper.whisperimax import WhisperiMax

from .wavlm.hydra import HydraWavLM
from .wavlm.surgical_hydra import SurgicalHydraWavLM

Models = {
    "whisperidou": Whisperidou,
    "whisperimax": WhisperiMax,
    "pyannet": PyanNet,
    "pyannet_slim": PyanNetSlim,
    "surgical_whisper": SurgicalWhisper,
    "hydra_whisper": HydraWhisper,
    "surgical_hydra": SurgicalHydra,
    "wavlm_hydra": HydraWavLM,
    "surgical_wavlm_hydra": SurgicalHydraWavLM
}


__all__ = [
    "Whisperidou",
    "WhisperiMax",
    "PyanNet",
    "PyanNetSlim",
    "SurgicalWhisper",
    "HydraWhisper",
    "SurgicalHydra",
    "HydraWavLM"
    "SurgicalHydraWavLM"
    "Models",
]
