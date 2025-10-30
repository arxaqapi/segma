from .hubert.surgical_hydra import SurgicalHydraHubert
from .whisper.hydra import HydraWhisper
from .whisper.surgical import SurgicalWhisper
from .whisper.surgical_hydra import SurgicalHydra
from .whisper.whisperidou import Whisperidou
from .whisper.whisperimax import WhisperiMax

Models = {
    "whisperidou": Whisperidou,
    "whisperimax": WhisperiMax,
    "surgical_whisper": SurgicalWhisper,
    "hydra_whisper": HydraWhisper,
    "surgical_hydra": SurgicalHydra,
    "surgical_hubert_hydra": SurgicalHydraHubert,
}


__all__ = [
    "Whisperidou",
    "WhisperiMax",
    "SurgicalWhisper",
    "HydraWhisper",
    "SurgicalHydra",
    "SurgicalHydraHubert",
    "Models",
]
