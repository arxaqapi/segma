from pathlib import Path

from segma.models import Minisinc  # , Whisperidou
from segma.predict import prediction
from segma.utils.encoders import PowersetMultiLabelEncoder

if __name__ == "__main__":
    l_encoder = PowersetMultiLabelEncoder(
        ["male", "female", "key_child", "other_child"]
    )

    prediction(Path("data/debug/wav/0000.wav"), model=Minisinc(l_encoder))
    # evaluate(Path("data/debug/wav/0000.wav"), model=Whisperidou(l_encoder))
