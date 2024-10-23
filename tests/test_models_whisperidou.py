from segma.models import Whisperidou
from segma.utils.encoders import PowersetMultiLabelEncoder


def test_Whisperidou_init():
    labels = ("aa", "bb", "cc")

    label_encoder = PowersetMultiLabelEncoder(labels)
    model = Whisperidou(label_encoder)

    assert model.w_encoder is not None
