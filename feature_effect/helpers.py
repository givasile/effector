import typing


def prep_features(feat: typing.Union[str, list]) -> list:
    assert type(feat) in [list, str, int]
    if feat == "all":
        feat = [i for i in range(self.D)]
    elif type(feat) == int:
        feat = [feat]
    return feat
