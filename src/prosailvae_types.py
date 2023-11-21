from dataclasses import asdict, dataclass

import torch

# Aliases
B2 = "B2"
B3 = "B3"
B4 = "B4"
B5 = "B5"
B6 = "B6"
B7 = "B7"
B8 = "B8"
B8A = "B8A"
B9 = "B9"
B10 = "B10"
B11 = "B11"
B12 = "B12"

# Band groups
GROUP_10M = [B2, B3, B4, B8]
GROUP_20M = [B5, B6, B7, B8A, B11, B12]
GROUP_60M = [B9, B10]
GROUP_BAND_PVAE = [B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12]
GROUP_BAND_BVNET = [B3, B4, B5, B6, B7, B8A, B11, B12]

# Aliases
sunzen = "sunzen"
viewzen = "viewzen"
relazi = "relazi"

# Angle groups
GROUP_PVAE_ANGLES = [sunzen, viewzen, relazi]
GROUP_BVNET_ANGLES = [viewzen, sunzen, relazi]

# Aliases
N = "N"
cab = "cab"
car = "car"
cbrown = "cbrown"
cw = "cbrown"
cm = "cm"
lai = "lai"
ant = "ant"
prot = "prot"
cbc = "cbc"
lidfa = "lidfa"
hspot = "hspot"
psoil = "psoil"
rsoil = "rsoil"


# Angle groups
GROUP_P5_BV = [N, cab, car, cbrown, cw, cm, lai, lidfa, hspot, psoil, rsoil]
GROUP_PD_BV = [N, cab, car, cbrown, cw, cm, ant, lai, lidfa, hspot, psoil, rsoil]
GROUP_PPRO_BV = [
    N,
    cab,
    car,
    cbrown,
    cw,
    cm,
    ant,
    prot,
    cbc,
    lai,
    lidfa,
    hspot,
    psoil,
    rsoil,
]


@dataclass
class S2IO:
    pass

    def asdict(self):
        return asdict(self)


@dataclass
class ViewAngles(S2IO):
    sunzen: torch.Tensor | None = None
    obszen: torch.Tensor | None = None
    relazi: torch.Tensor | None = None


@dataclass
class S2Bands(S2IO):
    B2: torch.Tensor | None = None
    B3: torch.Tensor | None = None
    B4: torch.Tensor | None = None
    B5: torch.Tensor | None = None
    B6: torch.Tensor | None = None
    B7: torch.Tensor | None = None
    B8: torch.Tensor | None = None
    B8A: torch.Tensor | None = None
    B11: torch.Tensor | None = None
    B12: torch.Tensor | None = None


class ProsailBV:
    N: torch.Tensor | None = None
    cab: torch.Tensor | None = None
    car: torch.Tensor | None = None
    cbrown: torch.Tensor | None = None
    cw: torch.Tensor | None = None
    cm: torch.Tensor | None = None
    lai: torch.Tensor | None = None
    lidfa: torch.Tensor | None = None
    hspot: torch.Tensor | None = None
    psoil: torch.Tensor | None = None
    rsoil: torch.Tensor | None = None
    ant: torch.Tensor | None = None
    prot: torch.Tensor | None = None
    cbc: torch.Tensor | None = None


class S2Data:
    def __init__(
        self,
        data: S2IO | torch.Tensor,
        features_name: str | None = None,
        features_dim: int = 0,
    ) -> None:
        self.features_dim = features_dim
        self.data = None
        self.features_name = features_name
        if isinstance(data, S2IO):
            self.init_from_data(data)
        elif isinstance(data, torch.Tensor):
            self.init_from_tensor(data, features_name)
        pass

    def init_from_data(self, data: S2IO) -> None:
        self.data = data
        self.features_name = [
            key for (key, item) in data.asdict().items() if item is not None
        ]

    def init_from_tensor(self, data: torch.Tensor, features_name: str) -> None:
        raise NotImplementedError

    def to_tensor(self, features_name: str | None = None):
        if features_name is None:
            return torch.cat(
                [
                    self.data.asdict()[f]
                    for f in self.features_name
                    if self.data.asdict()[f] is not None
                ],
                dim=self.features_dim,
            )
        none_features = []
        for f in features_name:
            if self.data.asdict()[f] is None:
                none_features.append(f)
        if len(none_features):
            raise ValueError(f"{', '.join(none_features)} not initialized!")
        return torch.cat(
            [self.data.asdict()[f] for f in features_name], dim=self.features_dim
        )

    def get_data(self) -> S2IO:
        return self.data

    def check_init_features(self, features_name) -> bool:
        for f in features_name:
            if f not in self.features_name:
                return False
        return True

    def normalize(self, loc, scale):
        for norm_data in [loc, scale]:
            assert isinstance(norm_data, type(self))
            assert norm_data.features_dim == 0
            assert self.check_init_features(norm_data.features_name)
        return torch.cat(
            [
                (self.to_tensor([f]) - loc.to_tensor([f])) / scale.to_tensor([f])
                for f in self.features_name
            ],
            dim=self.features_dim,
        )

    def denormalize(self, loc, scale):
        for norm_data in [loc, scale]:
            assert isinstance(norm_data, type(self))
            assert norm_data.features_dim == 0
            assert self.check_init_features(norm_data.features_name)
        return torch.cat(
            [
                self.to_tensor([f]) * scale.to_tensor([f]) + loc.to_tensor([f])
                for f in self.features_name
            ],
            dim=self.features_dim,
        )


class S2R(S2Data):
    def __init__(
        self,
        data: S2Bands | torch.Tensor,
        features_name: str | None = None,
        features_dim: int = 0,
    ) -> None:
        super().__init__(
            data=data, features_name=features_name, features_dim=features_dim
        )

    def init_from_tensor(self, data: torch.Tensor, features_name: str):
        data_dict = {}
        for i, f in enumerate(features_name):
            data_dict[f] = data.select(dim=self.features_dim, index=i).unsqueeze(
                dim=self.features_dim
            )
        self.data = S2Bands(**data_dict)


class S2A(S2Data):
    def __init__(
        self,
        data: S2Bands | torch.Tensor,
        features_name: str | None = None,
        features_dim: int = 0,
    ) -> None:
        super().__init__(
            data=data, features_name=features_name, features_dim=features_dim
        )

    def init_from_tensor(self, data: torch.Tensor, features_name: str):
        data_dict = {}
        for i, f in enumerate(features_name):
            data_dict[f] = data.select(dim=self.features_dim, index=i).unsqueeze(
                dim=self.features_dim
            )
        self.data = ViewAngles(**data_dict)


class BV(S2Data):
    def __init__(
        self,
        data: S2Bands | torch.Tensor,
        features_name: str | None = None,
        features_dim: int = 0,
    ) -> None:
        super().__init__(
            data=data, features_name=features_name, features_dim=features_dim
        )

    def init_from_tensor(self, data: torch.Tensor, features_name: str):
        data_dict = {}
        for i, f in enumerate(features_name):
            data_dict[f] = data.select(dim=self.features_dim, index=i).unsqueeze(
                dim=self.features_dim
            )
        self.data = ProsailBV(**data_dict)


if __name__ == "__main__":
    t = torch.tensor([[0, 1], [2, 3]])
    s = S2R(data=t, features_name=[B2, B3])
    loc = S2R(data=torch.tensor([0, 1]), features_name=[B2, B3])
    scale = S2R(data=torch.tensor([1, 2]), features_name=[B2, B3])
    tn = S2R(data=s.normalize(loc, scale), features_name=[B2, B3])
    tdn = S2R(data=tn.denormalize(loc, scale), features_name=[B2, B3])
    pass
