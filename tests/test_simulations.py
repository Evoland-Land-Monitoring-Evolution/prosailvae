from pathlib import Path

import torch

from prosailvae.ProsailSimus import ProsailSimulator, SensorSimulator


def get_ref_data(file_name: str, batch_size: int = 10) -> torch.Tensor:
    assert batch_size <= 100
    return torch.load(f"{Path(__file__).parent}/../data/{file_name}").float()[
        :batch_size
    ]


def prosail_params(batch_size: int = 10) -> torch.Tensor:
    return get_ref_data("prosail_sim_vars.pt", batch_size)


def prosail_full_simus_ref(batch_size: int = 10):
    return get_ref_data("prosail_simus_full.pt", batch_size)


def s2_simus_dummy_ref(batch_size: int = 10):
    return get_ref_data("s2_simus_dummy.pt", batch_size)


def test_prosail_simulation() -> None:
    """Simulate full spectra"""
    simulator = ProsailSimulator()
    params = prosail_params()
    simus = simulator(params)
    ref_data = prosail_full_simus_ref()
    assert torch.isclose(simus, ref_data).all()


def test_s2_simulation() -> None:
    """Simulate S2 spectra from dummy full spectra"""
    batch_size = 10
    prospect_range = (400, 2500)
    prospect_size = prospect_range[1] - prospect_range[0] + 1
    bands = list(range(1, 13))
    rsr_file = f"{Path(__file__).parent}/../data/sentinel2.rsr"
    simulator = SensorSimulator(rsr_file, prospect_range, bands)
    dummy_spectra = prosail_full_simus_ref(batch_size)[:, :prospect_size]
    s2_simu = simulator(dummy_spectra)
    ref_data = s2_simus_dummy_ref(batch_size)
    assert torch.isclose(s2_simu, ref_data).all()
