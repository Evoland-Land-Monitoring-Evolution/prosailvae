import logging

import torch
from mmdc_singledate.datamodules.mmdc_datamodule import destructure_batch

MMDC_DATA_COMPONENTS = 8

# Configure logging
NUMERIC_LEVEL = getattr(logging, "INFO", None)
logging.basicConfig(
    level=NUMERIC_LEVEL, format="%(asctime)-15s %(levelname)s: %(message)s"
)

logger = logging.getLogger(__name__)


def mmdc2pvae_batch(
    batch: list[list[torch.Tensor, ...]]
) -> tuple[torch.Tensor, torch.Tensor]:
    # TODO: use the bands selected for the model …
    mbatch = destructure_batch(batch)
    if (nb_nans := torch.isnan(mbatch.s2_x.view(-1)).sum()) > 0:
        logger.info(f"{nb_nans=}")
        print(f"{nb_nans=}")
    if (nb_nans_a := torch.isnan(mbatch.s2_a.view(-1)).sum()) > 0:
        logger.info(f"{nb_nans_a=}")
        print(f"{nb_nans_a=}")
    s2_x = (
        mbatch.s2_x.nan_to_num() / 10000
    )  # MMDC S2 reflectances are *1000, while PVAE expects [0-1]
    s2_a = mbatch.s2_a.nan_to_num()
    return s2_x, torch.cat(
        (
            torch.rad2deg(torch.acos(s2_a[:, 0:1, ...])),  # sun_zen
            torch.rad2deg(torch.acos(s2_a[:, 3:4, ...])),  # view_zen
            torch.rad2deg(
                torch.acos(s2_a[:, 1:2, ...]) - torch.acos(s2_a[:, 4:5, ...])
            ),  # sun_az-view_az
        ),
        dim=1,
    )
