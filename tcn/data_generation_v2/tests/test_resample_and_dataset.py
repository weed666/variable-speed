from __future__ import annotations

import numpy as np

from tcn.data_generation_v2.resample_100hz import zoh


def test_zoh_holds_discrete_contact_state() -> None:
    raw_t = np.asarray([0.0, 1.0 / 30.0, 2.0 / 30.0])
    contact = np.asarray([[1, 0], [0, 1], [1, 1]])
    target_t = np.asarray([0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06])
    held = zoh(raw_t, contact, target_t)
    np.testing.assert_allclose(held[:4], np.tile(contact[0], (4, 1)))
    np.testing.assert_allclose(held[4:7], np.tile(contact[1], (3, 1)))
