from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np
from astropy import units as u
from astropy.time import Time
from rich.console import Group
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn

if TYPE_CHECKING:
    from typing import Literal

    Det = Literal['LE', 'ME', 'HE']
    NDArray = np.ndarray


def get_hxmt_ebounds(det: Det):
    """Get the energy bounds of telescope."""
    assert det in ('LE', 'ME', 'HE')
    path = os.path.dirname(__file__)
    ebounds = np.loadtxt(f'{path}/data/HXMT_EBOUNDS/{det}.txt')
    ebounds = {
        'PI': ebounds[:, 0].astype(int),
        'E_MIN': ebounds[:, 1],
        'E_MAX': ebounds[:, 2],
    }
    return ebounds


def get_trigger_info():
    """Get the trigger information."""
    path = os.path.dirname(__file__)
    path = os.path.abspath(f'{path}/../')
    trigger = np.loadtxt(f'{path}/trigger.txt', dtype=str)
    trigger = {(obsid, n): float(t0) for obsid, n, t0 in trigger}

    if os.path.exists(f'{path}/ignore.txt'):
        ignore = np.loadtxt(f'{path}/ignore.txt', dtype=str)
        if ignore.size:
            obsid = ignore[:, :2].tolist()
            flag = ignore[:, 2:].astype(bool).tolist()
            ignore = {
                tuple(i): dict(zip(['LE', 'ME', 'HE'], j, strict=False))
                for i, j in zip(obsid, flag, strict=False)
            }
        else:
            ignore = {}
    else:
        ignore = {}

    return {'trigger': trigger, 'ignore': ignore}


def get_pi_range(det: Det, emin: float, emax: float):
    """Get the PI range of telescope."""
    ebounds = get_hxmt_ebounds(det)
    pi_min = ebounds['PI'][np.flatnonzero(ebounds['E_MIN'] >= emin)[0]]
    pi_max = ebounds['PI'][np.flatnonzero(ebounds['E_MAX'] <= emax)[-1]]
    return pi_min, pi_max


def met_to_utc(met: float) -> str:
    """Convert HXMT MET to UTC.

    Parameters
    ----------
    met : float
        The HXMT MET time.

    Returns
    -------
    str
        The UTC time in the format of 'YYYY-MM-DDTHH:MM:SS.SSS'.
    """
    utc0 = Time('2012-01-01T00:00:00', scale='utc', format='isot')
    return (utc0 + met * u.s).isot


def utc_to_met(utc: str) -> float:
    """Convert UTC to HXMT MET.

    Parameters
    ----------
    utc : str
        The UTC time in the format of 'YYYY-MM-DDTHH:MM:SS.SSS'.

    Returns
    -------
    float
        The HXMT MET time.
    """
    utc0 = Time('2012-01-01T00:00:00', scale='utc', format='isot')
    delta = Time(utc, scale='utc', format='isot') - utc0
    return delta.sec


class CustomProgress(Progress):
    def __init__(self, renderable_generator, *args, **kwargs):
        self._renderable_generator = renderable_generator
        self._renderable = None
        self.update_renderable()
        super().__init__(*args, **kwargs)

    @classmethod
    def get_default_columns(cls):
        return (
            SpinnerColumn(),
            *super().get_default_columns(),
            '->',
            TimeElapsedColumn(),
        )

    def update_renderable(self):
        self._renderable = self._renderable_generator()

    def get_renderable(self):
        renderable = Group(*self.get_renderables(), self._renderable)
        return renderable
