from __future__ import annotations

import glob
import json
import multiprocessing as mp
import os
import subprocess
import traceback
from typing import TYPE_CHECKING

import numpy as np
from astropy.io import fits
from rich.table import Table

from . import CustomProgress, get_pi_range

if TYPE_CHECKING:
    from typing import Literal

    NDArray = np.ndarray

# fix bug: Unable to redirect prompts to the /dev/tty (at headas_stdio.c:152)
os.environ['HEADASNOQUERY'] = 'False'
GTI_TYPE: Literal['standard', 'burst'] = 'burst'


def set_gti_type(gti_type: Literal['standard', 'burst']):
    assert gti_type in ('standard', 'burst')
    global GTI_TYPE
    GTI_TYPE = gti_type


def get_default_det_expr(det: Literal['LE', 'ME', 'HE']) -> str:
    if det == 'LE':
        return (
            '0,2-4,6-10,12,14,20,22-26,28,30,32,34-36,38-42,44,46,'
            '52,54-58,60-62,64,66-68,70-74,76,78,84,86,88-90,92-94'
        )
    elif det == 'ME':
        return '0-7,11-25,29-43,47-53'
    elif det == 'HE':
        return '0-15,17'
    else:
        raise ValueError(f'Unknown detector: {det}')


def get_default_gti_expr(det: Literal['LE', 'ME', 'HE']) -> str:
    if det == 'LE':
        if GTI_TYPE == 'standard':
            return (
                'ELV>10&&DYE_ELV>30&&COR>8'
                '&&SAA_FLAG==0&&T_SAA>=300&&TN_SAA>=300'
                '&&ANG_DIST<=0.04'
            )
        elif GTI_TYPE == 'burst':
            return 'ELV>1&&SAA_FLAG==0&&ANG_DIST<=0.5'
        else:
            raise ValueError(f'Unknown GTI_TYPE: {GTI_TYPE}')

    elif det == 'ME':
        if GTI_TYPE == 'standard':
            return (
                'ELV>10&&COR>8'
                '&&SAA_FLAG==0&&T_SAA>=300&&TN_SAA>=300'
                '&&ANG_DIST<=0.04'
            )
        elif GTI_TYPE == 'burst':
            return 'ELV>1&&SAA_FLAG==0&&ANG_DIST<=0.5'
        else:
            raise ValueError(f'Unknown GTI_TYPE: {GTI_TYPE}')

    elif det == 'HE':
        if GTI_TYPE == 'standard':
            return (
                'ELV>10&&COR>8'
                '&&SAA_FLAG==0&&T_SAA>=300&&TN_SAA>=300'
                '&&ANG_DIST<=0.04'
            )
        elif GTI_TYPE == 'burst':
            return 'ELV>0&&SAA_FLAG==0&&ANG_DIST<=0.5'
        else:
            raise ValueError(f'Unknown GTI_TYPE: {GTI_TYPE}')

    else:
        raise ValueError(f'Unknown detector: {det}')


def get_default_he_pm_expr() -> str:
    if GTI_TYPE == 'standard':
        return 'Cnt_PM_0<50&&Cnt_PM_1<50&&Cnt_PM_2<50'
    elif GTI_TYPE == 'burst':
        return ''
    else:
        raise ValueError(f'Unknown GTI_TYPE: {GTI_TYPE}')


def get_he_ascend_file() -> str:
    if GTI_TYPE == 'standard':
        return '$HEADAS/refdata/HE_D0_C26-255_Ascend_PHA_Map.txt'
    elif GTI_TYPE == 'burst':
        return 'NONE'
    else:
        raise ValueError(f'Unknown GTI_TYPE: {GTI_TYPE}')


def get_he_descend_file() -> str:
    if GTI_TYPE == 'standard':
        return '$HEADAS/refdata/HE_D0_C26-255_Descend_PHA_Map.txt'
    elif GTI_TYPE == 'burst':
        return 'NONE'
    else:
        raise ValueError(f'Unknown GTI_TYPE: {GTI_TYPE}')


def get_obsid(proposal_id: str) -> list[str]:
    proposal_id = str(proposal_id)
    assert len(proposal_id) == 8
    year = proposal_id[1:3]
    dirs = glob.glob(
        f'/hxmtfs2/work/HXMT-DATA/1L/A{year}/{proposal_id}/{proposal_id}???/'
        f'{proposal_id}?????-????????-??-??'
    )
    obsid = sorted({d.split('/')[-1].split('-')[0] for d in dirs})
    return obsid


def get_files(obsid: str, det: Literal['LE', 'ME', 'HE']) -> dict[str, str]:
    obsid = str(obsid)
    det = str(det).upper()
    p1 = f'/hxmtfs2/work/HXMT-DATA/1L/A{obsid[1:3]}/{obsid[:-5]}/{obsid[:-2]}'
    p2 = f'{p1}/{obsid}-????????-??-??'
    files = {
        'Att': get_file_by_pattern(
            f'{p2}/ACS/HXMT_{obsid}_Att_FFFFFF_V?_L1P.FITS',
            f'{p1}/ACS/HXMT_{obsid[:-2]}_Att_FFFFFF_V?_L1P.FITS',
        ),
        'EHK': get_file_by_pattern(
            f'{p2}/AUX/HXMT_{obsid}_EHK_FFFFFF_V?_L1P.FITS',
            f'{p1}/AUX/HXMT_{obsid[:-2]}_EHK_FFFFFF_V?_L1P.FITS',
        ),
    }
    if det == 'LE':
        patterns = {
            'LE_Evt': f'{p2}/LE/HXMT_{obsid}_LE-Evt_FFFFFF_V?_L1P.FITS',
            'LE_TH': f'{p2}/LE/HXMT_{obsid}_LE-TH_FFFFFF_V?_L1P.FITS',
            'LE_InsStat': (
                f'{p2}/LE/HXMT_{obsid}_LE-InsStat_FFFFFF_V?_L1P.FITS'
            ),
        }
    elif det == 'ME':
        patterns = {
            'ME_Evt': f'{p2}/ME/HXMT_{obsid}_ME-Evt_FFFFFF_V?_L1P.FITS',
            'ME_TH': f'{p2}/ME/HXMT_{obsid}_ME-TH_FFFFFF_V?_L1P.FITS',
        }
    elif det == 'HE':
        patterns = {
            'HE_Evt': f'{p2}/HE/HXMT_{obsid}_HE-Evt_FFFFFF_V?_L1P.FITS',
            'HE_DTime': f'{p2}/HE/HXMT_{obsid}_HE-DTime_FFFFFF_V?_L1P.FITS',
            'HE_HV': f'{p2}/HE/HXMT_{obsid}_HE-HV_FFFFFF_V?_L1P.FITS',
            'HE_TH': f'{p2}/HE/HXMT_{obsid}_HE-TH_FFFFFF_V?_L1P.FITS',
            'HE_PM': f'{p2}/HE/HXMT_{obsid}_HE-PM_FFFFFF_V?_L1P.FITS',
        }
    else:
        raise ValueError(f'Unknown detector: {det}')
    files |= {k: get_file_by_pattern(v) for k, v in patterns.items()}
    return files


def get_file_by_pattern(
    pattern: str,
    alt_pattern: str | None = None,
    throw: bool = True,
) -> str:
    files = glob.glob(pattern)
    if len(files):
        return sorted(files)[-1]
    else:
        if alt_pattern is not None:
            files = glob.glob(alt_pattern)
            if len(files):
                return sorted(files)[-1]
            else:
                if throw:
                    raise ValueError(
                        f'No files matched pattern:\n{pattern}\n'
                        f'or {alt_pattern}'
                    )
                else:
                    return ''
        else:
            if throw:
                raise ValueError(f'No files matched pattern:\n{pattern}')
            else:
                return ''


def run_cmd(cmd: str) -> dict:
    """Run command in shell."""
    result = subprocess.run(
        cmd,
        capture_output=True,
        shell=True,
    )
    return {
        'success': result.returncode == 0,
        'cmd': result.args,
        'code': result.returncode,
        'stdout': result.stdout.decode('utf-8'),
        'stderr': result.stderr.decode('utf-8'),
    }


def reduce_le_data(
    obsid: str,
    out_path: str,
    det_expr: str | None = None,
    gti_expr: str | None = None,
    overwrite: bool = True,
    overwrite_pi: bool = False,
    overwrite_recon: bool = False,
) -> tuple[dict[str, str], dict[str, str]]:
    """Insight-HXMT/LE Data reduction.

    Parameters
    ----------
    obsid : str
        ObsID (PYYAAAAABBBCC) of HXMT observation, e.g., P051435700205.
    """
    if det_expr is None:
        det_expr = get_default_det_expr('LE')
    if gti_expr is None:
        gti_expr = get_default_gti_expr('LE')

    out_path = os.path.abspath(out_path)
    clobber = {
        'clobber': 'yes' if overwrite else 'no',
        'clobber_pi': 'yes' if overwrite_pi else 'no',
        'clobber_recon': 'yes' if overwrite_recon else 'no',
    }

    files = get_files(obsid, 'LE')

    prefix = f'{out_path}/{obsid}/LE/{obsid}_LE'
    files |= {
        'LE_PI': f'{prefix}_PI.fits',
        'LE_PEDESTAL': f'{prefix}_PEDESTAL.fits',
        'LE_RECON': f'{prefix}_RECON.fits',
        'LE_GTI': f'{prefix}_GTI.fits',
        'LE_GTI_CORR': f'{prefix}_GTI_CORR.fits',
        'LE_SCREEN': f'{prefix}_SCREEN.fits',
    }

    os.makedirs(f'{out_path}/{obsid}', exist_ok=True)
    os.makedirs(f'{out_path}/{obsid}/LE', exist_ok=True)

    mapping = files | clobber
    cmd_pical = str.format_map(
        'lepical '
        'evtfile="{LE_Evt}" '
        'tempfile="{LE_TH}" '
        'outfile="{LE_PI}" '
        'pedestalfile="{LE_PEDESTAL}" '
        'clobber={clobber_pi}',
        mapping,
    )
    cmd_recon = str.format_map(
        'lerecon '
        'evtfile="{LE_PI}" '
        'outfile="{LE_RECON}" '
        'instatusfile="{LE_InsStat}" '
        'clobber={clobber_recon}',
        mapping,
    )
    cmd_gti = str.format_map(
        'legtigen '
        'evtfile=NONE '
        'instatusfile="{LE_InsStat}" '
        'tempfile="{LE_TH}" '
        'ehkfile="{EHK}" '
        'outfile="{LE_GTI}" '
        'defaultexpr=NONE '
        f'expr="{gti_expr}" '
        'clobber={clobber_recon}',
        mapping,
    )
    cmd_gti_corr = str.format_map(
        'legticorr "{LE_RECON}" "{LE_GTI}" "{LE_GTI_CORR}"',
        mapping,
    )

    if GTI_TYPE == 'standard':
        gti_file = files['LE_GTI_CORR']
    else:
        gti_file = files['LE_GTI']
    cmd_screen = str.format_map(
        'lescreen '
        'evtfile="{LE_RECON}" '
        f'gtifile="{gti_file}" '
        'outfile="{LE_SCREEN}" '
        f'userdetid="{det_expr}" '
        'eventtype=1 '
        'starttime=0 '
        'stoptime=0 '
        'minPI=0 '
        'maxPI=1535 '
        'clobber={clobber}',
        mapping,
    )

    status = {
        'lepical': run_cmd(cmd_pical),
        'lerecon': run_cmd(cmd_recon),
        'legtigen': run_cmd(cmd_gti),
        'legticorr': run_cmd(cmd_gti_corr),
        'lescreen': run_cmd(cmd_screen),
    }

    return files, status


def reduce_me_data(
    obsid: str,
    out_path: str,
    det_expr: str | None = None,
    gti_expr: str | None = None,
    overwrite: bool = True,
    overwrite_pi: bool = False,
    overwrite_grade: bool = False,
) -> tuple[dict[str, str], dict[str, str]]:
    if det_expr is None:
        det_expr = get_default_det_expr('ME')
    if gti_expr is None:
        gti_expr = get_default_gti_expr('ME')

    out_path = os.path.abspath(out_path)
    clobber = {
        'clobber': 'yes' if overwrite else 'no',
        'clobber_pi': 'yes' if overwrite_pi else 'no',
        'clobber_grade': 'yes' if overwrite_grade else 'no',
    }

    files = get_files(obsid, 'ME')

    prefix = f'{out_path}/{obsid}/ME/{obsid}_ME'
    files |= {
        'ME_PI': f'{prefix}_PI.fits',
        'ME_GRADE': f'{prefix}_GRADE.fits',
        'ME_DTIME': f'{prefix}_DTIME.fits',
        'ME_GTI': f'{prefix}_GTI.fits',
        'ME_GTI_CORR': f'{prefix}_GTI_CORR.fits',
        'ME_DETSTAT': f'{prefix}_DETSTAT.fits',
        'ME_SCREEN': f'{prefix}_SCREEN.fits',
    }

    os.makedirs(f'{out_path}/{obsid}', exist_ok=True)
    os.makedirs(f'{out_path}/{obsid}/ME', exist_ok=True)

    mapping = files | clobber
    cmd_pical = str.format_map(
        'mepical '
        'evtfile="{ME_Evt}" '
        'tempfile="{ME_TH}" '
        'outfile="{ME_PI}" '
        'clobber={clobber_pi}',
        mapping,
    )

    cmd_grade = str.format_map(
        'megrade '
        'evtfile="{ME_PI}" '
        'outfile="{ME_GRADE}" '
        'deadfile="{ME_DTIME}" '
        'binsize=0.001 '
        'clobber={clobber_grade}',
        mapping,
    )

    cmd_gtigen = str.format_map(
        'megtigen '
        'tempfile="{ME_TH}" '
        'ehkfile="{EHK}" '
        'outfile="{ME_GTI}" '
        'defaultexpr=NONE '
        f'expr="{gti_expr}" '
        'clobber={clobber}',
        mapping,
    )

    cmd_gticorr = str.format_map(
        'megticorr "{ME_GRADE}" "{ME_GTI}" "{ME_GTI_CORR}" '
        '$HEADAS/refdata/medetectorstatus.fits "{ME_DETSTAT}"',
        mapping,
    )

    if GTI_TYPE == 'standard':
        gti_file = files['ME_GTI_CORR']
    else:
        gti_file = files['ME_GTI']
    cmd_screen = str.format_map(
        'mescreen '
        'evtfile="{ME_GRADE}" '
        f'gtifile="{gti_file}" '
        'outfile="{ME_SCREEN}" '
        'baddetfile="{ME_DETSTAT}" '
        f'userdetid="{det_expr}" '
        'starttime=0 stoptime=0 '
        'minPI=0 maxPI=1023 '
        'clobber={clobber}',
        mapping,
    )

    status = {
        'mepical': run_cmd(cmd_pical),
        'megrade': run_cmd(cmd_grade),
        'megtigen': run_cmd(cmd_gtigen),
        'megticorr': run_cmd(cmd_gticorr),
        'mescreen': run_cmd(cmd_screen),
    }

    return files, status


def reduce_he_data(
    obsid: str,
    out_path: str,
    det_expr: str | None = None,
    gti_expr: str | None = None,
    pm_expr: str | None = None,
    overwrite: bool = True,
    overwrite_pi: bool = False,
) -> dict[str, int]:
    if det_expr is None:
        det_expr = get_default_det_expr('HE')
    if gti_expr is None:
        gti_expr = get_default_gti_expr('HE')
    if pm_expr is None:
        pm_expr = get_default_he_pm_expr()

    out_path = os.path.abspath(out_path)
    clobber = {
        'clobber': 'yes' if overwrite else 'no',
        'clobber_pi': 'yes' if overwrite_pi else 'no',
    }

    files = get_files(obsid, 'HE')

    prefix = f'{out_path}/{obsid}/HE/{obsid}_HE'
    files |= {
        'HE_PI': f'{prefix}_PI.fits',
        'HE_SPIKE': f'{prefix}_SPIKE.fits',
        'HE_GTI': f'{prefix}_GTI.fits',
        'HE_SCREEN': f'{prefix}_SCREEN.fits',
    }

    os.makedirs(f'{out_path}/{obsid}', exist_ok=True)
    os.makedirs(f'{out_path}/{obsid}/HE', exist_ok=True)

    mapping = files | clobber
    cmd_pical = str.format_map(
        'hepical '
        'evtfile="{HE_Evt}" '
        'outfile="{HE_PI}" '
        'glitchfile="{HE_SPIKE}" '
        'clobber={clobber_pi}',
        mapping,
    )

    cmd_gtigen = str.format_map(
        'hegtigen '
        'hvfile="{HE_HV}" '
        'tempfile="{HE_TH}" '
        'pmfile="{HE_PM}" '
        'outfile="{HE_GTI}" '
        'ehkfile="{EHK}" '
        'defaultexpr=NONE '
        f'expr="{gti_expr}" '
        f'pmexpr="{pm_expr}" '
        f'ascendfile="{get_he_ascend_file()}" '
        f'descendfile="{get_he_descend_file()}" '
        'clobber={clobber}',
        mapping,
    )

    res_screen = str.format_map(
        'hescreen '
        'evtfile="{HE_PI}" '
        'gtifile="{HE_GTI}" '
        'outfile="{HE_SCREEN}" '
        f'userdetid="{det_expr}" '
        'eventtype=1 '
        'anticoincidence=yes '
        'starttime=0 stoptime=0 '
        'minPI=0 maxPI=255 '
        'clobber={clobber}',
        mapping,
    )

    status = {
        'hepical': run_cmd(cmd_pical),
        'hegtigen': run_cmd(cmd_gtigen),
        'hescreen': run_cmd(res_screen),
    }

    return files, status


def batch_reduce(
    obsid_list: list[str],
    out_path: str,
    le_kwargs: dict | None = None,
    me_kwargs: dict | None = None,
    he_kwargs: dict | None = None,
    ncpu: int = mp.cpu_count() - 1,
):
    if not isinstance(obsid_list, list):
        raise TypeError('obsid_list must be a list')
    obsid_list = np.unique(obsid_list)
    obsid_list = [str(obsid).upper() for obsid in obsid_list]
    out_path = os.path.abspath(out_path)
    if le_kwargs is None:
        le_kwargs = {}
    else:
        le_kwargs = dict(le_kwargs)
    if me_kwargs is None:
        me_kwargs = {}
    else:
        me_kwargs = dict(me_kwargs)
    if he_kwargs is None:
        he_kwargs = {}
    else:
        he_kwargs = dict(he_kwargs)

    def generate_table():
        table = Table(title='HXMT Data Reduction', title_style='bold')
        table.add_column('ObsID', no_wrap=True, justify='center')
        table.add_column('LE', no_wrap=True, justify='center')
        table.add_column('ME', no_wrap=True, justify='center')
        table.add_column('HE', no_wrap=True, justify='center')
        for obsid in obsid_list:
            row = [obsid]
            for det in ('LE', 'ME', 'HE'):
                s = status[obsid].get(det, {})
                if s:
                    if all(si['success'] for si in s.values()):
                        mark = '✅'
                    else:
                        mark = '❌'
                else:
                    mark = '🕘'
                row.append(mark)
            table.add_row(*row)
        return table

    def callback_generator(obsid, det, progress, task):
        def handler(res):
            files[obsid].update(res[0])
            status[obsid][det] = res[1]
            progress.update(task, advance=1, refresh=True)
            progress.update_renderable()

        return handler

    def error_callback_generator(obsid, det, progress, task):
        def handler(error):
            status[obsid][det] = {
                'error': {
                    'success': False,
                    'msg': traceback.format_exception(error),
                }
            }
            progress.update(task, advance=1, refresh=True)
            progress.update_renderable()

        return handler

    os.makedirs(out_path, exist_ok=True)
    files = {obsid: {} for obsid in obsid_list}
    status = {obsid: {} for obsid in obsid_list}
    reduce_fn = {
        'LE': reduce_le_data,
        'ME': reduce_me_data,
        'HE': reduce_he_data,
    }
    kwds = {
        'LE': le_kwargs,
        'ME': me_kwargs,
        'HE': he_kwargs,
    }

    with CustomProgress(generate_table, refresh_per_second=2) as p:
        task = p.add_task('', total=len(obsid_list) * 3)
        with mp.Pool(ncpu) as pool:
            for obsid in obsid_list:
                for det, fn in reduce_fn.items():
                    pool.apply_async(
                        fn,
                        args=(obsid, out_path),
                        kwds=kwds[det],
                        callback=callback_generator(obsid, det, p, task),
                        error_callback=error_callback_generator(
                            obsid, det, p, task
                        ),
                    )
            pool.close()
            pool.join()

    with open(f'{out_path}/files.json', 'w') as f:
        json.dump(files, f, indent=4, ensure_ascii=False)

    with open(f'{out_path}/status.json', 'w') as f:
        json.dump(status, f, indent=4, ensure_ascii=False)


def generate_hxmt_gti(infile, intervals, outfile):
    intervals = np.reshape(intervals, newshape=(-1, 2))

    with fits.open(infile) as f:
        primary_hdu = fits.PrimaryHDU()
        primary_hdu.header = f['PRIMARY'].header

        gti0_hdu = fits.BinTableHDU.from_columns(
            columns=fits.ColDefs(
                [
                    fits.Column('START', '1D', 's', array=intervals[:, 0]),
                    fits.Column('STOP', '1D', 's', array=intervals[:, 1]),
                ]
            ),
            header=f['GTI0'].header,
            name='GTI0',
        )

        gtidesc_hdu = fits.BinTableHDU()
        gtidesc_hdu.name = 'GTIDesc'
        gtidesc_hdu.header = f['GTIDesc'].header
        gtidesc_hdu.header.remove(
            keyword='HISTORY', ignore_missing=True, remove_all=True
        )
        gtidesc_hdu.header.add_history(
            f'GTI specified by user: {np.squeeze(intervals).tolist()}'
        )
        gtidesc_hdu.data = f['GTIDesc'].data

    hdul = fits.HDUList([primary_hdu, gti0_hdu, gtidesc_hdu])
    hdul.writeto(outfile, overwrite=True)


def le_screen(
    recon_file: str,
    gti_normal: str,
    out_path: str,
    gti_file: str,
    out_file: str,
    intervals: list,
    det_expr: str | None = None,
):
    if det_expr is None:
        det_expr = get_default_det_expr('LE')

    out_path = os.path.abspath(out_path)
    os.makedirs(out_path, exist_ok=True)
    gti_file = f'{out_path}/{gti_file}'
    out_file = f'{out_path}/{out_file}'

    generate_hxmt_gti(gti_normal, intervals, gti_file)

    res = run_cmd(
        'lescreen '
        f'evtfile="{recon_file}" '
        f'gtifile="{gti_file}" '
        f'outfile="{out_file}" '
        f'userdetid="{det_expr}" '
        'eventtype=1 '
        'starttime=0 '
        'stoptime=0 '
        'minPI=0 '
        'maxPI=1535 '
        'clobber=yes',
    )

    files = {'LE_GTI': gti_file, 'LE_SCREEN': out_file}
    status = {'lescreen': res}

    return files, status


def me_screen(
    grade_file: str,
    gti_normal: str,
    detstat_file: str,
    out_path: str,
    gti_file: str,
    out_file: str,
    intervals: list,
    det_expr: str | None = None,
):
    if det_expr is None:
        det_expr = get_default_det_expr('ME')

    # Note:
    # this assumes that the detstat data for intervals is already included
    # in detstat_file, and no need to generate a new one from megticorr
    out_path = os.path.abspath(out_path)
    os.makedirs(out_path, exist_ok=True)
    gti_file = f'{out_path}/{gti_file}'
    out_file = f'{out_path}/{out_file}'

    generate_hxmt_gti(gti_normal, intervals, gti_file)

    res = run_cmd(
        'mescreen '
        f'evtfile="{grade_file}" '
        f'gtifile="{gti_file}" '
        f'outfile="{out_file}" '
        f'baddetfile="{detstat_file}" '
        f'userdetid="{det_expr}" '
        'starttime=0 '
        'stoptime=0 '
        'minPI=0 '
        'maxPI=1023 '
        'clobber=yes',
    )

    files = {'ME_GTI': gti_file, 'ME_SCREEN': out_file}
    status = {'mescreen': res}

    return files, status


def he_screen(
    pi_file: str,
    gti_normal: str,
    out_path: str,
    gti_file: str,
    out_file: str,
    intervals: list,
    det_expr: str | None = None,
):
    if det_expr is None:
        det_expr = get_default_det_expr('HE')

    out_path = os.path.abspath(out_path)
    os.makedirs(out_path, exist_ok=True)
    gti_file = f'{out_path}/{gti_file}'
    out_file = f'{out_path}/{out_file}'

    generate_hxmt_gti(gti_normal, intervals, gti_file)

    res = run_cmd(
        'hescreen '
        f'evtfile="{pi_file}" '
        f'gtifile="{gti_file}" '
        f'outfile="{out_file}" '
        f'userdetid="{det_expr}" '
        'eventtype=1 '
        'anticoincidence=yes '
        'starttime=0 '
        'stoptime=0 '
        'minPI=0 '
        'maxPI=255 '
        'clobber=yes',
    )

    files = {'HE_GTI': gti_file, 'HE_SCREEN': out_file}
    status = {'hescreen': res}

    return files, status


def batch_screen(
    files: dict[str, dict[str, str]],
    obsid_gti_outpath: dict[str, dict[str, tuple[tuple[float, ...], str]]],
    screen_postfix: str,
    gti_postfix: str,
    files_json_path: str,
    status_json_path: str,
    le_det_expr: str | None = None,
    me_det_expr: str | None = None,
    he_det_expr: str | None = None,
    ncpu: int = mp.cpu_count() - 1,
):
    if le_det_expr is None:
        le_det_expr = get_default_det_expr('LE')
    if me_det_expr is None:
        me_det_expr = get_default_det_expr('ME')
    if he_det_expr is None:
        he_det_expr = get_default_det_expr('HE')

    def generate_table():
        table = Table(title='HXMT Data Screen', title_style='bold')
        table.add_column('ObsID', no_wrap=True, justify='center')
        table.add_column('#', no_wrap=True, justify='center')
        table.add_column('LE', no_wrap=True, justify='center')
        table.add_column('ME', no_wrap=True, justify='center')
        table.add_column('HE', no_wrap=True, justify='center')
        for obsid, i in obsid_gti_outpath.items():
            for n in i:
                row = [obsid, n]
                for det in ('LE', 'ME', 'HE'):
                    s = screen_status[obsid][n].get(det, {})
                    if s:
                        if all(si['success'] for si in s.values()):
                            mark = '✅'
                        else:
                            mark = '❌'
                    else:
                        mark = '🕘'
                    row.append(mark)
                table.add_row(*row)
        return table

    def callback_generator(obsid, n, det, progress, task):
        def handler(res):
            screen_files[obsid][n].update(res[0])
            screen_status[obsid][n][det] = res[1]
            progress.update(task, advance=1, refresh=True)
            progress.update_renderable()

        return handler

    def error_callback_generator(obsid, n, det, progress, task):
        def handler(error):
            screen_files[obsid][n][det] = {
                f'{det}_GTI': '',
                f'{det}_SCREEN': '',
            }
            screen_status[obsid][n][det] = {
                f'{det.lower()}screen': {
                    'success': False,
                    'cmd': '',
                    'code': -1,
                    'stdout': '',
                    'stderr': traceback.format_exception(error),
                }
            }
            progress.update(task, advance=1, refresh=True)
            progress.update_renderable()

        return handler

    screen_files = {
        obsid: {n: {} for n in i} for obsid, i in obsid_gti_outpath.items()
    }
    screen_status = {
        obsid: {n: {} for n in i} for obsid, i in obsid_gti_outpath.items()
    }

    with CustomProgress(generate_table, refresh_per_second=2) as progress:
        total = 3 * sum(len(i) for _, i in obsid_gti_outpath.items())
        task = progress.add_task('', total=total)
        with mp.Pool(ncpu) as pool:
            for obsid, v in obsid_gti_outpath.items():
                for n, (gti, outpath) in v.items():
                    os.makedirs(outpath, exist_ok=True)
                    pool.apply_async(
                        le_screen,
                        args=(),
                        kwds={
                            'recon_file': files[obsid]['LE_RECON'],
                            'gti_normal': files[obsid]['LE_GTI'],
                            'out_path': outpath,
                            'gti_file': f'LE{gti_postfix}',
                            'out_file': f'LE{screen_postfix}',
                            'intervals': gti,
                            'det_expr': le_det_expr,
                        },
                        callback=callback_generator(
                            obsid, n, 'LE', progress, task
                        ),
                        error_callback=error_callback_generator(
                            obsid, n, 'LE', progress, task
                        ),
                    )
                    pool.apply_async(
                        me_screen,
                        args=(),
                        kwds={
                            'grade_file': files[obsid]['ME_GRADE'],
                            'gti_normal': files[obsid]['ME_GTI'],
                            'detstat_file': files[obsid]['ME_DETSTAT'],
                            'out_path': outpath,
                            'gti_file': f'ME{gti_postfix}',
                            'out_file': f'ME{screen_postfix}',
                            'intervals': gti,
                            'det_expr': me_det_expr,
                        },
                        callback=callback_generator(
                            obsid, n, 'ME', progress, task
                        ),
                        error_callback=error_callback_generator(
                            obsid, n, 'ME', progress, task
                        ),
                    )
                    pool.apply_async(
                        he_screen,
                        args=(),
                        kwds={
                            'pi_file': files[obsid]['HE_PI'],
                            'gti_normal': files[obsid]['HE_GTI'],
                            'out_path': outpath,
                            'gti_file': f'HE{gti_postfix}',
                            'out_file': f'HE{screen_postfix}',
                            'intervals': gti,
                            'det_expr': he_det_expr,
                        },
                        callback=callback_generator(
                            obsid, n, 'HE', progress, task
                        ),
                        error_callback=error_callback_generator(
                            obsid, n, 'HE', progress, task
                        ),
                    )
            pool.close()
            pool.join()

    with open(files_json_path, 'w') as f:
        json.dump(screen_files, f, indent=4, ensure_ascii=False)

    with open(status_json_path, 'w') as f:
        json.dump(screen_status, f, indent=4, ensure_ascii=False)


def le_lc(
    screen_file: str,
    out_path: str,
    out_file: str,
    binsize: float,
    tstart: float | int = 0,
    tstop: float | int = 0,
    emin: float | int = 2.0,
    emax: float | int = 10.0,
    det_expr: str | None = None,
):
    if det_expr is None:
        det_expr = get_default_det_expr('LE')

    out_path = os.path.abspath(out_path)
    os.makedirs(out_path, exist_ok=True)
    out_file = f'{out_path}/{out_file}'
    pi_min, pi_max = get_pi_range('LE', emin, emax)

    res = run_cmd(
        f'lelcgen '
        f'evtfile="{screen_file}" '
        f'outfile="{out_file}" '
        f'userdetid="{det_expr}" '
        f'binsize={binsize} '
        f'starttime={tstart} '
        f'stoptime={tstop} '
        f'minPI={pi_min} maxPI={pi_max} '
        f'eventtype=1 '
        f'clobber=yes'
    )

    lc_file = get_file_by_pattern(f'{out_file}_g0.lc', throw=False)
    if os.path.exists(out_file):
        os.remove(out_file)
    if os.path.exists(lc_file):
        os.rename(lc_file, out_file)
    else:
        out_file = ''
        res['success'] = False

    files = {'LE_LC': out_file}
    status = {'lelcgen': res}

    return files, status


def le_spec(
    screen_file: str,
    out_path: str,
    out_file: str,
    tstart: float | int = 0,
    tstop: float | int = 0,
    det_expr: str | None = None,
):
    if det_expr is None:
        det_expr = get_default_det_expr('LE')

    out_path = os.path.abspath(out_path)
    os.makedirs(out_path, exist_ok=True)
    out_file = f'{out_path}/{out_file}'

    res = run_cmd(
        'lespecgen '
        f'evtfile="{screen_file}" '
        f'outfile="{out_file}" '
        'eventtype=1 '
        f'userdetid="{det_expr}" '
        f'starttime={tstart} '
        f'stoptime={tstop} '
        'minPI=0 '
        'maxPI=1535 '
        'clobber=yes'
    )

    spec_file = get_file_by_pattern(f'{out_file}_g0.pha', throw=False)
    if os.path.exists(out_file):
        os.remove(out_file)
    if os.path.exists(spec_file):
        os.rename(spec_file, out_file)
    else:
        out_file = ''
        res['success'] = False

    files = {'LE_SPEC': out_file}
    status = {'lespecgen': res}

    return files, status


def le_rsp(
    spec_file: str,
    att_file: str,
    temp_file: str,
    out_path: str,
    out_file: str,
    ra: float | int = -1,
    dec: float | int = -91,
):
    out_path = os.path.abspath(out_path)
    os.makedirs(out_path, exist_ok=True)
    out_file = f'{out_path}/{out_file}'

    res = run_cmd(
        'lerspgen '
        f'phafile="{spec_file}" '
        f'outfile="{out_file}" '
        f'attfile="{att_file}" '
        f'tempfile="{temp_file}" '
        f'ra={ra} '
        f'dec={dec} '
        'clobber=yes'
    )

    if not os.path.exists(out_file):
        out_file = ''

    files = {'LE_RSP': out_file}
    status = {'lerspgen': res}

    return files, status


def me_lc(
    screen_file: str,
    dtime_file: str,
    out_path: str,
    out_file: str,
    binsize: float,
    tstart: float | int = 0,
    tstop: float | int = 0,
    emin: float | int = 10.0,
    emax: float | int = 35.0,
    det_expr: str | None = None,
):
    if det_expr is None:
        det_expr = get_default_det_expr('ME')

    out_path = os.path.abspath(out_path)
    os.makedirs(out_path, exist_ok=True)
    out_file = f'{out_path}/{out_file}'
    pi_min, pi_max = get_pi_range('ME', emin, emax)

    res = run_cmd(
        'melcgen '
        f'evtfile="{screen_file}" '
        f'outfile="{out_file}" '
        f'deadfile="{dtime_file}" '
        f'userdetid="{det_expr}" '
        f'binsize={binsize} '
        f'starttime={tstart} '
        f'stoptime={tstop} '
        f'minPI={pi_min} maxPI={pi_max} '
        'deadcorr=no '
        'clobber=yes'
    )

    lc_file = get_file_by_pattern(f'{out_file}_g0.lc', throw=False)
    if os.path.exists(out_file):
        os.remove(out_file)
    if os.path.exists(lc_file):
        os.rename(lc_file, out_file)
    else:
        out_file = ''
        res['success'] = False

    files = {'ME_LC': out_file}
    status = {'melcgen': res}

    return files, status


def me_spec(
    screen_file: str,
    dtime_file: str,
    out_path: str,
    out_file: str,
    tstart: float | int = 0,
    tstop: float | int = 0,
    det_expr: str | None = None,
):
    if det_expr is None:
        det_expr = get_default_det_expr('ME')

    out_path = os.path.abspath(out_path)
    os.makedirs(out_path, exist_ok=True)
    out_file = f'{out_path}/{out_file}'

    res = run_cmd(
        'mespecgen '
        f'evtfile="{screen_file}" '
        f'outfile="{out_file}" '
        f'deadfile="{dtime_file}" '
        f'userdetid="{det_expr}" '
        f'starttime={tstart} '
        f'stoptime={tstop}'
        'minPI=0 '
        'maxPI=1023 '
        'clobber=yes'
    )

    spec_file = get_file_by_pattern(f'{out_file}_g0.pha', throw=False)
    if os.path.exists(out_file):
        os.remove(out_file)
    if os.path.exists(spec_file):
        os.rename(spec_file, out_file)
    else:
        out_file = ''
        res['success'] = False

    files = {'ME_SPEC': out_file}
    status = {'mespecgen': res}

    return files, status


def me_rsp(
    spec_file: str,
    att_file: str,
    out_path: str,
    out_file: str,
    ra: float | int = -1,
    dec: float | int = -91,
):
    out_path = os.path.abspath(out_path)
    os.makedirs(out_path, exist_ok=True)
    out_file = f'{out_path}/{out_file}'

    res = run_cmd(
        'merspgen '
        f'phafile="{spec_file}" '
        f'outfile="{out_file}" '
        f'attfile="{att_file}" '
        f'ra={ra} '
        f'dec={dec} '
        'clobber=yes'
    )

    if not os.path.exists(out_file):
        out_file = ''

    files = {'ME_RSP': out_file}
    status = {'merspgen': res}

    return files, status


def he_lc(
    screen_file: str,
    dtime_file: str,
    out_path: str,
    out_file: str,
    binsize: float,
    tstart: float | int = 0,
    tstop: float | int = 0,
    emin: float | int = 20.0,
    emax: float | int = 200.0,
    det_expr: str | None = None,
):
    if det_expr is None:
        det_expr = get_default_det_expr('HE')

    out_path = os.path.abspath(out_path)
    os.makedirs(out_path, exist_ok=True)
    out_file = f'{out_path}/{out_file}'
    pi_min, pi_max = get_pi_range('HE', emin, emax)

    res = run_cmd(
        'helcgen '
        f'evtfile="{screen_file}" '
        f'outfile="{out_file}" '
        f'deadfile="{dtime_file}" '
        f'userdetid="{det_expr}" '
        f'binsize={binsize} '
        f'starttime={tstart} '
        f'stoptime={tstop} '
        f'minPI={pi_min} '
        f'maxPI={pi_max} '
        'deadcorr=no '
        'clobber=yes'
    )

    lc_file = get_file_by_pattern(f'{out_file}_g0.lc', throw=False)
    if os.path.exists(out_file):
        os.remove(out_file)
    if os.path.exists(lc_file):
        os.rename(lc_file, out_file)
    else:
        out_file = ''
        res['success'] = False

    files = {'HE_LC': out_file}
    status = {'helcgen': res}

    return files, status


def he_spec(
    screen_file: str,
    dtime_file: str,
    out_path: str,
    out_file: str,
    tstart: float | int = 0,
    tstop: float | int = 0,
    det_expr: str | None = None,
):
    if det_expr is None:
        det_expr = get_default_det_expr('HE')

    out_path = os.path.abspath(out_path)
    os.makedirs(out_path, exist_ok=True)
    out_file = f'{out_path}/{out_file}'

    res = run_cmd(
        'hespecgen '
        f'evtfile="{screen_file}" '
        f'outfile="{out_file}" '
        f'deadfile="{dtime_file}" '
        f'userdetid="{det_expr}" '
        f'starttime={tstart} '
        f'stoptime={tstop} '
        'minPI=0 '
        'maxPI=255 '
        'clobber=yes'
    )

    spec_file = get_file_by_pattern(f'{out_file}_g0.pha', throw=False)
    if os.path.exists(out_file):
        os.remove(out_file)
    if os.path.exists(spec_file):
        os.rename(spec_file, out_file)
    else:
        out_file = ''
        res['success'] = False

    files = {'HE_SPEC': out_file}
    status = {'hespecgen': res}

    return files, status


def he_rsp(
    spec_file: str,
    att_file: str,
    out_path: str,
    out_file: str,
    ra: float | int = -1,
    dec: float | int = -91,
):
    out_path = os.path.abspath(out_path)
    os.makedirs(out_path, exist_ok=True)
    out_file = f'{out_path}/{out_file}'

    res = run_cmd(
        'herspgen '
        f'phafile="{spec_file}" '
        f'outfile="{out_file}" '
        f'attfile="{att_file}" '
        f'ra={ra} '
        f'dec={dec} '
        'clobber=yes'
    )

    if not os.path.exists(out_file):
        out_file = ''

    files = {'HE_RSP': out_file}
    status = {'herspgen': res}

    return files, status


def generate_counts_phaii(
    specfile: str,
    tstart: NDArray,
    tstop: NDArray,
    channel: NDArray,
    emin: NDArray,
    emax: NDArray,
    counts: NDArray,
    exposure: NDArray,
    telescope: str,
    instrument: str,
    detname: str,
    quality: NDArray | None = None,
    group: NDArray | None = None,
    backfile: NDArray | None = None,
    respfile: NDArray | None = None,
    chantype: Literal['PHA', 'PI'] = 'PI',
    spectype: Literal['TOTAL', 'BKG', 'NET'] = 'TOTAL',
) -> None:
    """Generate spectral file containing counts data.

    Parameters
    ----------
    specfile : str
    tstart : (t,) array_like
    tstop : (t,) array_like
    channel : (c,) array_like
    emin : (c,) array_like
    emax : (c,) array_like
    counts : (t, c) array_like
    exposure : (t,) array_like
    telescope : str
    instrument : str
    detname : str
    quality : None or (t, c) array_like
    group : None or (t, c) array_like
    backfile : None or (t,) array_like
    respfile : None or (t,) array_like
    chantype : {'PHA', 'PI'}
        Channel type.
    spectype : {'TOTAL', 'BKG', 'NET'}
        Spectral type.
    """
    tstart = np.atleast_1d(np.array(tstart, dtype=np.float64))
    tstop = np.atleast_1d(np.array(tstop, dtype=np.float64))
    channel = np.array(channel, dtype=np.int64)
    emin = np.array(emin, dtype=np.float64)
    emax = np.array(emax, dtype=np.float64)
    counts = np.array(counts, dtype=np.float64)
    exposure = np.atleast_1d(np.array(exposure, dtype=np.float64))

    if not tstart.shape == tstop.shape == exposure.shape:
        raise ValueError(
            f'tstart {tstart.shape}, tstop {tstop.shape}, and exposure '
            f'{exposure.shape} are not matched'
        )
    if not channel.shape == emin.shape == emax.shape == (counts.shape[1],):
        raise ValueError(
            f'channel {channel.shape}, emin {emin.shape}, emax {emax.shape} '
            f'and counts {counts.shape} are not matched'
        )

    if not counts.shape[0] == tstart.shape[0]:
        raise ValueError(f'counts {counts.shape} and time {tstart.shape[0]})')

    if group is not None:
        group = np.array(group, dtype=np.int64)
        if not group.shape == counts.shape:
            raise ValueError(
                f'group {group.shape} and counts {counts.shape} are not '
                'matched'
            )
    else:
        group = np.ones_like(counts)

    if quality is not None:
        quality = np.array(quality, dtype=np.int64)
        if not quality.shape == counts.shape:
            raise ValueError(
                f'quality {quality.shape} and counts {counts.shape} are not '
                'matched'
            )
    else:
        quality = np.zeros_like(counts)

    if backfile is not None:
        if type(backfile) is str:
            if '{' in backfile and '}' in backfile:
                backfile = np.array([backfile] * tstart.size)
            else:
                backfile = np.array(
                    [backfile + f'{{{i + 1}}}' for i in range(tstart.size)],
                )
        else:
            backfile = np.array(backfile, dtype=str)
            if not backfile.shape == tstart.shape:
                raise ValueError(
                    f'backfile {backfile.shape} and tstart {tstart.shape} are '
                    'not matched'
                )
    else:
        backfile = np.array(['' for _ in range(tstart.size)], dtype=str)

    if respfile is not None:
        if type(respfile) is str:
            respfile = np.array(
                [respfile for _ in range(tstart.size)], dtype=str
            )
        else:
            respfile = np.array(respfile, dtype=str)
            if not respfile.shape == tstart.shape:
                raise ValueError(
                    f'respfile {respfile.shape} and tstart {tstart.shape} are '
                    'not matched'
                )
    else:
        respfile = np.array(['' for _ in range(tstart.size)], dtype=str)

    primary = fits.PrimaryHDU()
    creator = 'XRB221021 util v0.1'
    primary.header['CREATOR'] = (creator, 'Software and version creating file')
    primary.header['FILETYPE'] = ('PHAII', 'Name for this type of FITS file')
    primary.header['FILE-VER'] = (
        '1.0.0',
        'Version of the format for this filetype',
    )
    primary.header['TELESCOP'] = (telescope, 'Name of mission/satellite')
    primary.header['INSTRUME'] = (
        instrument,
        'Specific instrument used for observation',
    )
    primary.header['DETNAM'] = (detname, 'Individual detector name')
    primary.header['FILENAME'] = (specfile.split('/')[-1], 'Name of this file')

    ebounds_columns = [
        fits.Column(name='CHANNEL', format='1I', array=channel),
        fits.Column(name='E_MIN', format='1E', unit='keV', array=emin),
        fits.Column(name='E_MAX', format='1E', unit='keV', array=emax),
    ]
    ebounds = fits.BinTableHDU.from_columns(ebounds_columns)
    ebounds.header['EXTNAME'] = (
        'EBOUNDS',
        'Name of this binary table extension',
    )
    ebounds.header['TELESCOP'] = (telescope, 'Name of mission/satellite')
    ebounds.header['INSTRUME'] = (
        instrument,
        'Specific instrument used for observation',
    )
    ebounds.header['DETNAM'] = (detname, 'Individual detector name')
    ebounds.header['HDUCLASS'] = (
        'OGIP',
        'Conforms to OGIP standard indicated in HDUCLAS1',
    )
    ebounds.header['HDUCLAS1'] = (
        'RESPONSE',
        'These are typically found in RMF files',
    )
    ebounds.header['HDUCLAS2'] = ('EBOUNDS', 'From CAL/GEN/92-002')
    ebounds.header['HDUVERS'] = ('1.2.1', 'Version of HDUCLAS1 format in use')
    ebounds.header['CHANTYPE'] = (chantype, 'Channel type')
    ebounds.header['DETCHANS'] = (
        len(channel),
        'Total number of channels in each rate',
    )

    spectrum_columns = [
        fits.Column(name='TIME', format='1D', unit='s', array=tstart),
        fits.Column(name='ENDTIME', format='1D', unit='s', array=tstop),
        fits.Column(name='EXPOSURE', format='1E', unit='s', array=exposure),
        fits.Column(name='COUNTS', format=f'{len(channel)}D', array=counts),
        fits.Column(name='QUALITY', format=f'{len(channel)}I', array=quality),
        fits.Column(name='GROUPING', format=f'{len(channel)}I', array=group),
        fits.Column(name='BACKFILE', format='150A', array=backfile),
        fits.Column(name='RESPFILE', format='150A', array=respfile),
    ]
    spectrum = fits.BinTableHDU.from_columns(spectrum_columns)
    spectrum.header['EXTNAME'] = (
        'SPECTRUM',
        'Name of this binary table extension',
    )
    spectrum.header['TELESCOP'] = (telescope, 'Name of mission/satellite')
    spectrum.header['INSTRUME'] = (
        instrument,
        'Specific instrument used for observation',
    )
    spectrum.header['DETNAM'] = (detname, 'Individual detector name')
    spectrum.header['AREASCAL'] = (
        1.0,
        'No special scaling of effective area by channel',
    )
    spectrum.header['BACKSCAL'] = (1.0, 'No scaling of background')
    spectrum.header['CORRSCAL'] = (0.0, 'Correction scaling file')
    spectrum.header['ANCRFILE'] = (
        'none',
        'Name of corresponding ARF file (if any)',
    )
    spectrum.header['SYS_ERR'] = (0, 'No systematic errors')
    spectrum.header['POISSERR'] = (True, 'Assume Poisson Errors')
    spectrum.header['GROUPING'] = (1, 'Grouping of the data has been defined')
    spectrum.header['QUALITY'] = (1, 'Data quality information specified')
    spectrum.header['HDUCLASS'] = (
        'OGIP',
        'Conforms to OGIP standard indicated in HDUCLAS1',
    )
    spectrum.header['HDUCLAS1'] = (
        'SPECTRUM',
        'PHA dataset (OGIP memo OGIP-92-007)',
    )
    spectrum.header['HDUCLAS2'] = (spectype, 'Indicates TOTAL/NET/BKG data')
    spectrum.header['HDUCLAS3'] = ('COUNT', 'Indicates data stored as counts')
    spectrum.header['HDUCLAS4'] = (
        'TYPE:II',
        'Indicates PHA Type II file format',
    )
    spectrum.header['HDUVERS'] = ('1.2.1', 'Version of HDUCLAS1 format in use')
    spectrum.header['CHANTYPE'] = (chantype, 'Channel type')
    spectrum.header['DETCHANS'] = (
        len(channel),
        'Total number of channels in each rate',
    )

    fits.HDUList([primary, ebounds, spectrum]).writeto(
        specfile, overwrite=True
    )
