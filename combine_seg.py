# Combine segmented FITS files into one file
# making sure we have all the segments

from astropy.io import fits
import numpy as np
import pandas as pd
from pathlib import Path

# Find all segmented FITS files
fits_fns = sorted(Path('fitsfiles').glob('*seg*fits'))

# Make it a DF so we can group by Visit ID
fits_df = pd.DataFrame({"filename": fits_fns,})
# Extract info from each file
fits_df['visit'] = fits_df.apply(
    lambda x: str(x['filename']).split('/')[-1].split('_')[0],
    axis=1)


def parse_params(row):
    with fits.open(row['filename']) as hdul:
        row['nints'] = hdul[0].header['NINTS']
        row['ngroups'] = hdul[0].header['NGROUPS']
        row['int_start'] = hdul[0].header['INTSTART']
        row['int_end'] = hdul[0].header['INTEND']
        row['detector'] = hdul[0].header['DETECTOR']
    return row


# Parse info from FITS headers
fits_df = fits_df.apply(parse_params, axis=1)


def check_and_combine(group):
    # Check we have all the segments for a collection and if yes, combine
    check_multi_files = len(group) > 1
    check_first_int = min(group['int_start']) == 1
    check_last_int = max(group['int_end']) == group['nints'].iloc[0]
    check_cont_int = np.all([
        group['int_start'].iloc[i+1] == (group['int_end'].iloc[i] + 1)
        for i in range(len(group)-1)
    ])
    # Combine if possible
    if np.all([check_multi_files, check_first_int, check_last_int, check_cont_int]):
        out_path = group['filename'].iloc[0].with_name(
            group['filename'].iloc[0].name.replace('-seg001', '-ALL')
        )
        print(f'Combining {group["filename"].values} into {out_path} ...')
        # Combine the record tables
        groups_list = []
        inttimes_list = []
        primheader_list = []
        for fits_fn in group['filename']:
            with fits.open(fits_fn) as hdul:
                groups_list.append(np.array(hdul['GROUP'].data))
                inttimes_list.append(np.array(hdul['INT_TIMES'].data))
                primheader_list.append(hdul[0].header.copy())
        groups_all = np.concatenate(groups_list)
        inttimes_all = np.concatenate(inttimes_list)
        # Generate the primary header for the combined file
        primheader_all = primheader_list[0]
        primheader_all['INTSTART'] = primheader_list[0]['INTSTART']
        primheader_all['INTEND'] = primheader_list[-1]['INTEND']
        # Save into a FITS file
        fits.HDUList([
            fits.PrimaryHDU(header=primheader_all),
            fits.BinTableHDU(groups_all, name='GROUP'),
            fits.BinTableHDU(inttimes_all, name='INT_TIMES'),
        ]).writeto(out_path, overwrite=True)


# Check whether each segmented collection is complete then combine
fits_df.groupby(['visit', 'detector']).apply(check_and_combine)