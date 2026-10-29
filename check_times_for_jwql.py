# Check times from GROUP and INT_TIMES table with new algorithm do not differ
# too much from old algorithm
import pandas as pd
from astropy.io import fits
from astropy.time import Time
from datetime import datetime, timedelta
import numpy as np
from pathlib import Path
import sqlite3


# ===== MAIN FUNCTIONS ========================================================

def compare_times(fits_fn):

    # Open FITS
    with fits.open(fits_fn) as fits_hdul:

        # Getting parameters from primary header
        ngroups = fits_hdul[0].header['NGROUPS']
        nframes = fits_hdul[0].header['NFRAMES']
        nints = fits_hdul[0].header['NINTS']
        groupgap = fits_hdul[0].header['GROUPGAP']
        nresets = fits_hdul[0].header['NRESETS']
        tframe = fits_hdul[0].header['TFRAME']
        drpfrms3 = fits_hdul[0].header['DRPFRMS3']
        instrume = fits_hdul[0].header['INSTRUME']
        subarray = fits_hdul[0].header['SUBARRAY']
        readpatt = fits_hdul[0].header['READPATT']
        zerofram = fits_hdul[0].header['ZEROFRAM']

        # Segmented files
        if 'seg' in str(fits_fn):
            intstart = fits_hdul[0].header['INTSTART']
            intend = fits_hdul[0].header['INTEND']
        else:
            intstart = 1
            intend = nints

        # Get numResetRows from db file
        nresetrows = get_num_reset_rows(
            instrume=instrume, subarray=subarray, readpatt=readpatt
        )

        # Getting all info from GROUP and INT_TIMES tables
        groups_rec = fits_hdul['GROUP'].data
        inttimes_rec = fits_hdul['INT_TIMES'].data
        # Convert to pd.DF to make it easier to play with
        groups_df = pd.DataFrame(groups_rec)
        inttimes_df = pd.DataFrame(inttimes_rec)
        # Drop bary_end_time and helio_end_time because of data type issue
        groups_df.drop(columns=['bary_end_time', 'helio_end_time'],
                       inplace=True)

        # The group_number and integration_number from the records table is
        # giving the first group/int included in the next "image". We want the
        # last group/int instead so we shift all entries by 1 group down,
        # remove the first entry, and add a last entry ngroups/intend

        # Get "image" group and integration ending numbers
        group_numbers = groups_df['group_number'].to_numpy()
        integration_numbers = groups_df['integration_number'].to_numpy()
        group_numbers = group_numbers - 1
        if zerofram:
            integration_numbers = np.where(
                group_numbers == -1,
                integration_numbers - 1,
                integration_numbers
            )
            group_numbers = np.where(
                group_numbers == -1,
                ngroups,
                group_numbers
            )
        else:
            integration_numbers = np.where(
                group_numbers == 0,
                integration_numbers - 1,
                integration_numbers
            )
            group_numbers = np.where(
                group_numbers == 0,
                ngroups,
                group_numbers
            )
        # Store new values, skip first, add last
        groups_df['end_group_number'] = np.append(group_numbers[1:], ngroups)
        groups_df['end_integration_number'] = np.append(integration_numbers[1:], intend)

        # ----- Implement JWSTSDP code to compute the time entries -----

        # Determine how many total groups there are in this exposure
        tot_grps = ngroups * nints
        # Calculate the duration of a single group
        grp_duration = nframes * tframe
        # Calculate the time between groups
        grp_gap_time = groupgap * tframe
        # Calculate background row reset duration
        bkgd_row_reset_dur = 0.00001
        # Calculate the gap between integrations
        int_gap_time = (nresetrows * bkgd_row_reset_dur) \
                       + (nresets * tframe) \
                       + (drpfrms3 * tframe)

        # Obtain final group end time, convert to datetime object
        dt_format = "%Y-%m-%dT%H:%M:%S.%f"
        group_end_datetime = pd.to_datetime(
            groups_df['group_end_time'].iloc[-1])

        # Recursively calculate all group times backwards from the last entry
        all_group_end_times = []
        all_grp_nums = []
        all_int_nums = []
        # Make lists for entries that are in the existing table, starting from last
        new_group_end_times = [groups_df['group_end_time'].iloc[-1]]
        # Loop through the total number of groups, starting with the final packets group_end_time
        # to determine the group start and end times for each group in each integration
        grp_num = ngroups
        int_num = intend
        for i in range(1, tot_grps):
            # If we're at the beginning of an integration, the grp_time is longer
            if (i % ngroups) == 0:
                grp_time = grp_duration + int_gap_time
                grp_num = ngroups
                int_num -= 1
            # Normal group time between each group within an integration
            else:
                grp_time = grp_duration + grp_gap_time
                grp_num -= 1

            # break in segmented files
            if int_num < intstart:
                break

            # Calculate times
            group_end_datetime = group_end_datetime - timedelta(seconds=grp_time)
            # Store all times for model
            all_group_end_times.append(group_end_datetime.strftime(dt_format))
            all_grp_nums.append(grp_num)
            all_int_nums.append(int_num)
            # Add new times if the old record has entry for (int_num, grp_num)
            if len(groups_df.loc[(groups_df['end_group_number'] == grp_num)
                                 & (groups_df['end_integration_number'] == int_num)]) == 1:
                new_group_end_times.append(group_end_datetime.strftime(dt_format))

        # Add new column
        groups_df['new_group_end_time'] = np.flip(new_group_end_times)

        # Compute difference NEW - OLD
        groups_df['diff_group_end_time'] = (
                pd.to_datetime(groups_df['new_group_end_time'])
                - pd.to_datetime(groups_df['group_end_time'])
        ) / timedelta(seconds=1)

        # Check if any difference is greater than 64ms
        n_discr = np.sum(
            groups_df['diff_group_end_time'].abs() > 64e-3)
        avg_discr = np.mean(groups_df['diff_group_end_time'])
        max_discr = np.max(np.abs(groups_df['diff_group_end_time']))

        # Correct all times by average discrepancy to account for systematics
        groups_df['corr_diff_group_end_time'] = (
                groups_df['diff_group_end_time'] - avg_discr)
        # Check if any difference is greater than 64ms after offset correction
        n_discr_corr = np.sum(
            groups_df['corr_diff_group_end_time'].abs() > 64e-3)
        avg_discr_corr = np.mean(groups_df['corr_diff_group_end_time'])
        max_discr_corr = np.max(np.abs(groups_df['corr_diff_group_end_time']))

        # Print results on screen for now
        if n_discr_corr > 0:
            print(f'{fits_fn}, {readpatt},'
                  f' {n_discr / len(groups_rec)*100:.1f}%,'
                  f' {avg_discr:.5f}, {max_discr:.5f}',
                  f' ---> {n_discr_corr / len(groups_rec)*100:.1f}%,'
                  f' {avg_discr_corr:.5f}, {max_discr_corr:.5f}')

        # ===== TODO: rebuild the INT_TIMES table with a model

        # Offset all new times for model
        all_group_end_times = np.array([
            datetime.fromisoformat(s)
            for s in new_group_end_times
        ]) - timedelta(seconds=avg_discr)

        # Then we have a model of the times, from which we can extract those
        # for each integration and build the INT_TIMES table
        new_int_end_times = []


def get_num_reset_rows(instrume: str, subarray: str, readpatt: str) -> float:
    """Return NumResetRows and FrameReadTime from SubarrayFrameTime table for given instrument
    and subarray."""

    num_reset_rows = 0
    where_oss = f"AND OSS_Version <= '012.000.000.000' ORDER BY OSS_Version DESC LIMIT 1"

    query = (f"SELECT NumResetRows "
             f"FROM SubarrayFrameTime "
             f"WHERE UPPER(Instrument) = '{instrume}' "
             f"AND SubarrayName = '{subarray}' ")

    query1 = query + where_oss

    # Connect to the prd.db file and extract the necessary information
    # +++ TFD NOTE THIS MIGHT STILL CHANGE BASED ON FEEDBACK FROM SEs
    try:
        prd_db = sqlite3.connect('prd.db')
        db_cursor = prd_db.cursor()
        # print("Executing: ", query1)
        db_cursor.execute(query1)
        db_row = db_cursor.fetchone()
        num_reset_rows = db_row[0]
        if num_reset_rows is None:
            # Try query with ReadoutPattern
            # print("NumResetRows not found, try with ReadoutPattern.")
            where_readpatt = f"AND ReadoutPattern = '{readpatt}' "
            query2 = query + where_readpatt + where_oss
            # print("Executing: ", query2)
            db_cursor.execute(query2)
            db_row = db_cursor.fetchone()
            num_reset_rows = db_row[0]
            if num_reset_rows is None:
                # print("NumResetRows not found, using value of 0.")
                num_reset_rows = 0
    except sqlite3.Error as err:
        # print(f"NumResetRows query FAILED: {err}")
        raise
    finally:
        if prd_db:
            prd_db.close()

    return num_reset_rows


# ===== EXECUTION =============================================================

all_nints = []
all_ngroups = []
all_cadences_min = []
all_cadences = []
all_cadences_max = []
all_obsids = []
all_tframes = []

# Find all FITS files
fits_fns = sorted(Path('fitsfiles').glob('*fits'))

print('Files with new group end times more than 64ms away from the old')
print('fits_fn, readpatt, f_discrepancies, mean_discrepancy(ms), max_discrepancy(ms)'
      ' ---> after average offset correction')
for fits_fn in fits_fns:
    compare_times(fits_fn)
