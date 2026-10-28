# Check times from GROUP table

from astropy.io import fits
from datetime import datetime, timedelta
import numpy as np
from pathlib import Path
import sqlite3


# ===== MAIN FUNCTIONS ========================================================

def compare_times(fits_fn):

    # Open FITS
    with fits.open(fits_fn, mode='update') as fits_hdul:

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

        # Get numResetRows from db file
        nresetrows = get_num_reset_rows(
            instrume=instrume, subarray=subarray, readpatt=readpatt
        )

        # Getting all info from GROUP table
        groups_rec = fits_hdul['GROUP'].data

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
        int_gap_time = (nresetrows * bkgd_row_reset_dur) + (
                    nresets * tframe) + \
                  (drpfrms3 * tframe)

        # Obtain final group end time, convert to datetime object
        dt_format = "%Y-%m-%dT%H:%M:%S.%f"
        group_end_datetime = datetime.strptime(groups_rec[-1]['group_end_time'],
                                               dt_format)
        # Calculate last start and mid times
        group_start_datetime = group_end_datetime - timedelta(seconds=grp_duration)
        group_mid_datetime = group_end_datetime - timedelta(seconds=grp_duration/2)

        # Recursively calculate all group times backwards from the last entry
        # Make lists for entries that are in the existing table
        new_end_datetime = [group_end_datetime.strftime(dt_format)]
        new_start_datetime = [group_start_datetime.strftime(dt_format)]
        new_mid_datetime = [group_mid_datetime.strftime(dt_format)]
        # Loop through the total number of groups, starting with the final packets group_end_time
        # to determine the group start and end times for each group in each integration
        grp_num = groups_rec[-1]['group_number']
        int_num = groups_rec[-1]['integration_number']
        for i in range(1, tot_grps):
            grp_num -= 1
            # If we're at the beginning of an integration, the grp_time is longer
            if grp_num == 0:
                grp_time = grp_duration + int_gap_time
                grp_num = ngroups
                int_num -= 1
            # Normal group time between each group within an integration
            else:
                grp_time = grp_duration + grp_gap_time
            # Calculate times
            group_end_datetime = group_end_datetime - timedelta(seconds=grp_time)
            group_start_datetime = group_start_datetime - timedelta(seconds=grp_time)
            group_mid_datetime = group_start_datetime - timedelta(seconds=grp_time/2)
            # Add new times if the old record has  entry for (int_num, grp_num)
            if len(groups_rec[(groups_rec['group_number'] == grp_num)
                              & (groups_rec['integration_number'] == int_num)]) == 1:
                new_end_datetime.append(group_end_datetime.strftime(dt_format))
                new_start_datetime.append(group_start_datetime.strftime(dt_format))
                new_mid_datetime.append(group_mid_datetime.strftime(dt_format))

        # Reverse order to match original table
        new_end_datetime = np.flip(np.array([
            datetime.fromisoformat(s)
            for s in new_end_datetime
        ]))
        # Difference between OLD and NEW in seconds
        old_end_datetime = np.array([
            datetime.fromisoformat(s)
            for s in groups_rec['group_end_time']
        ])
        diff_end_datetime = (new_end_datetime - old_end_datetime)/timedelta(seconds=1)

        # Check if any difference is greater than 64ms
        n_discrepancies = np.sum(np.abs(diff_end_datetime) > 64e-3)
        max_discrepancy = np.max(np.abs(diff_end_datetime))
        if n_discrepancies > 0:
            print(fits_fn, readpatt, n_discrepancies, max_discrepancy)


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
print('fits_fn, readpatt, n_discrepancies, max_discrepancy(ms)')
for fits_fn in fits_fns:
    compare_times(fits_fn)
