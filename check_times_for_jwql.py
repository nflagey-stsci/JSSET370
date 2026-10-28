# Check times from GROUP and INT_TIMES table with new algorithm do not differ
# too much from old algorithm

from astropy.io import fits
from astropy.time import Time
from datetime import datetime, timedelta
import numpy as np
from pathlib import Path
import sqlite3


# ===== MAIN FUNCTIONS ========================================================

def compare_times(fits_fn):

    # Open FITS
    with (fits.open(fits_fn, mode='update') as fits_hdul):

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

        # Getting all info from GROUP and INT_TIMES tables
        groups_rec = fits_hdul['GROUP'].data
        inttimes_rec = fits_hdul['INT_TIMES'].data

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
        group_end_datetime = datetime.strptime(groups_rec[-1]['group_end_time'],
                                               dt_format)

        # Recursively calculate all group times backwards from the last entry
        all_group_end_times = []
        all_grp_nums = []
        all_int_nums = []
        # Make lists for entries that are in the existing table
        new_group_end_times = [group_end_datetime.strftime(dt_format)]
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
            # Store all times for model
            all_group_end_times.append(group_end_datetime.strftime(dt_format))
            all_grp_nums.append(grp_num)
            all_int_nums.append(int_num)
            # Add new times if the old record has  entry for (int_num, grp_num)
            if len(groups_rec[(groups_rec['group_number'] == grp_num)
                              & (groups_rec['integration_number'] == int_num)]) == 1:
                new_group_end_times.append(group_end_datetime.strftime(dt_format))

        # Reverse order to match original table
        new_end_datetime = np.flip(np.array([
            datetime.fromisoformat(s)
            for s in new_group_end_times
        ]))
        # Difference between OLD and NEW in seconds
        old_end_datetime = np.array([
            datetime.fromisoformat(s)
            for s in groups_rec['group_end_time']
        ])
        diff_end_datetime = (new_end_datetime - old_end_datetime)/timedelta(seconds=1)

        # Check if any difference is greater than 64ms
        n_discrepancies = np.sum(np.abs(diff_end_datetime) > 64e-3)
        avg_discrepancy = np.mean(np.abs(diff_end_datetime))
        max_discrepancy = np.max(np.abs(diff_end_datetime))
        if n_discrepancies > -1:
            print(f'{fits_fn}, {readpatt},'
                  f' {n_discrepancies / len(groups_rec)*100:.1f}%,'
                  f' {avg_discrepancy:.5f}, {max_discrepancy:.5f}')

        # If avg_discrepancy is large, correct all times by this amount
        new_end_datetime -= timedelta(seconds=avg_discrepancy)

        # Offset all new times for model
        all_group_end_times = np.array([
            datetime.fromisoformat(s)
            for s in new_group_end_times
        ]) - timedelta(seconds=avg_discrepancy)

        # Convert times to UTC and MJD for populating the GROUP table
        new_end_utc_mjd = []
        for i in range(len(new_end_datetime)):
            tmp_end_time = new_end_datetime[i].strftime(dt_format)
            new_end_utc_mjd.append(Time(tmp_end_time, format='isot', scale='utc').mjd)

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
print('fits_fn, readpatt, f_discrepancies, mean_discrepancy(ms), max_discrepancy(ms)')
for fits_fn in fits_fns:
    compare_times(fits_fn)
