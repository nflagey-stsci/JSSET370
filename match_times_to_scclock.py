# Check times from GROUP and INT_TIMES table with new algorithm do not differ
# too much from old algorithm
from astropy.io import fits
from datetime import datetime, timedelta
from get_engdb import get_telem
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize_scalar
import sqlite3


# ===== MAIN FUNCTIONS ========================================================

def compare_times(fits_fn, force_tframe=None):

    # Open FITS
    with fits.open(fits_fn) as fits_hdul:

        # ----- Collect header parameters -----

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
        sctarate = fits_hdul[0].header['SCTARATE']
        date_beg = fits_hdul[0].header['DATE-BEG']
        date_end = fits_hdul[0].header['DATE-END']

        # If we want to for a different TFRAME than that in the header
        if force_tframe is not None:
            tframe = force_tframe

        # Get first and last INT for segmented files
        if 'seg' in str(fits_fn):
            int_start = fits_hdul[0].header['INTSTART']
            int_end = fits_hdul[0].header['INTEND']
        else:
            int_start = 1
            int_end = nints

        # ----- Get numResetRows for that visit file -----

        # Get numResetRows from db file
        nresetrows = get_num_reset_rows(
            instrume=instrume, subarray=subarray, readpatt=readpatt
        )

        # ----- Collect GROUP and INT_TIMES tables -----

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
        # remove the first entry, and add a last entry ngroups/nints at the end

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
        groups_df['end_integration_number'] = np.append(integration_numbers[1:], int_end)

        # ----- Fetch SC clock drift telemetry -----

        # define mnemonic and start/end
        mnemonics = [("FGDP_SCTA_OFFSET_D", True)]
        visits = [{"start_time": date_beg, "end_time": date_end}]
        # execute query
        sc_clock_df = get_telem(mnemonics, visits)
        # define expected results for optimizer
        expected = sc_clock_df['value'].values / 1e6

        # ----- Optimize JWSTSDP code to match SC clock drift -----

        def objective(offset):
            # compute new group end times with a free offset parameter
            tmp_df = compute_group_end_times(
                groups_df, ngroups, nints, int_start, int_end,
                nframes, tframe, groupgap, nresets,
                drpfrms3, nresetrows,
                offset=offset
            )
            # interpolate new group end time onto SC clock
            predicted = np.interp(pd.to_datetime(sc_clock_df['time']),
                                 pd.to_datetime(groups_df['new_group_end_time']),
                                 groups_df['corr_diff_group_end_time'])
            # avoid any y-offset
            predicted += np.mean(expected) - np.mean(predicted)
            # return metric of success
            return np.mean((predicted - expected) ** 2)

        # optimize
        result = minimize_scalar(
            objective,
            bounds=(-1e-2, 1e-2),
            method='bounded'
        )
        # TODO: convert offset into a 10us change?
        best_offset = result.x
        # assuming the overhead from Eddie: 0.9512ms
        best_10us = (best_offset - 0.9512e-3) / nresetrows

        breakpoint()

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
        int_gap_time = compute_int_gap_time(
            nresetrows, bkgd_row_reset_dur, nresets, tframe, drpfrms3)

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
        int_num = int_end
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
            if int_num < int_start:
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

        # Add new column with calculated group end times
        groups_df['new_group_end_time'] = np.flip(new_group_end_times)

        # Compute difference NEW - OLD
        groups_df['diff_group_end_time'] = (
                pd.to_datetime(groups_df['group_end_time'])
                - pd.to_datetime(groups_df['new_group_end_time'])
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
        if n_discr_corr > -1:
            print(f'{fits_fn}, {readpatt}, {subarray},'
                  f' {n_discr / len(groups_rec)*100:.1f}%,'
                  f' {avg_discr:.5f}, {max_discr:.5f}',
                  f' ---> {n_discr_corr / len(groups_rec)*100:.1f}%,'
                  f' {avg_discr_corr:.5f}, {max_discr_corr:.5f}')

        # ===== "Total" duration
        old_duration = (
            pd.to_datetime(groups_df['group_end_time'].iloc[-1])
            - pd.to_datetime(groups_df['group_end_time'].iloc[0])
        ) / timedelta(seconds=1)
        new_duration = (
            pd.to_datetime(groups_df['new_group_end_time'].iloc[-1])
            - pd.to_datetime(groups_df['new_group_end_time'].iloc[0])
        ) / timedelta(seconds=1)
        if new_duration != old_duration:
            print(f'Old duration: {old_duration:.5f}s,'
                  f' New duration: {new_duration:.5f}s,'
                  f' Old - New duration: {old_duration - new_duration:.5f}s')

        # What TFRAME would bring OLD and NEW into agreement?




        # ===== TODO: rebuild the INT_TIMES table with a model

        # Offset all new times for model
        all_group_end_times = np.array([
            datetime.fromisoformat(s)
            for s in new_group_end_times
        ]) - timedelta(seconds=avg_discr)

        # Then we have a model of the times, from which we can extract those
        # for each integration and build the INT_TIMES table
        new_int_end_times = []

        # === DEBUG section ===

        # compute time deltas between entries in OLD table
        deltas = (
            pd.to_datetime(groups_df['group_end_time'])
            - pd.to_datetime(groups_df['group_end_time'].shift())
        ) / timedelta(seconds=1)
        # compute time deltas between entries in NEW table
        new_deltas = (
            pd.to_datetime(groups_df['new_group_end_time'])
            - pd.to_datetime(groups_df['new_group_end_time'].shift())
        ) / timedelta(seconds=1)

        # Show time difference between calculation and GROUP table
        fig, ax = plt.subplots(layout='constrained', figsize=(10, 6))
        ax.plot(pd.to_datetime(groups_df['group_end_time']),
                groups_df['corr_diff_group_end_time'], 'k+:',
                label='(OLD - NEW) after correction')
        # overplot SC clock to ground difference
        ax.plot(sc_clock_df['time'], sc_clock_df['value'] / 1e6,
                 'r.', alpha=0.5, label='SC clock to ground difference')
        # labels and stuff
        ax.set_ylabel('Time (s)')
        ax.set_xlabel('Date')
        ax.set_title(fits_fn.name)
        ax.grid()
        plt.legend()
        plt.savefig('plots/' + fits_fn.name.replace('.fits', '.pdf'),
                    bbox_inches='tight')
        plt.show()

        # breakpoint()

        return old_duration, new_duration, groups_df, sc_clock_df


def compute_int_gap_time(nresetrows, bkgd_row_reset_dur, nresets, tframe, drpfrms3, offset=1e-3):
    int_gap_time = (nresetrows * bkgd_row_reset_dur) \
                   + (nresets * tframe) \
                   + (drpfrms3 * tframe)

    return int_gap_time + offset


def compute_group_end_times(df, ngroups, nints, int_start, int_end,
                            nframes, tframe, groupgap, nresets, drpfrms3, nresetrows, offset=1e-3):
    # All arguments are those found in the header of the FITS file except the
    # 10us which we let free to vary: we want to figure out

    # Determine how many total groups there are in this exposure
    tot_grps = ngroups * nints
    # Calculate the duration of a single group
    grp_duration = nframes * tframe
    # Calculate the time between groups
    grp_gap_time = groupgap * tframe
    # Calculate background row reset duration
    bkgd_row_reset_dur = 10e-6
    # Calculate the gap between integrations
    int_gap_time = compute_int_gap_time(
        nresetrows, bkgd_row_reset_dur, nresets, tframe, drpfrms3, offset)

    # Obtain final group end time, convert to datetime object
    dt_format = "%Y-%m-%dT%H:%M:%S.%f"
    group_end_datetime = pd.to_datetime(df['group_end_time'].iloc[-1])

    # Recursively calculate all group times backwards from the last entry
    all_group_end_times = []
    all_grp_nums = []
    all_int_nums = []
    # Make lists for entries that are in the existing table, starting from last
    new_group_end_times = [df['group_end_time'].iloc[-1]]
    # Loop through the total number of groups, starting with the final packets group_end_time
    # to determine the group start and end times for each group in each integration
    grp_num = ngroups
    int_num = int_end
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
        if int_num < int_start:
            break

        # Calculate times
        group_end_datetime = group_end_datetime - timedelta(seconds=grp_time)
        # Store all times for model
        all_group_end_times.append(group_end_datetime.strftime(dt_format))
        all_grp_nums.append(grp_num)
        all_int_nums.append(int_num)
        # Add new times if the old record has entry for (int_num, grp_num)
        if len(df.loc[(df['end_group_number'] == grp_num)
                      & (df['end_integration_number'] == int_num)]) == 1:
            new_group_end_times.append(group_end_datetime.strftime(dt_format))

    # Add new column with calculated group end times
    df['new_group_end_time'] = np.flip(new_group_end_times)

    # Compute difference NEW - OLD
    df['diff_group_end_time'] = (
        pd.to_datetime(df['group_end_time'])
        - pd.to_datetime(df['new_group_end_time'])
    ) / timedelta(seconds=1)

    # Average discrepancy
    avg_discr = np.mean(df['diff_group_end_time'])

    # Correct all times by average discrepancy to account for systematics
    df['corr_diff_group_end_time'] = df['diff_group_end_time'] - avg_discr

    return df



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
        print(f"NumResetRows query FAILED: {err}")
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

all_new_durations = []
all_old_durations = []

# Find all FITS files
fits_fns = sorted(Path('fitsfiles').glob('*fits'))

print('Files with new group end times more than 64ms away from the old')
print('fits_fn, readpatt, subarray, f_discrepancies, mean_discrepancy(ms), max_discrepancy(ms)'
      ' ---> after average offset correction')
for fits_fn in fits_fns:
    # if 'jw05177001001_04102_00001' in fits_fn.name:
    # if ('jw02454' in fits_fn.name) & ('ALLSEG' in fits_fn.name):
    if ('jw01729' in fits_fn.name) & ('ALLSEG' in fits_fn.name):
    # if 'jw03838001001_04102_00001' in fits_fn.name:
    # if 'nrs' in fits_fn.name:
    # if ('2457' in fits_fn.name) & ('nrs' in fits_fn.name):
        old_d, new_d, groups_df, sc_clock_df = compare_times(fits_fn, force_tframe=None)
        all_new_durations.append(new_d)
        all_old_durations.append(old_d)

fig, ax = plt.subplots(layout='constrained', figsize=(10, 6))
ax.plot(all_old_durations, all_new_durations, 'k+', alpha=0.5)
ax.set_xlabel('Old durations (s)')
ax.set_ylabel('New durations (s)')
ax.grid(alpha=0.25)
plt.savefig('plots/new_vs_old_durations.pdf',
            bbox_inches='tight')

fig, ax = plt.subplots(layout='constrained', figsize=(10, 6))
ax.plot(all_old_durations, [old-new for old, new in zip(all_old_durations, all_new_durations)], 'k+', alpha=0.5)
ax.set_xlabel('Old durations (s)')
ax.set_ylabel('Old durations - New durations (s)')
ax.hlines([0.064, -0.064], xmin=0, xmax=max(all_old_durations), color='r', linestyle='dashed')
ax.grid(alpha=0.25)
plt.savefig('plots/diff_vs_old_durations.pdf',
            bbox_inches='tight')



# TODO: investigate jw01729 vs SC clock then incorporate into above code

y_interp = np.interp(pd.to_datetime(sc_clock_df['time']),
                     pd.to_datetime(groups_df['group_end_time']),
                     groups_df['corr_diff_group_end_time'])
