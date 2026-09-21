### Check times from GROUP
from astropy.io import fits
from astropy.time import Time
from datetime import datetime
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.optimize import curve_fit
from scipy.stats import linregress


# Pick FITS file
fits_fn = Path('fitsfiles/jw05168004001_05101_00001_mirimage_uncal.fits')
fits_fn = Path('fitsfiles/jw04979005001_03101_00001-seg001_nrs1_uncal.fits')
# fits_fn = Path('fitsfiles/jw02183117001_03103_00001-seg001_nrcalong_uncal.fits')
fits_fn = Path('fitsfiles/jw02512018001_04102_00001-seg003_nrs2_uncal.fits')
# fits_fn = Path('fitsfiles/jw01373006001_03102_00001_nrcb1_uncal.fits')
fits_hdul = fits.open(fits_fn)

# Getting parameters from primary header
NGROUPS = fits_hdul[0].header['NGROUPS']
NFRAMES = fits_hdul[0].header['NFRAMES']
NINTS = fits_hdul[0].header['NINTS']
GROUPGAP = fits_hdul[0].header['GROUPGAP']
NRESETS = fits_hdul[0].header['NRESETS']
TFRAME = fits_hdul[0].header['TFRAME']
DRPFRMS1 = fits_hdul[0].header['DRPFRMS1']
NRSTSTRT = fits_hdul[0].header['NRSTSTRT']

# Getting all info from GROUP table
groups_rec = fits_hdul['GROUP'].data

# Frame index for each entry on the GROUP table
frame_idx = []
# total number of frames per integratiom
frames_per_int = NGROUPS * NFRAMES + (NGROUPS - 1) * GROUPGAP + NRESETS
for i in range(len(groups_rec)):
    this_int = groups_rec[i]['integration_number']
    this_grp = groups_rec[i]['group_number']
    # compute index and add it to the list
    this_idx = (this_int - 1) * frames_per_int + this_grp * NFRAMES + (this_grp - 1) * GROUPGAP
    # adjust by one up if last group because RESET is part of it
    if this_grp == NGROUPS:
        this_idx += 1
    frame_idx.append(this_idx)
frame_idx = np.array(frame_idx)

# Number of frames that the time difference is measuring
dframe_idx = np.array([a - b for a, b, in zip(frame_idx[1:], frame_idx[:-1])])

# GenCalcExp from OnBoardScript expression
gencalcexp = TFRAME * (NGROUPS * NFRAMES + (NGROUPS - 1) * GROUPGAP) * NINTS
# Then for MIRI, in MIRCALCEXP
exptime = gencalcexp + NRESETS * TFRAME * (NINTS - 1)
pctime = exptime + TFRAME * NINTS * DRPFRMS1

# Compute differences between consecutive GROUP entries
# TODO: check that all entries are in the same day
seconds = (groups_rec['end_milliseconds']/1e3
           + groups_rec['end_submilliseconds']/1e6)
diff_seconds = seconds[1:] - seconds[:-1]

# Remove first row (no diff available) to avoid issues with NaN or 0
new_groups_rec = np.delete(groups_rec, 0).copy()

# Convert group_end_time to a Time
group_end_times = np.array([datetime.fromisoformat(t)
                            for t in new_groups_rec['group_end_time']])

# Plot all differences with inset for zoom on first 10% values
fig, ax = plt.subplots()
ax.plot(group_end_times, diff_seconds, 'kx', alpha=0.1)
# inset
axins = ax.inset_axes(
    [0.5, 0.5, 0.47, 0.47],
    xlim=(group_end_times[0], group_end_times[int(len(group_end_times)/10)]),
    ylim=ax.get_ylim(),
    xticklabels=[], yticklabels=[])
axins.plot(group_end_times, diff_seconds, 'kx', alpha=0.1)
ax.indicate_inset_zoom(axins, edgecolor="cyan")
# date axis formatting
locator = mdates.AutoDateLocator()
formatter = mdates.ConciseDateFormatter(locator)
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(formatter)
# labels and stuff
ax.set_xlabel('Group End Time')
ax.set_ylabel(r'$\Delta$ between GROUP times (s)')
plt.show()


# Find clusters in a rudimentary way (sort values and find gaps)
threshold = 1e-3  # splitting values separated by more than 1ms
diff_seconds_s = np.sort(diff_seconds)
gaps = np.diff(diff_seconds_s)
splits = np.where(gaps > threshold)[0]
# adding 0 and None to have all the splits
splits = np.append(np.insert(splits + 1, 0, 0), None)
# saving some parameters about each split
split_counts = np.zeros(len(splits)-1)
split_nframes = np.zeros(len(splits)-1)
split_mins = np.zeros(len(splits)-1)
split_maxs = np.zeros(len(splits)-1)
split_means = np.zeros(len(splits)-1)
for i, (beg, end) in enumerate(zip(splits[:-1], splits[1:])):
    # Select the difference values
    split_diff_seconds_s = diff_seconds_s[beg:end]
    split_count = end - beg if end is not None else len(diff_seconds) - beg
    split_counts[i] = split_count
    # Get some basic stats
    split_mean = np.mean(split_diff_seconds_s)
    split_means[i] = split_mean
    split_median = np.median(split_diff_seconds_s)
    split_min = min(split_diff_seconds_s)
    split_max = max(split_diff_seconds_s)
    split_mins[i] = split_min
    split_maxs[i] = split_max
    # Infer how many TFRAME this is equivalent to
    split_nframe = np.round(split_median / TFRAME).astype(int)
    split_nframes[i] = split_nframe
    # Display information
    print(f' === SPLIT {i+1} ===')
    print(f'Count = {split_count}')
    print(f' --> equivalent to {split_nframe} frames')
    print(f'Mean (s) = {split_mean:.6f}'
          f' --> {split_mean/split_nframe:.6f}')
    print(f'Median (s) = {split_median:.6f}'
          f' --> {split_median/split_nframe:.6f}')
    print(f'Min (s) = {split_min:.6f}'
          f' --> {split_min/split_nframe:.6f}')
    print(f'Max (s) = {split_max:.6f}'
          f' --> {split_max/split_nframe:.6f}')
    print(f'Range (ms) = {(split_max - split_min)*1e3:.3f}'
          f' --> {(split_max - split_min)*1e3/split_nframe:.6f}')

# Compute properly averaged observed TFRAME
mean_obs_tframe = np.sum(split_means * split_counts)/np.sum(split_counts * split_nframes)
# Uncertainty of ±32ms on all GROUP times
unc_obs_tframe = 32e-3 / np.sqrt(np.sum(split_counts))
print(f'\n=== MEAN OBSERVED TFRAME ===')
print(f'Mean ± unc (s) = {mean_obs_tframe:.6f} ± {unc_obs_tframe:.6f}')


# Plot all normalized differences with inset for zoom on first 10% values
# associate each diff_second to the correct number of frames
bins = np.insert(split_maxs, 0, 0)
idx = np.digitize(diff_seconds, bins, right=True) - 1
diff_nframes = np.where(
    (idx >= 0) & (idx < len(split_nframes)),
    split_nframes[idx],
    0.
)

# plot
fig, ax = plt.subplots()
ax.plot(group_end_times, diff_seconds/diff_nframes, 'kx', alpha=0.25)
# inset
axins = ax.inset_axes(
    [0.5, 0.5, 0.47, 0.47],
    xlim=(group_end_times[0], group_end_times[int(len(group_end_times)/10)]),
    ylim=ax.get_ylim(),
    xticklabels=[], yticklabels=[])
axins.plot(group_end_times, diff_seconds/diff_nframes, 'kx', alpha=0.1)
ax.indicate_inset_zoom(axins, edgecolor="cyan")
# date axis formatting
locator = mdates.AutoDateLocator()
formatter = mdates.ConciseDateFormatter(locator)
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(formatter)
# labels and stuff
ax.set_xlabel('Group End Time')
ax.set_ylabel(r'TFRAME from GROUP times (s)')
plt.show()

# Figure out TFRAME as the total duration divided by nframes
first_get = datetime.fromisoformat(groups_rec['group_end_time'][0])
last_get = datetime.fromisoformat(groups_rec['group_end_time'][-1])
first_time = seconds[0]
last_time = seconds[-1]
# Compute differences
duration = last_time - first_time
duration_get = last_get - first_get
print(f'\n === DURATION and TFRAME ===')
print(f'(from end of first GROUP entry to end of last GROUP entry)')
# Durations
print(f' from group_end_time (s): {duration_get.total_seconds():.3f}')
print(f' from sub/milliseconds (s): {duration:.6f}')
print(f' ---> difference (ms): {(duration - duration_get.total_seconds())*1e3:.3f}')
# How many frames?
total_nframes = frame_idx[-1] - frame_idx[0]
print(f' First frame in GROUP record: {frame_idx[0]}')
print(f' Last frame in GROUP record: {frame_idx[-1]}')
print(f' ---> number of frames for that duration: {total_nframes}')
# TFRAME
print(f' TFRAME from header (s): {TFRAME}')
print(f' ---> inferred TFRAME from group_end_time (s): {duration_get.total_seconds()/total_nframes:.6f}')
print(f' ---> inferred TFRAME from sub/milliseconds (s): {duration/total_nframes:.6f}')

# Compute last minus first with sampling uncertainty
max_duration = duration + 64e-3
min_duration = duration - 64e-3
obs_tframe_range = [min_duration / total_nframes, max_duration / total_nframes]
# TFRAME range with uncertainty
print(f' TFRAME with sampling uncertainty (s)')
print(f'{obs_tframe_range[0]:.6f} -- {obs_tframe_range[1]:.6f}')


# Linear regression
def line(x, m, b):
    return m*x + b


# Time vs frame_idx
popt, pcov = curve_fit(
    line, frame_idx, seconds - seconds[0],
    sigma=32e-3, absolute_sigma=True
)
m, b = popt
m_err = np.sqrt(pcov[0, 0])
b_err = np.sqrt(pcov[1, 1])
print('\nLinear regression (time vs frame)')
print(f'   slope = {m:.6f} ± {m_err:.6f}')
print(f'   intercept = {b:.6f} ± {b_err:.6f}')
# plot
fig, ax = plt.subplots()
ax.plot(frame_idx, seconds - seconds[0], 'kx', alpha=0.25)
# labels and stuff
ax.set_xlabel('Frame index')
ax.set_ylabel('Time since end of first record (s)')
plt.show()


# Diff time vs frame_idx
popt, pcov = curve_fit(
    line, frame_idx[1:], diff_seconds/dframe_idx,
    sigma=16e-3, absolute_sigma=True
)
m, b = popt
m_err = np.sqrt(pcov[0, 0])
b_err = np.sqrt(pcov[1, 1])
print('\nLinear regression (time difference vs frame)')
print(f'   slope = {m:.6f} ± {m_err:.6f}')
print(f'   intercept = {b:.6f} ± {b_err:.6f}')
# plot
fig, ax = plt.subplots()
ax.plot(frame_idx[1:], diff_seconds/dframe_idx, 'kx', alpha=0.25)
# labels and stuff
ax.set_xlabel('Frame index')
ax.set_ylabel('Time difference with previous record (s)')
plt.show()
