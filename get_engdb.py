from jwst.lib.engdb_mast import EngdbMast
import os
import pandas as pd


def get_telem(mnemonics, visits, token=None):
    """

    :param mnemonics: a list of tuples (str, bool) where the string is the
     mnemonic's name and the boolean indicates whether or not all values are
     retrieved or only those that changed
    :param visits: a list of dictionaries {"start_time", "end_time"} where
     start_time and end_time are iso formatted strings
    :return:
    """

    # Connect to MAST API Eng DB
    if token is None:
        token = os.getenv("MAST_API_TOKEN")
    eng_db_conn = EngdbMast(base_url=None, token=token)

    values = []

    # Loop over mnemonics
    for mnemonic in mnemonics:
        # Loop over visits
        for visit in visits:
            eng_db_values = eng_db_conn.get_values(
                mnemonic[0], visit["start_time"], visit["end_time"],
                include_obstime=True
            )
            eu_values = []

            # parse each value returned by Eng DB into the eu value object
            for eng_db_value in eng_db_values:
                parsed_value = {
                    "time": eng_db_value.obstime.value,
                    "eu": mnemonic[0],
                    "value": eng_db_value.value
                }

                # if retrieve_all is False and we have at least one value:
                # compare to previous value and skip if it is equal
                if len(eu_values) != 0 and mnemonic[1] == False:
                    if eu_values[-1]["value"] == parsed_value["value"]:
                        continue

                eu_values.append(parsed_value)
            values = values + eu_values

        sorted_values = sorted(values, key=lambda d: d["time"])
        df = pd.DataFrame.from_dict(sorted_values)

        df["time"] = pd.to_datetime(df["time"])
        df["timedelta"] = df["time"].diff().dt.total_seconds()

    return df