import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import ccf
from statsmodels.tsa.filters.hp_filter import hpfilter

pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

import matplotlib.pyplot as plt
import seaborn as sns

from tqdm.notebook import tqdm

# Rename FRED-MD features
features_to_names = {
    "GDPC1": "gdp",
    "PCNDx": "non_durable_consumption",
    "PCDGx": "durable_consumption",
    "GPDIC1": "investment",
    "HOABS": "all_bus_hours",
    "HOAMS": "all_manuf_hours",
    "HOANBS": "all_nonfarm_hours",
    "AWHMAN": "avg_hours", # Actually is it avg_manuf_hour, but who cares...
    "AWHNONAG": "avg_private_hours",
    "AHETPIx": "average_hourly_earnings",
    "CE16OV": "employment",
    "POPTHM": "population",
    "DFF": "interest_rate"
}


def mult_diff_logs(x):
    return np.log(x).diff(periods=1) * 1200


def no_trend_logs(x):
    log_x = np.log(1 + x)
    cycle, trend = hpfilter(log_x)
    return log_x - trend


def log(x):
    return np.log(x)


def id_trans(x):
    return x


feat_to_transform = {
    "gdp": no_trend_logs,
    "gdp_cap": no_trend_logs,
    "employment": no_trend_logs,
    "productivity": no_trend_logs,
    "non_durable_consumption": no_trend_logs,
    "durable_consumption": no_trend_logs,
    "consumption": no_trend_logs,
    "consumption_cap": no_trend_logs,
    "investment": no_trend_logs,
    "investment_cap": no_trend_logs,
    "all_hours": no_trend_logs,
    "avg_hours": no_trend_logs,
    "average_hourly_earnings": no_trend_logs,
    "interest_rate": id_trans
}



# ---------------------------------- Some macroeconomic notes __________________________________________________________
# - **Nonfarm payrolls** - All Employees: Total Nonfarm, commonly known as Total Nonfarm Payroll, is a measure of the number of U.S. workers in the economy that excludes proprietors, private household employees, unpaid volunteers, farm employees, and the unincorporated self-employed.
# - **PPI fin goods** - The Producer Price Index is a family of indexes that measures the average change over time in the selling prices received by domestic producers of goods and services. Finished goods are commodities that will not undergo further processing and are ready for sale to the final-demand user, either an individual consumer or business firm. PPIs measure price change from the perspective of the seller
# - **PCEPI** - Personal expentiture consumption price index. The PCE price index is known for capturing inflation (or deflation) across a wide range of consumer expenses and reflecting changes in consumer behavior
# - **Housing starts** - measure of new residential construction, and are considered a key economic indicator. A housing start is counted as soon as groundbreaking begins, and each unit in a multi-family housing project is treated as a separate housing start.
# - **Treasury yield** - Treasury yield is the effective annual interest rate that the U.S. government pays on one of its debt obligations, expressed as a percentage. Put another way, Treasury yield is the annual return investors can expect from holding a U.S. government security with a given maturity. **Note**: Yield is the annual net profit that an investor earns on an investment. The interest rate is the percentage charged by a lender for a loan. The yield on new investments in debt of any kind reflects interest rates at the time they are issued

#%%
