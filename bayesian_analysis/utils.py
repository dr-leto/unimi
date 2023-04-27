import numpy as np
import pandas as pd

pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

import matplotlib.pyplot as plt
import seaborn as sns

from tqdm.notebook import tqdm

# Rename FRED-MD features
features_to_names = {
    "RPI": "real_income",
    "DPCERA3M086SBEA": "real_consumption",
    "INDPRO": "industrial_production_index",
    "CUMFNS": "capacity_utilization",
    "UNRATE": "uneployment_rate",
    "PAYEMS": "nonfarm_payrolls",
    "CES0600000007": "avg_weekly_hours",
    "CES0600000008": "avg_hourly_ernings",
    "WPSFD49207": "ppi_fin_goods",
    "PCEPI": "pcepi",
    "HOUST": "housing_starts",
    "S&P 500": "sp500",
    "EXUSUKx": "us_to_gbt_rate",
    "GS5": "5_year_treasury",
    "GS10": "10_year_treasury",
    "BAAFFM": "baa_corp_bond_rate"
}


def mult_diff_logs(x):
    return np.log(x).diff(periods=1) * 1200


def log(x):
    return np.log(x)


def id_trans(x):
    return x

feat_to_transform = {
    "RPI": mult_diff_logs,
    "DPCERA3M086SBEA": mult_diff_logs,
    "INDPRO": mult_diff_logs,
    "CUMFNS": id_trans,
    "UNRATE": id_trans,
    "PAYEMS": mult_diff_logs,
    "CES0600000007": id_trans,
    "CES0600000008": mult_diff_logs,
    "WPSFD49207": mult_diff_logs,
    "PCEPI": mult_diff_logs,
    "HOUST": log,
    "S&P 500": mult_diff_logs,
    "EXUSUKx": mult_diff_logs,
    "GS5": id_trans,
    "GS10": id_trans,
    "BAAFFM": id_trans
}



# ---------------------------------- Some macroeconomic notes __________________________________________________________
# - **Nonfarm payrolls** - All Employees: Total Nonfarm, commonly known as Total Nonfarm Payroll, is a measure of the number of U.S. workers in the economy that excludes proprietors, private household employees, unpaid volunteers, farm employees, and the unincorporated self-employed.
# - **PPI fin goods** - The Producer Price Index is a family of indexes that measures the average change over time in the selling prices received by domestic producers of goods and services. Finished goods are commodities that will not undergo further processing and are ready for sale to the final-demand user, either an individual consumer or business firm. PPIs measure price change from the perspective of the seller
# - **PCEPI** - Personal expentiture consumption price index. The PCE price index is known for capturing inflation (or deflation) across a wide range of consumer expenses and reflecting changes in consumer behavior
# - **Housing starts** - measure of new residential construction, and are considered a key economic indicator. A housing start is counted as soon as groundbreaking begins, and each unit in a multi-family housing project is treated as a separate housing start.
# - **Treasury yield** - Treasury yield is the effective annual interest rate that the U.S. government pays on one of its debt obligations, expressed as a percentage. Put another way, Treasury yield is the annual return investors can expect from holding a U.S. government security with a given maturity. **Note**: Yield is the annual net profit that an investor earns on an investment. The interest rate is the percentage charged by a lender for a loan. The yield on new investments in debt of any kind reflects interest rates at the time they are issued
