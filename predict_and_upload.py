#imports
import datetime
import numpy as np
from pathlib import Path
import os
import time

import matplotlib
import matplotlib.pyplot as plt
    
from ocean_lib.example_config import ExampleConfig
from ocean_lib.ocean.ocean import Ocean
from ocean_lib.web3_internal.wallet import Wallet


#helper functions: setup
def create_ocean_instance() -> Ocean:
    config = ExampleConfig.get_config("https://polygon-rpc.com") # points to Polygon mainnet
    config["BLOCK_CONFIRMATIONS"] = 1 #faster
    ocean = Ocean(config)
    return ocean


def create_alice_wallet(ocean: Ocean) -> Wallet:
    config = ocean.config_dict
    alice_private_key = os.getenv('REMOTE_TEST_PRIVATE_KEY1')
    alice_wallet = Wallet(ocean.web3, alice_private_key, config["BLOCK_CONFIRMATIONS"], config["TRANSACTION_TIMEOUT"])
    bal = ocean.from_wei(alice_wallet.web3.eth.get_balance(alice_wallet.address))
    print(f"alice_wallet.address={alice_wallet.address}. bal={bal}")
    assert bal > 0, f"Alice needs MATIC"
    return alice_wallet


#helper functions: time
def to_unixtime(dt: datetime.datetime):
    return time.mktime(dt.timetuple())


def to_unixtimes(dts: list) -> list:
    return [to_unixtime(dt) for dt in dts]


def to_datetime(ut) -> datetime.datetime:
    return datetime.datetime.utcfromtimestamp(ut)


def to_datetimes(uts: list) -> list:
    return [to_datetime(ut) for ut in uts]


def print_datetime_info(descr:str, uts: list):
    dts = to_datetimes(uts)
    print(descr + ":")
    print(f"  first datetime: {dts[0].strftime('%Y/%m/%d, %H:%M:%S')}")
    print(f"  last datetime: {dts[-1].strftime('%Y/%m/%d, %H:%M:%S')}")
    print(f"  {len(dts)} datapoints")
    print(f"  time interval between datapoints: {(dts[1]-dts[0])}")


def target_24h_unixtimes(start_dt: datetime.datetime) -> list:
    target_dts = [start_dt + datetime.timedelta(hours=h) for h in range(24)]
    target_uts = to_unixtimes(target_dts)
    return target_uts


#helper-functions: higher level
def load_from_ohlc_data(file_name: str) -> tuple:
    """Returns (list_of_unixtimes, list_of_close_prices)"""
    with open(file_name, "r") as file:
        data_str = file.read().rstrip().replace('"', '')
    x = eval(data_str) #list of lists
    uts = [xi[0]/1000 for xi in x]
    vals = [xi[4] for xi in x]
    return (uts, vals)


def filter_to_target_uts(target_uts:list, unfiltered_uts:list, unfiltered_vals:list) -> list:
    """Return filtered_vals -- values at at the target timestamps"""
    filtered_vals = [None] * len(target_uts)
    for i, target_ut in enumerate(target_uts):
        time_diffs = np.abs(np.asarray(unfiltered_uts) - target_ut)
        tol_s = 1 #should always align within e.g. 1 second
        assert min(time_diffs) <= tol_s, min(time_diffs) 
        j = np.argmin(time_diffs)
        filtered_vals[i] = unfiltered_vals[j]
    return filtered_vals


#helpers: save/load list
def save_list(list_: list, file_name: str):
    """Save a file shaped: [1.2, 3.4, 5.6, ..]"""
    p = Path(file_name)
    p.write_text(str(list_))


def load_list(file_name: str) -> list:
    """Load from a file shaped: [1.2, 3.4, 5.6, ..]"""
    p = Path(file_name)
    s = p.read_text()
    list_ = eval(s)
    return list_


#helpers: prediction performance
def calc_nmse(y, yhat) -> float:
    assert len(y) == len(yhat)
    mse_xy = np.sum(np.square(np.asarray(y) - np.asarray(yhat)))
    mse_x = np.sum(np.square(np.asarray(y)))
    nmse = mse_xy / mse_x
    return nmse


def plot_prices(cex_vals, pred_vals):
    matplotlib.rcParams.update({'font.size': 22})
    
    x = [h for h in range(0,24)]
    assert len(x) == len(cex_vals) == len(pred_vals)
    
    fig, ax = plt.subplots()
    ax.plot(x, cex_vals, '--', label="CEX values")
    ax.plot(x, pred_vals, '-', label="Pred. values")
    ax.legend(loc='lower right')
    plt.ylabel("ETH price")
    plt.xlabel("Hour")
    fig.set_size_inches(18, 18)
    plt.xticks(x)
    plt.show()

#ocean = create_ocean_instance()

#alice_wallet = create_alice_wallet(ocean)

#Download file from ocean market

#ETH_USDT_did = "did:op:0dac5eb4965fb2b485181671adbf3a23b0133abf71d2775eda8043e8efc92d19"
#file_name = ocean.assets.download_file(ETH_USDT_did, alice_wallet)
file_name = 'USDT.json'
allcex_uts, allcex_vals = load_from_ohlc_data(file_name)
print_datetime_info("CEX data info", allcex_uts)

dts = to_datetimes(allcex_uts)

import pandas as pd

df = pd.DataFrame({"ds": dts, "y": allcex_vals})

df.y.plot()

plt.show()

train = df.iloc[0:-12, :]
test = df.iloc[-12:,:]

########################

#Predict Here and store in pred_vals

#######################

start_dt = datetime.datetime.now() - datetime.timedelta(hours=24) #must be >= 12h ago
start_dt = round_to_nearest_hour(start_dt) # so that times line up
target_uts = target_12h_unixtimes(start_dt)
print_datetime_info("target times", target_uts)

# get the actual ETH values at that time
import ccxt
allcex_x = ccxt.binance().fetch_ohlcv('ETH/USDT', '1h')
allcex_uts = [xi[0]/1000 for xi in allcex_x]
allcex_vals = [xi[4] for xi in allcex_x]
print_datetime_info("allcex times", allcex_uts)

cex_vals = filter_to_target_uts(target_uts, allcex_uts, allcex_vals)

# now, we have predicted and actual values. Let's find error, and plot!
nmse = calc_nmse(cex_vals, pred_vals)
print(f"NMSE = {nmse}")
plot_prices(cex_vals, pred_vals)







