from numpy import array
from numpy import hstack
from pandas import DataFrame, read_csv, concat, read_csv
import time

n_in = 5


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
        # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
        # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


df = read_csv("datapoints.csv", parse_dates=[0])

mask = df.name.str.contains("BLOK61")
videolab = df[~mask]

timestamps = DataFrame(videolab.pop("timestamp"))
values = DataFrame(videolab.pop("value"))
prev_values = series_to_supervised(values, n_in=n_in, n_out=0)


dataset = []

for i in range(len(timestamps)-n_in):
    tmp = []
    tmp.append(timestamps.iloc[i])
    tmp.append(prev_values.iloc[i])
    dataset.append(tmp)
    dataset.append(values.iloc[i])
