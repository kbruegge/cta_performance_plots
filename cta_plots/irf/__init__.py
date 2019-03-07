import numpy as np

def _bin_center(bin_edges):
    center = np.sign(bin_edges[:-1]) * np.sqrt(bin_edges[:-1] * bin_edges[1:])
    return center

def _log_tick_formatter(val, pos=None):
    if float(val).is_integer():
        return f'$10^{{{int(val)}}}$'

def _log_scale_formatter(val, pos=None):
    pos = np.round((10**val), decimals=1)
    if float(pos).is_integer():
        pos = pos.astype(np.int)
    return pos

def _log_data(data):
    m = (data == 0)
    data[m] = np.nan
    log_data =  np.log10(data)
    m = ~np.isfinite(log_data)
    log_data[m] = log_data[~m].min()
    return log_data