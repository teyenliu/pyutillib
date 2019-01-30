#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os

import pandas as pd
import numpy as np
import time

COLUMNS=['start', 'cost', 'end']

def do_analysis(log_file):
    #Suppressing scientific notation in pandas
    pd.options.display.float_format = '{:20,.10f}'.format

    with open(log_file) as f:
        content = f.readlines()

    # remove characters like '[', ']', whitespace and '\n' at the end of each line
    data = [float(x.strip().split(":")[1].strip("[").strip("]")) for x in content]

    # convert python list to numpy
    data = np.asarray(data, dtype=np.longfloat)
    # reshape to 2D np array
    data = np.reshape(data, (-1, len(COLUMNS)))
    # convert np array to pandas
    pd_data = pd.DataFrame(data=data, columns=COLUMNS)

    # calculate and add new column: idle ( the idle time between 2 steps )
    pd_data['idle'] = pd.Series(0.0, dtype=np.longfloat, index=pd_data.index)
    pd_data['step'] = pd.Series(0.0, dtype=np.longfloat, index=pd_data.index)
    idle_idx = pd_data.columns.get_loc('idle')
    step_idx = pd_data.columns.get_loc('step')
    for i in range(1, len(pd_data.index)):
        value1 = (pd_data.iloc[i]['start'] - pd_data.iloc[i-1]['end'])
        value2 = (pd_data.iloc[i]['start'] - pd_data.iloc[i-1]['start'])
        pd_data.set_value(i, "idle", value1)
        pd_data.set_value(i, "step", value2)
 
    pd_data['start_to_end'] = pd_data['end'] - pd_data['start']
    pd_data['start_to_cost'] = pd_data['cost'] - pd_data['start']
    pd_data['cost_to_end'] = pd_data['end'] - pd_data['cost']

    # get the average of each column
    print(pd_data.iloc[:][1:-1].mean(axis=0))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', help='the log file which you want to do analysis.')
    args = parser.parse_args()

    do_analysis(args.file)


""" Reference complete example of numpy to pandas
pd.DataFrame(data=data[1:,1:],        # values
...              index=data[1:,0],    # 1st column as index
...              columns=data[0,1:])  # 1st row as the column names
"""
