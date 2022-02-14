# encoding: utf-8
# !/usr/bin/env python3
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from glob import glob
from datetime import datetime

from prefect import task, Flow, Parameter


@task
def extract_json(jfilepath):
    # JSONをチェック
    if not os.path.exists(jfilepath):
        js = {'len': 0, 'data': []}
        with open(jfilepath, 'w') as _f:
            json.dump(js, _f, indent=4)
    else:
        with open(jfilepath) as f:
            js = f.read()
        try:
            js = json.loads(js)
        except:
            decoder = json.JSONDecoder()
            js = decoder.raw_decode(js)[0]

    df = pd.DataFrame(js['data']).sort_values(by=['expid'])
    return df


@task
def transform(data, file):
    print(file)
    print(data.shape)
    print(data.mean().iloc[:7])


def main(outcome):
    with Flow("etl") as flow:
        # outcome = Parameter("outcome", default="MAX")
        files = glob('./data/result/*%s*.json' % outcome)
        data = []
        for file in files:
            _js = extract_json(file)
            data.append(_js)
            transform(_js, file)
    flow.run()


if __name__ == '__main__':
    main('MAX')
    main('MEAN')
    main('STD')
    print(0)
