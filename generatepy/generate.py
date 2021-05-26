import json
import pandas as pd
import numpy as np

example_date = { 
    'date' : '2021-01-01 10:00:00',
    'lat' : 1000,
    'lon' : 2000,
    'is_checked' : True,
    'name' : 'name123',
    '1_date' : '2021-01-01 10:00:00',
    '1_lat' : 1000,
    '1_lon' : 2000,
    '1_is_checked' : True,
    '1_name' : 'name123',
    '2_date' : '2021-01-01 10:00:00',
    '2_lat' : 1000,
    '2_lon' : 2000,
    '2_is_checked' : True,
    '2_name' : 'name123',
    '3_date' : '2021-01-01 10:00:00',
    '3_lat' : 1000,
    '3_lon' : 2000,
    '3_is_checked' : True,
    '3_name' : 'name123',
}


def generate_sample(P):
    dates = pd.date_range(start='2010-01-01', end='2021-01-01', periods=P).round("T").strftime("%Y-%m-%d %H:%M:%S")

    example_df = { 
        'date' : dates,
        'lat' : np.random.randint(0, 9000, P),
        'lon' : np.random.randint(0, 9000, P),
        'is_checked' : np.random.choice([True, False], P),
        'name' : pd.Series(np.random.randint(0, 999, P)).map(lambda x: 'name'+str(x)),
        '1_date' : dates,
        '1_lat' : np.random.randint(0, 9000, P),
        '1_lon' : np.random.randint(0, 9000, P),
        '1_is_checked' : np.random.choice([True, False], P),
        '1_name' : pd.Series(np.random.randint(0, 999, P)).map(lambda x: 'name'+str(x)),
        '2_date' : dates,
        '2_lat' : np.random.randint(0, 9000, P),
        '2_lon' : np.random.randint(0, 9000, P),
        '2_is_checked' : np.random.choice([True, False], P),
        '2_name' : pd.Series(np.random.randint(0, 999, P)).map(lambda x: 'name'+str(x)),
        '3_date' : dates,
        '3_lat' : np.random.randint(0, 9000, P),
        '3_lon' : np.random.randint(0, 9000, P),
        '3_is_checked' : np.random.choice([True, False], P),
        '3_name' : pd.Series(np.random.randint(0, 999, P)).map(lambda x: 'name'+str(x)),
    }

    df = pd.DataFrame(example_df)
    df.to_json(f"sample_{P}.json", orient='records', lines=True, date_format='iso')


sizes = [] 
sizes.extend(range(10, 100, 10))
sizes.extend(range(100, 1000, 100))
sizes.extend(range(1000, 10000, 1000))
sizes.extend(range(10000, 100000, 10000))
sizes.extend(range(100000, 1000000, 100000))


for i in sizes:
    generate_sample(i)
