import pandas as pd
import numpy as np
import json
from functools import reduce

def clean_numeric(series):
    if series.dtype == 'object':
        return (
            series.astype(str)
            .str.replace(r'[%$ ]', '', regex=True)
            .str.replace(',', '.')
            .replace(['н/д', 'N/A', ''], np.nan)
        )
    return series


# Gold prices
gold = pd.read_csv('data/gold_prices.csv', sep=';')
gold['Date'] = pd.to_datetime(gold['Дата'], dayfirst=True, errors='coerce')
gold['Gold_Price'] = pd.to_numeric(clean_numeric(gold['Цена_USD']), errors='coerce')
gold = gold[['Date', 'Gold_Price']].dropna().drop_duplicates('Date').sort_values('Date')

gold['SMA_7_Gold'] = gold['Gold_Price'].rolling(window=7).mean()


# DXY index
dxy_raw = pd.read_excel('dxy_index.xlsx', sheet_name='Data', header=None)

records = []
for _, row in dxy_raw.iterrows():
    values = [str(v).strip() for v in row if pd.notna(v)]
    if len(values) < 2:
        continue
    try:
        if any(x in values[0] for x in ['/', '.']):
            date_val, price_val = values[0], values[1]
        else:
            date_val, price_val = values[1], values[0]

        date_val = pd.to_datetime(date_val, errors='coerce')
        price_val = float(price_val.replace(',', '.'))

        if pd.notna(date_val):
            records.append({'Date': date_val, 'DXY': price_val})
    except Exception:
        continue

dxy = pd.DataFrame(records).drop_duplicates('Date').sort_values('Date')


# FED rate
fed_rows = []

with open('data/fed_rate.json', 'r') as f:
    fed_json = json.load(f)

for obs in fed_json.get('observations', []):
    date_raw = obs.get('data', {}).get('effective_date')
    rate_raw = obs.get('data', {}).get('rate_upper')

    date = pd.to_datetime(date_raw, errors='coerce')
    rate = pd.to_numeric(rate_raw, errors='coerce')

    if pd.notna(date) and pd.notna(rate):
        fed_rows.append({
            'Date': date,
            'Fed_Rate': rate
        })

fed = (
    pd.DataFrame(fed_rows)
    .drop_duplicates('Date')
    .sort_values('Date')
)



# Inflation CPI
cpi = pd.read_excel('inflation_cpi.xlsx')
cpi['Date'] = pd.to_datetime(cpi['Period'], errors='coerce')
cpi['CPI'] = pd.to_numeric(clean_numeric(cpi['CPI_YoY_%']), errors='coerce')
cpi = cpi[['Date', 'CPI']].dropna()


# Oil Brent
oil_rows = []
with open('data/oil_brent.txt', encoding='utf-8', errors='ignore') as f:
    for line in f:
        if line.startswith('#') or not line.strip():
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            date = pd.to_datetime(" ".join(parts[:3]), errors='coerce')
            price = float(parts[-1].replace(',', '.'))
            if pd.notna(date):
                oil_rows.append({'Date': date, 'Brent': price})
        except Exception:
            continue

oil = pd.DataFrame(oil_rows).drop_duplicates('Date').sort_values('Date')


# Silver
silver = pd.read_csv('data/silver_prices.csv')
silver['Date'] = pd.to_datetime(silver['Date'], errors='coerce')
silver['Silver'] = pd.to_numeric(clean_numeric(silver['Close']), errors='coerce')
silver = silver[['Date', 'Silver']].dropna()


# S&P 500
sp500 = pd.read_csv('data/sp500.csv', sep=';', encoding='cp1251')

date_col = [c for c in sp500.columns if 'Дата' in c or 'Date' in c][0]
val_col = [c for c in sp500.columns if 'S&P' in c or 'Значение' in c][0]

sp500['Date'] = pd.to_datetime(sp500[date_col], dayfirst=True, errors='coerce')
sp500['SP500'] = (
    sp500[val_col].astype(str)
    .str.replace('.', '', regex=False)
    .str.replace(',', '.', regex=False)
)

sp500['SP500'] = pd.to_numeric(sp500['SP500'], errors='coerce')
sp500 = sp500[['Date', 'SP500']].dropna()


# Treasury 10Y
treasury = pd.read_csv('data/treasury_10y.csv')
treasury.columns = treasury.columns.str.strip()

d_col = [c for c in treasury.columns if 'Date' in c][0]
y_col = [c for c in treasury.columns if 'Yield' in c][0]

treasury['Date'] = pd.to_datetime(treasury[d_col], errors='coerce')
treasury['Treasury_10Y'] = pd.to_numeric(clean_numeric(treasury[y_col]), errors='coerce')
treasury = treasury[['Date', 'Treasury_10Y']].dropna()


# USD / CNY
usd_cny = pd.read_excel('usd_cny.xlsx', skiprows=2)
usd_cny.columns = usd_cny.columns.str.strip()

usd_cny['Date'] = (
    usd_cny.iloc[:, 0].astype(str)
    .str.replace('年', '-')
    .str.replace('月', '-')
    .str.replace('日', '')
)

usd_cny['Date'] = pd.to_datetime(usd_cny['Date'], errors='coerce')
usd_cny['USD_CNY'] = pd.to_numeric(clean_numeric(usd_cny.iloc[:, 1]), errors='coerce')
usd_cny.loc[usd_cny['USD_CNY'] > 20, 'USD_CNY'] /= 10
usd_cny = usd_cny[['Date', 'USD_CNY']].dropna()


# VIX
vix = pd.read_csv('data/vix_index.csv')
vix.columns = vix.columns.str.strip()

vix['Date'] = pd.to_datetime(vix.iloc[:, 0], errors='coerce')
vix['VIX'] = pd.to_numeric(clean_numeric(vix.iloc[:, 1]), errors='coerce')
vix = vix[['Date', 'VIX']].dropna()


# Merge datasets
datasets = [
    gold[['Date', 'Gold_Price', 'SMA_7_Gold']],
    dxy, fed, cpi, oil, silver, sp500, treasury, usd_cny, vix
]

df_final = reduce(lambda l, r: pd.merge(l, r, on='Date', how='outer'), datasets)

df_final = (
    df_final[df_final['Date'] >= '2024-01-01']
    .sort_values('Date')
    .drop_duplicates('Date')
    .ffill()
)

df_final.to_csv('final_dataset.csv', index=False, sep=';', encoding='utf-8-sig')


# Correlation
valid_corr = df_final.dropna(subset=['SMA_7_Gold', 'DXY'])
correlation = valid_corr['SMA_7_Gold'].corr(valid_corr['DXY'])

with open('correlation_result.txt', 'w') as f:
    f.write(f"{correlation:.4f}")

print(f"Pearson correlation (SMA-7 Gold vs DXY): {correlation:.4f}")
