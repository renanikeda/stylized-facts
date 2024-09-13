## Imports
import math
import itertools
import datetime as dt
import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.stats import skew, kurtosis
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.graphics.gofplots import qqplot
import yfinance as yf

## Definições de plot
color_dark_blue = 'darkslategrey'
color_dark_green = 'darkolivegreen'

colors = [color_dark_blue, color_dark_green]
colors_asset = {'IBOV': color_dark_blue, 'SPX': color_dark_green}

color_line = list(itertools.product(['-', ':', '--', ':' '-', ':'], colors))
today = dt.date.today().strftime('%Y-%m-%d')
start = '2015-01-01'

def get_none_index (l):
    try:
        return l.index(None)
    except:
        l.append(None)
        return len(l) - 1

## Definindo as séries
spx_ticker = yf.Ticker("^SPX")
ibov_ticker = yf.Ticker("^BVSP")
df_ibov = ibov_ticker.history(start=start, end=today)['Close']
df_ibov.index = pd.to_datetime(df_ibov.index.date, format='%Y-%m-%d')
df_spx = spx_ticker.history(start=start, end=today)['Close']
df_spx.index = pd.to_datetime(df_spx.index.date, format='%Y-%m-%d')
df_total = { 'IBOV': df_ibov, 'SPX': df_spx }

fig = [None]*10
axs = [None]*10

## Asset indexes
fig[0], axs[0] = plt.subplots(2)
for (index, asset) in enumerate(df_total):
    axs[0][index].set_title(asset)
    # axs[1].set_title("S&P500")

    df_total[asset].plot(color=color_line[index][1], linestyle=color_line[index][0], ax=axs[0][index], legend = True)
    # df_spx.plot(color=color_line[1][1], linestyle=color_line[1][0], ax=axs[1], legend = True)

    axs[0][index].legend(['Fechamento'])
    # axs[1].legend(['Fechamento'])

plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.3)

plt.xticks(rotation=45, ha='right')
plt.xlim([df_ibov.index[0], df_ibov.index[-1]]) 
plt.tight_layout()
# plt.show()

## Defining Returns
returns = { 'IBOV': {}, 'SPX': {} }
periods = ['diário', 'semanal', 'mensal']
for asset in df_total:
    for period in periods:
        returns[asset][period] = np.log(df_total[asset]).diff().dropna() if period == 'diário' else np.log(df_total[asset].resample('W').last()).diff().dropna() if period == 'semanal' else np.log(df_total[asset].resample('ME').last()).diff().dropna()

## Autocorrelation Returns
for (index_asset, asset) in enumerate(returns):
    total_index = 1 + index_asset
    fig[total_index] = plt.figure(figsize=(10, 5))
    for (index, period) in enumerate(returns[asset]):
        lags = 20
        alpha = 0.05
        ax = fig[total_index].add_subplot(int(f'13{index + 1}'))

        sm.graphics.tsa.plot_acf(returns[asset][period], lags=lags, ax=ax, alpha=alpha, color=colors_asset[asset])
        plt.xlabel('lags')
        plt.ylabel(f'AC {asset} Retorno {period.title()}')
        plt.suptitle(f'Autocorrelação {asset}')
        ax.set_title('')

plt.tight_layout()
# plt.show()

## Calculando a Curtose
for asset in df_total:
    for period in returns[asset]:
        print(f'Curtose Retorno {period.title()} {asset}', round(kurtosis(returns[asset][period], axis=0, bias=True, fisher=False), 2))
    print('\n')

## Calculando a Assimetria
for asset in df_total:
    for period in returns[asset]:
        print(f'Assimetria Retorno {period.title()} {asset}', round(skew(returns[asset][period], axis=0, bias=True), 2))
    print('\r')

## Agregação Gaussiana
for (index_asset, asset) in enumerate(returns):
    total_index = 3 + index_asset
    fig[total_index] = plt.figure(figsize=(10, 7))
    for (index_period, period) in enumerate(returns[asset]):
        ax = fig[total_index].add_subplot(int(f'23{index_period + 1}'))
        sns.histplot(returns[asset][period], bins=50, color=colors_asset[asset], stat='density', ax=ax)
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, np.mean(returns[asset][period]), np.std(returns[asset][period]))
        plt.plot(x, p, color='r', linestyle='dashed', linewidth=3, label='Gaussian')
        plt.xlabel(f'{asset} retorno {period}')
        plt.ylabel('Frequência')
        plt.legend()

        ax = fig[total_index].add_subplot(int(f'23{index_period + 1 + 3}'))
        qq = qqplot(returns[asset][period], line='s', ax=ax, markerfacecolor=colors_asset[asset], fit=True)
        plt.xlabel('Quantil Teórico')
        plt.ylabel('Quantil Amostral')
        plt.xlim([-4,4]);

        plt.tight_layout()

# plt.show()

## Intermitência
for asset in returns:
    total_index = get_none_index(fig)
    fig[total_index] = plt.figure(figsize=(10, 5))
    for (index_period, period) in enumerate(returns[asset]):
        # print(f'Variância {asset} dos retornos {period}: ', returns[asset][period].std())
        ax = fig[total_index].add_subplot(int(f'31{index_period + 1}'))
        returns[asset][period].plot(figsize=(10, 7), color=colors_asset[asset], linestyle='-', ax=ax)
        plt.legend(fontsize="10")
        plt.xticks(rotation=45, ha='right')
        plt.xlim([returns[asset][period].index[0], returns[asset][period].index[-1]]);
        plt.ylabel(f'{asset} retorno {period}')
        ax.legend(['Fechamento'])
    plt.tight_layout()

def standard_deviation(log_return, window=30, trading_periods=252, clean=True):

    result = log_return.rolling(window=window, center=False).std() * math.sqrt(
        trading_periods
    )

    if clean:
        return result.dropna()
    else:
        return result
    
for asset in returns:
    total_index = get_none_index(fig)
    fig[total_index] = plt.figure(figsize=(10, 5))
    for (index_period, period) in enumerate(returns[asset]):
        ax = fig[total_index].add_subplot(int(f'31{index_period + 1}'))
        std_returns = standard_deviation(returns[asset][period], window = 15)
        std_returns.plot(figsize=(10, 7), color=colors_asset[asset], linestyle='-', ax=ax)
        plt.legend(fontsize="10")
        plt.xticks(rotation=45, ha='right')
        plt.xlim([returns[asset][period].index[0], returns[asset][period].index[-1]]);
        plt.ylabel(f'{asset} volatilidade {period}')
        ax.legend(['Volatilidade'])
    plt.tight_layout()
#plt.show()

##Agrupamento de Volatilidade
total_index = get_none_index(fig)
fig[total_index] = plt.figure(figsize=(10, 5))
for (index_asset, asset) in enumerate(returns):
    for (index_period, period) in enumerate(returns[asset]):
        ax = fig[total_index].add_subplot(int(f'23{index_period+1 + (index_asset*3)}'))
        sm.graphics.tsa.plot_acf(returns[asset][period]**2, lags=40, ax=ax, alpha=0.05, color=colors_asset[asset])
        plt.xlabel('lags')
        plt.ylabel(f'AC {asset} Retorno quadrado {period.title()}')
        plt.ylim([-0.25, 1.1])
        plt.title(f'Autocorrelação {asset}')
    plt.tight_layout()
#plt.show()

## Decaimento por lei de potencias do retorno absoluto
power_law = lambda x, a, b: a / np.power(x, b)
def fit_power_law(log_returns, lags = 40):
    init_params = [0.5, 0.5]
    acf = sm.tsa.stattools.acf(np.abs(log_returns), nlags=lags)
    popt, _ = curve_fit(power_law, np.arange(1, lags+1), acf[1:], p0=init_params, maxfev=5000)
    return popt

total_index = get_none_index(fig)
fig[total_index] = plt.figure(figsize=(10, 5))
lags = 30
for (index_asset, asset) in enumerate(returns):
    for (index_period, period) in enumerate(returns[asset]):
        ax = fig[total_index].add_subplot(int(f'23{index_period+1 + (index_asset*3)}'))
        sm.graphics.tsa.plot_acf(abs(returns[asset][period]), lags=lags, ax=ax, alpha=0.05, color=colors_asset[asset])
        popt = fit_power_law(abs(returns[asset][period]), lags = lags)
        print(f'Parametros Lei de Potências {asset} {period}: ', popt)

        lags_range = np.arange(1, lags+1)
        ax.plot(lags_range, power_law(lags_range, *popt), 'r-')
        plt.xlabel('lags')
        plt.ylabel(f'AC {asset} Retorno quadrado {period.title()}')
        plt.ylim([-0.25, 1.01])
        plt.title(f'Autocorrelação {asset}')
    plt.tight_layout()
plt.show()
