import requests
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
from mpl_finance import candlestick_ohlc
import matplotlib
import pylab
from pandas_datareader import data as pdr

matplotlib.rcParams.update({'font.size': 9})


# RSI is a momentum oscillator that measures the speed and change of price movements. RSI oscillates between zero and 100.
# Traditionally, RSI is considered overbought when above 70 and oversold when below 30.

def rsiFunc(prices, n=14):
    deltas = np.diff(prices)
    seed = deltas[:n+1]
    up = seed[seed>=0].sum()/n
    down = -seed[seed<0].sum()/n
    rs = up/down
    rsi = np.zeros_like(prices)
    rsi[:n] = 100. - 100./(1.+rs)

    for i in range(n, len(prices)):
        delta = deltas[i-1] # cause the diff is 1 shorter

        if delta>0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta

        up = (up*(n-1) + upval)/n
        down = (down*(n-1) + downval)/n

        rs = up/down
        rsi[i] = 100. - 100./(1.+rs)

    return rsi

def movingaverage(values,window):
    weigths = np.repeat(1.0, window)/window
    smas = np.convolve(values, weigths, 'valid')
    return smas # as a numpy array

def ExpMovingAverage(values, window):
    weights = np.exp(np.linspace(-1., 0., window))
    weights /= weights.sum()
    a =  np.convolve(values, weights, mode='full')[:len(values)]
    a[:window] = a[window]
    return a


def computeMACD(x, slow=26, fast=12):
    """
    compute the MACD (Moving Average Convergence/Divergence) using a fast and slow exponential moving avg'
    return value is emaslow, emafast, macd which are len(x) arrays
    """
    emaslow = ExpMovingAverage(x, slow)
    emafast = ExpMovingAverage(x, fast)
    return emaslow, emafast, emafast - emaslow

def graphData(name, start, end):
    #Moving Averages
    MA1 = 10
    MA2 = 30
    # GETS STOCK data
    stock = pdr.DataReader(name, 'yahoo', start, end) 
    # date, closep, highp, lowp, openp, volume = np.loadtxt(stockFile,delimiter=',', unpack=True,
    #                                                       converters={ 0: mdates.strpdate2num('%Y%m%d')})
    stock = stock.reset_index(level=0)
    if ('Adj. Close' not in stock.columns):
            stock['Adj. Close'] = stock['Close']
            stock['Adj. Open'] = stock['Open']

    date = stock['Date']
    openp = stock['Adj. Open']
    closep = stock['Adj. Close']
    highp = stock['High']
    lowp = stock['Low']
    volume = stock['Volume']
    rsi = rsiFunc(closep)

    # # if rsi[-1] < 30:
    x = 0
    y = len(date)
    newAr = []
    while x < y:
        appendLine = mdates.date2num(date[x]),openp[x],highp[x],lowp[x], closep[x]
        newAr.append(appendLine)
        x+=1

    rsiCol = 'purple'
    posCol = 'green'
    negCol = 'red'
    gridCol = 'black'
    fillcolor = '#4ee6fd'
    facecolor1 = '#F5F5F5'
    facecolor2 = 'darkgray'

    fig = plt.figure(facecolor="white")
    ax1 = plt.subplot2grid((6,4), (1,0), rowspan=4, colspan=4, facecolor=facecolor1)
    SP = 0 #Make sure this is 0 when MA2-1 > len(date)
    candlestick_ohlc(ax1, newAr[-SP:], width=.6, colorup=posCol, colordown=negCol)

    if (len(date) >= 2*MA2):
        SP2 = len(date[MA2-1:])
        Av1 = movingaverage(closep, MA1)
        Av2 = movingaverage(closep, MA2)

        Label1 = str(MA1)+' Day Average'
        Label2 = str(MA2)+' Day Average'

        ax1.plot(date[-SP2:],Av1[-SP2:],'cyan',label=Label1, linewidth=1.5)
        ax1.plot(date[-SP2:],Av2[-SP2:],'blue',label=Label2, linewidth=1.5)


    ax1.grid(True, color=gridCol)
    ax1.xaxis.set_major_locator(mticker.MaxNLocator(10))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax1.yaxis.label.set_color(gridCol)
    ax1.spines['bottom'].set_color(gridCol)
    ax1.spines['top'].set_color(gridCol)
    ax1.spines['left'].set_color(gridCol)
    ax1.spines['right'].set_color(gridCol)
    ax1.tick_params(axis='y', colors='w')
    plt.gca().yaxis.set_major_locator(mticker.MaxNLocator(prune='upper'))
    ax1.tick_params(axis='x', colors='w')
    plt.ylabel('Stock price and Volume')

    maLeg = plt.legend(loc=9, ncol=2, prop={'size':7},
               fancybox=True, borderaxespad=0.)
    maLeg.get_frame().set_alpha(0.4)
    textEd = pylab.gca().get_legend().get_texts()
    pylab.setp(textEd[0:5], color = 'orange')

    volumeMin = 0

    ax0 = plt.subplot2grid((6,4), (0,0), sharex=ax1, rowspan=1, colspan=4, facecolor=facecolor2)
    #rsi = rsiFunc(closep)

    ax0.plot(date[-SP:], rsi[-SP:], rsiCol, linewidth=1.5)
    ax0.axhline(70, color=negCol)
    ax0.axhline(30, color=posCol)
    ax0.fill_between(date[-SP:].dt.to_pydatetime(), rsi[-SP:], 70, where=(rsi[-SP:]>=70), facecolor=negCol, edgecolor=negCol, alpha=0.5)
    ax0.fill_between(date[-SP:].dt.to_pydatetime(), rsi[-SP:], 30, where=(rsi[-SP:]<=30), facecolor=posCol, edgecolor=posCol, alpha=0.5)
    ax0.set_yticks([30,70])
    ax0.yaxis.label.set_color(gridCol)
    ax0.spines['bottom'].set_color(gridCol)
    ax0.spines['top'].set_color(gridCol)
    ax0.spines['left'].set_color(gridCol)
    ax0.spines['right'].set_color(gridCol)
    ax0.tick_params(axis='y', colors=gridCol)
    ax0.tick_params(axis='x', colors=gridCol)
    plt.ylabel('RSI')

    ax1v = ax1.twinx()
    ax1v.fill_between(date[-SP:].dt.to_pydatetime(),volumeMin, volume[-SP:], facecolor=facecolor1, alpha=.4)
    ax1v.axes.yaxis.set_ticklabels([])
    ax1v.grid(False)
    ###Edit this to 3, so it's a bit larger
    ax1v.set_ylim(0, 3*volume.max())
    ax1v.spines['bottom'].set_color(gridCol)
    ax1v.spines['top'].set_color(gridCol)
    ax1v.spines['left'].set_color(gridCol)
    ax1v.spines['right'].set_color(gridCol)
    ax1v.tick_params(axis='x', colors=gridCol)
    ax1v.tick_params(axis='y', colors=gridCol)
    ax2 = plt.subplot2grid((6,4), (5,0), sharex=ax1, rowspan=1, colspan=4, facecolor=facecolor2)
    nslow = 26
    nfast = 12
    nema = 9
    emaslow, emafast, macd = computeMACD(closep)
    ema9 = ExpMovingAverage(macd, nema)
    ax2.plot(date[-SP:], macd[-SP:], color='lightblue', lw=2)
    ax2.plot(date[-SP:], ema9[-SP:], color='darkblue', lw=1)
    ax2.fill_between(date[-SP:].dt.to_pydatetime(), macd[-SP:]-ema9[-SP:], 0, alpha=0.5, facecolor=fillcolor, edgecolor=fillcolor)

    plt.gca().yaxis.set_major_locator(mticker.MaxNLocator(prune='upper'))
    ax2.spines['bottom'].set_color(gridCol)
    ax2.spines['top'].set_color(gridCol)
    ax2.spines['left'].set_color(gridCol)
    ax2.spines['right'].set_color(gridCol)
    ax2.tick_params(axis='x', colors=gridCol)
    ax2.tick_params(axis='y', colors=gridCol)
    plt.ylabel('MACD', color=gridCol)
    ax2.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5, prune='upper'))
    for label in ax2.xaxis.get_ticklabels():
        label.set_rotation(45)

    plt.suptitle(name.upper(),color=gridCol)

    plt.setp(ax0.get_xticklabels(), visible=False)
    plt.setp(ax1.get_xticklabels(), visible=False)

    # ax1.annotate('Look Here!',(date[100],Av1[100]),
    #     xytext=(0.8, 0.9), textcoords='axes fraction',
    #     arrowprops=dict(facecolor='white', shrink=0.05),
    #     fontsize=14, color = 'w',
    #     horizontalalignment='right', verticalalignment='bottom')

    plt.subplots_adjust(left=.09, bottom=.14, right=.94, top=.95, wspace=.20, hspace=0)
    plt.show()
    fig.savefig('example.png',facecolor=fig.get_facecolor())


if __name__ == "__main__":
    stocks = ['CRON', 'TLRY', 'CGC', 'ACBFF', 'CELG', 'AMD']
    start="2015-01-01"
    # start = "2018-7-20"
    end="2018-9-20"
    # for stock in stocks:
    #     graphData(stock, start, end)
    graphData(stocks[5], start, end)