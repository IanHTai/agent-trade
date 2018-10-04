import urllib.request
import re
import datetime
from collections import OrderedDict
from config import ALPHA_VANTAGE_KEY

import pytz
import time
import json
import os

class DataAggregator:
    BASE_URL = 'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=SPY&interval=1min&apikey='

    def __init__(self, live=False):
        self.__live = live
        self.lastUpdate = None
        self.timescale = '1min'
        self.url = self.BASE_URL + ALPHA_VANTAGE_KEY
        self.timezone_string = 'US/Eastern'



    def getRecent(self):
        contents = urllib.request.urlopen(self.url).read()
        contents = json.loads(contents)
        if self.lastUpdate == contents['Meta Data']['3. Last Refreshed']:
            raise MarketNotOpenException()
        else:
            self.lastUpdate = contents['Meta Data']['3. Last Refreshed']
            datapoint = list(OrderedDict(contents['Time Series (1min)']).items())[0]
            date = datapoint[0]
            open = float(datapoint[1]['1. open'])
            high = datapoint[1]['2. high']
            low = datapoint[1]['3. low']
            close = datapoint[1]['4. close']
            volume = datapoint[1]['5. volume']

            return date, open, high, low, close, volume

    def checkMarket(self):
        contents = urllib.request.urlopen(self.url).read()
        contents = json.loads(contents)
        if self.lastUpdate == contents['Meta Data']['3. Last Refreshed']:
            return False
        else:
            return True

    def collect(self):
        print("Collecting Data for SPY", DSTDateTime(timezone=self.timezone_string).datetime)
        while True:
            seconds = DSTDateTime(timezone=self.timezone_string).datetime.second
            if not self.checkMarket():
                sleepTime = (seconds - DSTDateTime(timezone=self.timezone_string).datetime.second) % 60
                print("Sleeping until", seconds)
                time.sleep(sleepTime)
                continue
            minute = DSTDateTime(timezone=self.timezone_string).datetime.minute
            now = DSTDateTime(timezone=self.timezone_string).datetime
            while True:
                try:
                    _date, _open, _high, _low, _close, _volume = self.getRecent()
                    break
                except MarketNotOpenException as err:
                    print("Market not open but tried to access data anyway", DSTDateTime(timezone=self.timezone_string).datetime)
                except Exception as err:
                    print(err)

            filename = 'resources\\' + str(now.year) + str(now.month) + str(now.day) + '.csv'
            with open(filename, 'a+') as dayFile:
                if os.stat(filename).st_size == 0:
                    dayFile.write("Date,Open,High,Low,Close,Volume\n")
                writeString = str(_date) + ',' + str(_open) + ',' + str(_high) + ',' + str(_low) + ',' + str(_close) + ',' + str(_volume) + '\n'
                dayFile.write(writeString)
                print("Wrote data for", str(_date))
            if DSTDateTime(timezone=self.timezone_string).datetime.hour > 17 or DSTDateTime(timezone=self.timezone_string).datetime.hour < 8:
                # Market closed, don't have to refresh API call every minute
                # Sleep until 8 am next day, or 8 am next monday if weekend
                # The boundaries 8 am and 5 pm are chosen to mitigate chances of bugs regarding Daylight Savings Time
                # At the cost of 120 additional requests per work day
                print("Market's closed", DSTDateTime(timezone=self.timezone_string).datetime)
                sleepTime = ((60 - DSTDateTime(timezone=self.timezone_string).datetime.minute)%60) * 60
                time.sleep(sleepTime)
                print("Adjusted minute to 00", DSTDateTime(timezone=self.timezone_string).datetime)
                sleepTime = ((8 - DSTDateTime(timezone=self.timezone_string).datetime.hour) % 24) * 60 * 60
                time.sleep(sleepTime)
                print("Slept until 8 am EST", DSTDateTime(timezone=self.timezone_string).datetime)
                if DSTDateTime(timezone=self.timezone_string).datetime.weekday() > 4:
                    print("It's the weekend, time to go home", DSTDateTime(timezone=self.timezone_string).datetime)
                    sleepTime = (7 - DSTDateTime(timezone=self.timezone_string).datetime.weekday()) * 60 * 60 * 24
                    time.sleep(sleepTime)
                    print("Weekend's over, time to work again", DSTDateTime(timezone=self.timezone_string).datetime)
            elif minute > DSTDateTime(timezone=self.timezone_string).datetime.minute and seconds > DSTDateTime(timezone=self.timezone_string).datetime.second:
                # Lagging behind (took > 1 minute to get data/write), don't sleep
                continue
            else:
                if DSTDateTime(timezone=self.timezone_string).datetime.second > 30:
                    # If we are starting to lag behind and the second we get updates is > 30 past the minute, we reset the
                    # seconds past the minute to 15
                    sleepTime = (60 - DSTDateTime(timezone=self.timezone_string).datetime.second) % 60 + 15
                    time.sleep(sleepTime)
                else:
                    # Sleep until the pre-set seconds mark from previous iteration
                    sleepTime = (seconds - DSTDateTime(timezone=self.timezone_string).datetime.second) % 60
                    time.sleep(sleepTime)


class MarketNotOpenException(Exception):
    def __init__(self, message=None):
        self.message = message

class DSTDateTime:
    def __init__(self, timezone='US/Eastern'):
        self.timezone = pytz.timezone(timezone)
        self.datetime = datetime.datetime.now(tz=self.timezone)



if __name__ == '__main__':
    aggregator = DataAggregator()
    aggregator.collect()