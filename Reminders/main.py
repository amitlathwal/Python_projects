import pandas as pd
import time
from plyer import notification
from datetime import date

todays_date = date.today()
file = pd.read_csv("list.csv")

while True:
    curr_time = time.localtime()

    for row in file.iterrows():
        dt = str(todays_date.day) + '-' + str(todays_date.month) + '-' + str(todays_date.year)

        if (time.strftime("%H:%M", curr_time) == row[1]['Time']) and (row[1]['Date'] == dt):
            notification.notify(
                title = row[1]['Title'],
                message = row[1]['Description'],
                app_icon = "Rem_icon.ico",
                timeout = 10
            )

    time.sleep(60)