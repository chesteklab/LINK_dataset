import csv
import os
import re
import config as config
import datetime

noteLocation = os.path.join(config.notesdir)

noteList = os.listdir(noteLocation)

extractedDates = [re.split('MonkeyNNotes|\.txt', note)[1] for note in noteList]
extractedDates.sort()
extractedDates = [[datetime.datetime.strptime(date, '%Y%m%d').date().isoformat()] for date in extractedDates]

with open('firstpass_dates.csv', 'w', newline='') as f:
    write = csv.writer(f)
    write.writerow(['Date', 'Runs'])
    write.writerows(extractedDates)