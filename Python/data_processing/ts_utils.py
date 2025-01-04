import datetime as dt

'''
Time-series utility functions for monthly/quarterly data
'''

def get_quarter(date):
    if(date.month <= 3):
        return dt.datetime(date.year, 1, 1)
    elif((date.month >= 4) & (date.month <= 6)):
        return dt.datetime(date.year, 4, 1)
    elif((date.month >= 7) & (date.month <= 9)):
        return dt.datetime(date.year, 7, 1)
    elif((date.month >= 10) & (date.month <= 12)):
        return dt.datetime(date.year, 10, 1)
    
def get_quarters(dates):
    return [get_quarter(date) for date in dates]

def get_last_quarter(date):
    if(date.month <= 3):
        return dt.datetime(date.year-1, 10, 1)
    elif((date.month >= 4) & (date.month <= 6)):
        return dt.datetime(date.year, 1, 1)
    elif((date.month >= 7) & (date.month <= 9)):
        return dt.datetime(date.year, 4, 1)
    elif((date.month >= 10) & (date.month <= 12)):
        return dt.datetime(date.year, 7, 1)
    
def get_month_first_date(date):
    return dt.datetime(date.year, date.month, 1)
    
def get_month_first_dates(dates):
    return [get_month_first_date(date) for date in dates]
    
def get_prev_months_first_date(month_first_date, num_prev=1):
    month_first_dates = []
    cur_month_first_date = month_first_date
    for i in range(num_prev):
        if(cur_month_first_date.month == 1):
            prev_month_first_date = dt.datetime(cur_month_first_date.year-1, 12, 1)
        else:
            prev_month_first_date = dt.datetime(cur_month_first_date.year, cur_month_first_date.month-1, 1)    
        month_first_dates = month_first_dates + [prev_month_first_date]
        cur_month_first_date = prev_month_first_date
    return month_first_dates

def get_next_month_first_date(month_first_date):
    if month_first_date.month == 12:
        return dt.datetime(month_first_date.year + 1, 1, 1)
    else:
        return dt.datetime(month_first_date.year, month_first_date.month+1, 1)

 
    