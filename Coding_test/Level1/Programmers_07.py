# 2016년

def solution(a, b):
    import datetime
    t=['MON','TUE','WED','THU','FRI','SAT','SUN']
    answer = t[datetime.datetime(2016, a, b).weekday()]
    return answer
