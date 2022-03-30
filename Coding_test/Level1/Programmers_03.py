# 완주하지 못한 선수

from collections import Counter

def solution(participant, completion):
    dic1 = Counter(participant)
    dic2 = Counter(completion)

    for na, cnt in dic1.items():
        if dic1[na] == dic2[na]: #완주
            pass
        elif dic2[na] == 0: #완주못함
            answer = na
        elif (dic1[na]>1) & (dic2[na]!=dic1[na]):#동명이인
            answer = na
                    
    return answer
