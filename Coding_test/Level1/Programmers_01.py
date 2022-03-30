# [크레인 인형뽑기 게임]


import numpy as np

def check(value):
    c = 0
    for i in range(len(value)-1):
        try:
            if  (value[i] == value[i+1]) | ((i>=1) &(value[i] == value[i-1])):
                c +=2
                del value[i:i+2]
                break
        except: pass
    return value, c

   
def solution(board, moves):
    b = np.array(board)
    answer = []
    for i in range(len(moves)):
        for j in range(len(b)):
            if b[j, moves[i]-1] == 0:
                pass
            else:
                answer.append(b[j, moves[i]-1])
                b[j, moves[i]-1] = 0
                break
    cnt = 0
    c=1
    while c!=0:
        answer, c = check(answer)
        cnt += c

    return cnt
