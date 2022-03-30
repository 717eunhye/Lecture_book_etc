# 3진법 뒤집기

def solution(n):
    answer=""
    T="0123456789ABCDEF"
    q, r = divmod(n, 3)
    answer += str(r)

    while q !=0 :
        if q ==0:
            answer = T[r]
        else:
            q, r = divmod(q, 3)
            answer += str(r)

    return int(answer, 3)
