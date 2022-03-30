# 모의고사

def solution(answers):
    n1 = [1,2,3,4,5]*(10000//len([1,2,3,4,5]))
    n2 = [2,1,2,3,2,4,2,5]*(10000//len([2,1,2,3,2,4,2,5]))
    n3 = [3,3,1,1,2,2,4,4,5,5]*(10000//len([3,3,1,1,2,2,4,4,5,5]))

    a1, a2, a3 = 0, 0, 0
    for i in range(len(answers)):
        if answers[i] == n1[i]:
            a1 += 1
        if answers[i] == n2[i]:
            a2 += 1
        if answers[i] == n3[i]:
            a3 += 1

    dic = {1:a1, 2:a2, 3:a3}

    answer = []
    max_score = max(dic.values())
    for key, value in dic.items():
        if value == max_score:
            answer.append(key)

    answer = sorted(answer)   

    return answer
