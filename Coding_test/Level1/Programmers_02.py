# 두 개 뽑아서 더하기

def solution(numbers):
    answer = []
    for i in range(len(numbers)-1):
        for ii in numbers[i+1:]:
            answer.append(int(numbers[i])+int(ii))
    answer = list(set(answer))
    answer.sort()
            
    return answer
