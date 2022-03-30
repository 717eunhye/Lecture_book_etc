# 같은 숫자는 싫어 

def solution(arr):
    answer=['x']
    for i in range(len(arr)):
        if answer[-1] != arr[i]: 
            answer.append(arr[i])
    return answer[1:]
