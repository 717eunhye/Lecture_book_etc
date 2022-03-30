# 정렬-k번째수

def solution(array, commands):
    answer = []
    for i in range(len(commands)):
        answer.append(sorted(array[commands[i][0]-1:commands[i][1]])[commands[i][2]-1])
    return answer
