if __name__=="main":
    # 人数n, 第s个开始报数, 数到第m的人退出
    l = input()
    n, s, m = map(int, input().split())
    human, i=range(n), 0
    temp=human[:s]
    human[:s]=[]
    human = human+temp.reverse()
    while len(human)>1:
        k = human.pop(0)
        i+=1
        if i%m!=0:
            human.append(k)
    print(human[0]+1)
