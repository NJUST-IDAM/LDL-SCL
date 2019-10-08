if __name__ =="__main__":
    n, m = map(int, input().split())
    add = input().split()
    add = [int(x) for x in add]
    get = input().split()
    get = [int(x) for x in get]

    k = 1
    res = []
    while k<=m:
        temp = add[:get[k-1]]
        temp.sort()
        print(temp)
        res.append(temp[k-1])
        k = k+1

    print(res)