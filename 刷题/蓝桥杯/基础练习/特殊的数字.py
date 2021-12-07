for i in range(100, 1000):
    s = str(i)
    sums = 0
    for t in s:
        sums += int(t)**3
    if sums == i:
        print(i)