n = int(input())
nums = []
for i in range(1, n + 1):
    if i == 1:
        print(1)
        nums.append(1)
    elif i == 2:
        print(1, 1)
        nums.append([1, 1])
    else:
        li = []
        for j in range(i):
            if j == 0:
                print(1, end=' ')
                li.append(1)
            elif j == i - 1:
                print(1)
                li.append(1)
            else:
                sum_ = nums[i - 2][j - 1] + nums[i - 2][j]
                print(sum_, end=' ')
                li.append(sum_)
        nums.append(li)