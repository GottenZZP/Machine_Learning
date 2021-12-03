n = int(input())
num = 0
cnt = 0
if n:
    l = input().split(' ')
    for i in l:
        flag = True
        flag1 = True
        temp = i
        for j in range(len(temp)):
            if j != 0 and temp[j] != '.' and temp[j].isdigit() == False:
                flag = False
                print("ERROR: " + temp + " is not a legal number")
                break
            if j != 0 and temp[j] == '.' and flag1 == False:
                flag = False
                print("ERROR: " + temp + " is not a legal number")
                break
            if j != 0 and temp[j] == '.' and ((len(temp) - j) >= 4):
                flag = False
                print("ERROR: " + temp + " is not a legal number")
                break
            if j != 0 and temp[j] == '.':
                flag1 = False
        if flag:
            if float(temp) >= 1000 or float(temp) <= -1000:
                print("ERROR: " + temp + " is not a legal number")
            else:
                num += float(temp)
                cnt += 1
    if cnt == 0:
        print("The average of 0 numbers is Undefined")
    elif cnt == 1:
        print(f"The average of 1 number is %.2f" % num, end='')
    else:
        print(f"The average of {cnt} numbers is %.2f" % (num / cnt), end='')
else:
    print("The average of 0 numbers is Undefined", end='')




