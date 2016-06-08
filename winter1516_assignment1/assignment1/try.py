a = [2, 5, 4, 8, 1, 4, 2, 5, 3]

i = 1
j = len(a) - 1
while i <= j:
    while j >= i and a[j] >= a[0]:
        j -= 1
    if j < i:
        break
    else:
        temp = a[j]
        a[j] = a[0]
        a[0] = temp
    while j >= i and a[i] <= a[0]:
        i += 1
    if j < i:
        break
    else:
        temp = a[i]
        a[i] = a[0]
        a[0] = temp
print 'OK'
print a