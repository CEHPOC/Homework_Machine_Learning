
match=3
miss=-10
d=-1

a=input()
b=input()
m=len(a)
n=len(b)
mas=[0]*(m+1)
for i in range(m+1):
    mas[i]=[0]*(n+1)
mas[0][0]=0
arr=[" "]*(m+1)
for i in range(m+1):
    arr[i]=[" "]*(n+1)
arr[0][0]="0"
for i in range(1,m+1):
    mas[i][0]=mas[i-1][0]+d
    arr[i][0]="V"
for i in range(1,n+1):
    mas[0][i]=mas[0][i-1]+d
    arr[0][i] = "H"
for i in range(1,m+1):
    for j in range(1,n+1):
        if a[i-1] == b[j-1]:
            k=match
        else:
            k=miss
        mas[i][j]=max(mas[i-1][j]+d,mas[i][j-1]+d,mas[i-1][j-1]+k)
        if mas[i][j] == mas[i-1][j]+d :
            arr[i][j]="V"
        if mas[i][j] == mas[i][j-1]+d:
            arr[i][j] = "H"
        if mas[i][j] == mas[i-1][j-1]+k:
            arr[i][j] = "D"
'''
for i in range(m+1):
    s=""
    for j in range(n+1):
        s+=str(mas[i][j])+" "
    print(s)
for i in range(m+1):
    s=""
    for j in range(n+1):
        s+=str(arr[i][j])+" "
    print(s)
'''
i=m
j=n
sa=""
sb=""
while(i!=0 or j!=0):
    if arr[i][j] == "D":
        j = j - 1
        i = i - 1
        sa = a[i] + sa
        sb = b[j] + sb
    elif arr[i][j]=="H":
        sa="-"+sa
        j=j-1
        sb=b[j]+sb
    elif arr[i][j]=="V":
        sb="-"+sb
        i=i-1
        sa=a[i]+sa
print(sa)
print(sb)
