from unittest import skip


def que1(T,P):
    flag =0
    arr=[0] * len(T)
    brr=[0] * len(P)
    for i in range(len(T)):
        flag = 0
        for j in range(len(P)):
            if brr[j]==0 and T[i] == P[j]:
                flag = 1
                arr[i] = 1
                brr[j] = 1
        if flag ==0:
            for j in range(len(P)):
                brr[j]= 0
        elif 0 not in brr:
            return 1

    return 0


def LPM(Text):
    j=0
    i=1
    arr =[0] * len(Text)
    while i<len(Text):
        if Text[j] == Text[i]:
            arr[i]= j+1
            i+=1
            j+=1
        else:
            if j==0:
                arr[i]=0
                i+=1
            else:
                j= arr[j-1]
    return arr



def que3(text,pattern):
    if(len(text)!=len(pattern)):
        return False

    textToSearch = text + text

    if pattern in textToSearch:
        return True
    return False



def check_rotation(s, goal):
    if (len(s) != len(goal)):
        skip

    q1 = []
    for i in range(len(s)):
        q1.append(s[i])

    q2 = []
    for i in range(len(goal)):
        q2.append(goal[i])

    k = len(goal)
    while (k > 0):
        ch = q2[0]
        q2.pop(0)
        #q2.insert(0, ch)
        q2.append(ch)
        if (q2 == q1):
            return True

        k -= 1

    return False


def que4(text):


    checkText=""
    ySize = int((len(text)-10)/2)
    newText= text[:ySize] + "$" +text[len(text)-ySize:]
    arr=LPM(newText)
    j = arr[int(len(arr)/2)]

    return arr[len(arr)-1]

def que5(T):

    return LPM(checkPalindrom(T))[-1]


def checkPalindrom(text):
     flip = text[::-1]
     return text+"$"+flip



#aabaaaaabbcbbaaaaabaa
print(LPM("ABABBAABCA"))


