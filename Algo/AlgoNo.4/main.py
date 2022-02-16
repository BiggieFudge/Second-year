import random
import numpy as np

#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXLas Vegas StyleXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
def findNaive(arr,n,value):
    count = 0
    while (1):
        count += 1
        pos = random.randint(0, n-1)
        if arr[pos] == value:
            return count
    return -1

#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXMote CarloXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
def findMoteCarlo(arr,n,value):
    index = list(range(len(arr)))
    for i in range(n-1,0,-1):
        pos = random.randint(0,i)
        if (arr[index[pos]] == value):
            return pos
        index.remove(index[pos])
    return -1




def checkmatrice(matrice1,matrice2,matrice3):
    y = [[random.randint(0,10),random.randint(0,10)],random.randint(0,10)]
    y=np.array([[1],[1],[1]])
    c= np.dot(matrice3,y)
    print(c)
    ab = np.dot(matrice2,y)
    newab = np.dot(matrice1,ab)

    if newab == c:
        return True
    return False


# Python program for KMP Algorithm
def KMPSearch(pat, txt):
    M = len(pat)
    N = len(txt)

    # create lps[] that will hold the longest prefix suffix
    # values for pattern
    lps = [0] * M
    j = 0  # index for pat[]

    # Preprocess the pattern (calculate lps[] array)
    computeLPSArray(pat, M, lps)

    i = 0  # index for txt[]
    while i < N:
        if pat[j] == txt[i]:
            i += 1
            j += 1

        if j == M:
            print("Found pattern at index " + str(i - j))
            j = lps[j - 1]

        # mismatch after j matches
        elif i < N and pat[j] != txt[i]:
            # Do not match lps[0..lps[j-1]] characters,
            # they will match anyway
            if j != 0:
                j = lps[j - 1]
            else:
                i += 1
    print("bb")


def computeLPSArray(pat, M, lps):
    len = 0  # length of the previous longest prefix suffix

    lps[0]  # lps[0] is always 0
    i = 1

    # the loop calculates lps[i] for i = 1 to M-1
    while i < M:
        if pat[i] == pat[len]:
            len += 1
            lps[i] = len
            i += 1
        else:
            # This is tricky. Consider the example.
            # AAACAAAA and i = 7. The idea is similar
            # to search step.
            if len != 0:
                len = lps[len - 1]

                # Also, note that we do not increment i here
            else:
                lps[i] = 0
                i += 1


txt = "ABABDABACDABABCABAB"
pat = "ABABCABAB"
KMPSearch(pat, txt)

m1=np.array([[1.2,3],[1,2,3],[1,2,3]])
m2=np.array([[1.2,3],[1,2,3],[0,0,0]])
m3= np.array([[3,6,99],[3,6,99],[3,6,99]])

print("Question1"+str(findMoteCarlo([0,1,2,3], 4,3)))
print("Question2 "+ checkmatrice(m1,m2,m3))