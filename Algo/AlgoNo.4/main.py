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



m1=np.array([[1.2,3],[1,2,3],[1,2,3]])
m2=np.array([[1.2,3],[1,2,3],[0,0,0]])
m3= np.array([[3,6,99],[3,6,99],[3,6,99]])

print("Question1"+str(findMoteCarlo([0,1,2,3], 4,3)))
print("Question2 "+ checkmatrice(m1,m2,m3))