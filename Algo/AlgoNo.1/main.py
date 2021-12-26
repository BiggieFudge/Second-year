# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def minmax(array,size):
    if array[0]> array[1]:
        max = array[0]
        min = array[1]
    else:
        min = array[0]
        max = array[1]
    for i in range(2,size-1,2):
        if array[i]>=array[i+1]:
            if array[i]>max:
                max = array[i]
            if array[i + 1] < min:
                min = array[i + 1]
        elif array[i+1] >= array[i]:
            if array[i+1]<min:
                min = array[i+1]
            if array[i] > max:
                max = array[i]
    return min,max


def checkNeighbour(dArray):
    flag = 1
    for i in range(0,len(dArray)):
        flag = 1
        for j in range(0,len(dArray[i])):
            if i != j:
                if dArray[i][j] == 'T' or dArray[j][i] == 'F':
                    flag = 0
        if flag == 1:
            return i


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    array = [4, 3, 9 ,1,0 ,7 ,5, 3 ,2 ,6]
    dArray= [ ['F','F','T','T','F'],
              ['T','F','T','T','F'],
              ['F','F','F','F','F'],
              ['F','T','T','F','T'],
              ['F','F','T','T','F']]
    print(checkNeighbour(dArray))


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
