#Recursive way of solve the matrix chain problem
# def MatrixChainOrder(p, i, j):
#     if i == j:
#         return 0
#     min = float('inf')
#     for k in range(i, j):
#         count = MatrixChainOrder(p, i, k) + MatrixChainOrder(p, k + 1, j) + p[i - 1] * p[k] * p[j]
#         if count < min:
#             min = count
#     return min


# Dynamic programming way of solve the matrix chain problem
import sys

dp = [[-1 for i in range(100)] for j in range(100)]


# Function for matrix chain multiplication
def matrixChainMemoised(p, i, j):
    if (i == j):
        return 0

    if (dp[i][j] != -1):
        return dp[i][j]

    dp[i][j] = sys.maxsize

    for k in range(i, j):
        dp[i][j] = min(dp[i][j],
                       matrixChainMemoised(p, i, k) + matrixChainMemoised(p, k + 1, j) + p[i - 1] * p[k] * p[j])

    return dp[i][j]


def MatrixChainOrder(p, n):
    i = 1
    j = n - 1
    return matrixChainMemoised(p, i, j)





# Driver code
arr = [40, 20, 30, 10, 30]
n = len(arr)

# print("Minimum number of multiplications is ",
#       MatrixChainOrder(arr, n))

# Dynamic Programming Python implementation of Matrix
# Chain Multiplication. See the Cormen book for details
# of the following algorithm


# Matrix Ai has dimension p[i-1] x p[i] for i = 1..n


def MatrixChainOrder(p, n):
    # For simplicity of the program,
    # one extra row and one
    # extra column are allocated in m[][].
    # 0th row and 0th
    # column of m[][] are not used
    m = [[0 for x in range(n)] for x in range(n)]

    # m[i, j] = Minimum number of scalar
    # multiplications needed
    # to compute the matrix A[i]A[i + 1]...A[j] =
    # A[i..j] where
    # dimension of A[i] is p[i-1] x p[i]

    # cost is zero when multiplying one matrix.
    for i in range(1, n):
        m[i][i] = 0

    # L is chain length.
    for L in range(2, n):
        for i in range(1, n - L + 1):
            j = i + L - 1
            m[i][j] = sys.maxsize
            for k in range(i, j):

                # q = cost / scalar multiplications
                q = m[i][k] + m[k + 1][j] + p[i - 1] * p[k] * p[j]
                if q < m[i][j]:
                    m[i][j] = q

    return m[1][n - 1]


# # Driver code
# arr = [5, 2,10, 5, 3]
# size = len(arr)
#
# print("Minimum number of multiplications is " +
#       str(MatrixChainOrder(arr, size)))





def game(arr,i,j):

    if (i>j):
        return 0
    if (i==j):
        return arr[i]

    if(i+1==j):
        return max(arr[i],arr[j])

    res1=arr[i]+min(game(arr,i+1,j-1),game(arr,i+2,j))
    res2=arr[j]+min(game(arr,i+1,j-1),game(arr,i,j-2))

    return max(res1,res2)


print(game([3,6,10,5],0,3))


"""given arr of whole numbers A, and target sum S, find sub-list of A that sums to S
For example if A={2,1,13,5,3,21} and S=16 return true because {2,1,13} sums to 16, for S=12 return false because no sub-list sums to 12"""
def sublist(arr,s):
    if(s==0):
        return True
    if(len(arr)==0):
        return False
    if(arr[0]==s):
        return True
    if(arr[0]>s):
        return sublist(arr[1:],s)
    return sublist(arr[1:],s-arr[0]) or sublist(arr[1:],s)


print(sublist([2,1,13,5,3],50))


