import sys

"""function to remove char from string"""
def remove_char(s, c):
    """remove char from string"""
    return s.replace(c, '')


"""replace char in string with another char"""
def replace_char(s, c1, c2):
    """replace char in string with another char"""
    return s.replace(c1, c2)

def insert_char(s, c, pos):
    """insert char in string at position"""
    return s[:pos] + c + s[pos:]


#          Sunday,Saturday
def editDIST(s1, s2):
    """edit distance between two strings"""
    if len(s1) == 0 and len(s2) == 0:
        return 0
    if len(s1) == 0:
        return len(s2)
    if len(s2) == 0:
        return len(s1)
    if s1[-1] == s2[-1]:
        return editDIST(s1[:-1], s2[:-1])

    ins =editDIST(s1[:-1], s2) + 1
    dels = editDIST(s1, s2[:-1]) + 1
    subs = editDIST(s1[:-1], s2[:-1]) + 1
    return min(ins, dels, subs)




print(editDIST("Sunday", "Saturday"))
