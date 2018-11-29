



# check if pair a is within pair b
# pair_a looks like (a1, a2), pair_b (b1, b2)
# within should satisfy the condition
# a1 >= b1 && a2 <= b2
def within(pair_a, pair_b):
    a1 = pair_a[0]
    a2 = pair_a[1]
    b1 = pair_b[0]
    b2 = pair_b[1]
    return a1 >= b1 and a2 <= b2



# check if the first element in pair_a is
# greater or equal than the second element in pair_b
def greater(pair_a, pair_b):
    return pair_a[0] >= pair_b[1] 
    



# l1 and l2 are nested list with two levels
# e.g. l1 may looks like [[1, 2], [3, 4]]
def same_length(l1, l2):
    if len(l1) != len(l2):
        return False
    else:
        i = 0
        while i < len(l1):
            if (len(l1[i]) != len(l2[i])):
                return False
            i += 1
    return True


# input a nested list, return max length of inner list
# e.g. if the input is [[1, 2], [3, 4, 5]], the result will be 3
def max_length(l):
    max_l = 0
    for l1 in l:
        if len(l1) > max_l:
            max_l = len(l1)
    return max_l



# l = [[1, 2], [5, 6, 7], [3], [4, 99, 2, -1]]
# print (max_length(l))




# p1 = (2, 4)
# p2 = (2, 4)
# p3 = (2, 9)
# p4 = (4, 9)
# p5 = (6, 9)

# print (within(p1, p2))
# print (within(p1, p3))
# print (within(p1, p4))
# print (within(p1, p5))

# print (greater(p1, p2))
# print (greater(p1, p3))
# print (greater(p5, p2))
# print (greater(p1, p5))
