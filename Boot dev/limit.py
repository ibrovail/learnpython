limit = int(input("Limit: "))

num = 1
sum_cons = 0
con_sum = "The consecutive sum: "

while limit > sum_cons:
    sum_cons = sum_cons + num
    # add "+" only if this is not the last number
    if limit > sum_cons:
        con_sum += f"{num} + "
    else:
        con_sum += f"{num}"
    num += 1
    print(f"{con_sum} = {sum_cons}")
    
print(f"{con_sum} = {sum_cons}")