def my_gen(n:int):
    for i in range(n):
        yield i


print(type(my_gen))
for x in my_gen:
    print(x)
