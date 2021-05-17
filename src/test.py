from functools import  partial
def sum(a,b):
    print( a+b)

myfunc= partial(sum , b= 4)

myfunc(10)
