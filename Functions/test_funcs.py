import numpy as np

def test1(a):
    b = a-a
    return a/b

def test2(a):
    new_a = a*test1(1)

    return new_a
