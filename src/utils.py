# coding=utf-8


def apply(list_func, obj):
    for f in list_func:
        obj = f(obj)
    return obj
