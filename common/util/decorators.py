"""
    Will contain all the custom decorators
"""


from typing import Any, Callable, Type, TypeVar, Union, overload


TFUNC2 = TypeVar("TFUNC2", bound=Callable[[Any], Any])
TCls = TypeVar("TCls")


@overload
def reflected(sub: Type[TCls]) -> Type[TCls]:
    ...


@overload
def reflected(sub: TFUNC2) -> TFUNC2:
    ...


def reflected(sub: Union[TCls, TFUNC2]) -> Union[TCls, TFUNC2]:
    """This function is used via reflection and must not be
    moved/renamed/altered unless the reflector gets updated as well"""
    return sub
