import abc

from .signals import Param, State, _SignalDescriptor


class BlockMeta(abc.ABCMeta):
    def __call__(cls, *args, **kwargs):
        instance = super().__call__(*args, **kwargs)

        # call __set_name__ after class instantiation
        # to enable descriptor
        instance_members = dir(instance)
        for member in instance_members:
            obj = getattr(instance, member)
            if isinstance(obj, _SignalDescriptor):
                obj.__set_name__(instance.__class__, member)

        return instance


class System(abc.ABC, metaclass=BlockMeta):
    def __setattr__(self, name: str, value: object) -> None:
        if isinstance(value, _SignalDescriptor):
            value.__set_name__(type(self), name)

        super().__setattr__(name, value)
