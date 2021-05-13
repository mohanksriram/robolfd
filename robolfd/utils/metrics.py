"""This module does nothing and exists solely for the sake of OS compatibility."""

from typing import Type, TypeVar

T = TypeVar('T')


def record_class_usage(cls: Type[T]) -> Type[T]:
  return cls