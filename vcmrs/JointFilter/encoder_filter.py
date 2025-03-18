from functools import partial
from vcmrs.JointFilter.filter import filter as _filter

filter = partial(_filter, mode='pre')
