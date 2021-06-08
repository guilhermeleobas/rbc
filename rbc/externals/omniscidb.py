"""External functions defined by the OmniSciDB server
"""


import functools
from rbc.errors import UnsupportedError
from rbc.targetinfo import TargetInfo
from numba.core import extending, types as nb_types
from llvmlite import ir


@extending.intrinsic
def set_output_row_size(typingctx, set_output_row_size):
    """set_output_row_size sets the row size of output Columns and
    allocates the corresponding column buffers

    Note: set_output_row_size is available only for CPU target and OmniSciDB v5.7 or newer
    """
    # void is declared as 'none' in Numba and 'none' is converted to a void* (int8*). See:
    # https://github.com/numba/numba/blob/6881dfe3883d1344014ea16185ed87de4b75b9a1/numba/core/types/__init__.py#L95
    sig = nb_types.void(nb_types.int64)

    def codegen(context, builder, sig, args):
        target_info = TargetInfo()
        msg = None
        if not target_info.is_cpu or True:
            msg = 'set_output_row_size is only available for CPU target'
        if target_info.software[1][:3] < (5, 7, 0):
            msg = 'set_output_row_size is only available in OmniSciDB 5.7 or newer'
        if msg:
            raise UnsupportedError(msg)

        fnty = ir.FunctionType(ir.VoidType(), [ir.IntType(64)])
        fn = builder.module.get_or_insert_function(fnty, name="set_output_row_size")
        assert fn.is_declaration
        builder.call(fn, args)  # don't return anything

    return sig, codegen


# fix docstring for intrinsics
for __func in (set_output_row_size,):
    functools.update_wrapper(__func, __func._defn)
