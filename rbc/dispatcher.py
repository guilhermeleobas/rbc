import inspect
import collections
import functools
from numba.core import dispatcher, compiler
from numba.core import utils, types, registry
# from rbc.irtools import JITRemoteTypingContext, JITRemoteTargetContext
from rbc.targetinfo import TargetInfo
# codegen, cpu, compiler_lock, \
#     registry, typing, compiler, sigutils, cgutils, \
#     extending, utils, types
from numba.core.caching import NullCache

# class JITDispatcher(dispatcher.Dispatcher):
class JITDispatcher(registry.CPUDispatcher):

    def __init__(self, py_func, locals={}, targetoptions={},
                 impl_kind='direct', pipeline_class=compiler.Compiler):
        """
        Parameters
        ----------
        py_func: function object to be compiled
        impl_kind: str
            Select the compiler mode for `@jit` and `@generated_jit`
        pipeline_class: type numba.compiler.CompilerBase
            The compiler pipeline type.
        """
        # self.typingctx = TargetInfo().typingctx
        # self.targetctx = TargetInfo().targetctx
        # self.typingctx = JITRemoteTypingContext()
        # self.targetctx = JITRemoteTargetContext(self.typingctx)
        self.typingctx = self.targetdescr.typing_context
        self.targetctx = self.targetdescr.target_context

        pysig = utils.pysignature(py_func)
        arg_count = len(pysig.parameters)
        can_fallback = not targetoptions.get('nopython', False)

        dispatcher._DispatcherBase.__init__(self, arg_count, py_func, pysig, can_fallback,
                                            exact_match_required=False)

        functools.update_wrapper(self, py_func)

        self.targetoptions = targetoptions
        self.locals = locals
        self._cache = NullCache()
        compiler_class = self._impl_kinds[impl_kind]
        self._impl_kind = impl_kind
        self._compiler = compiler_class(py_func, self.targetdescr,
                                        targetoptions, locals, pipeline_class)
        self._cache_hits = collections.Counter()
        self._cache_misses = collections.Counter()

        self._type = types.Dispatcher(self)
        self.typingctx.insert_global(self, self._type)

    def compile(self, sig):
        self.typingctx = TargetInfo().typingctx
        self.targetctx = TargetInfo().targetctx
        # print(TargetInfo().is_gpu, id(self.targetctx))
        super().compile(sig)


def jit(func, **dispatcher_args):

    dispatcher = JITDispatcher

    def wrapper(func):
        # if extending.is_jitted(func):
        #     raise TypeError(
        #         "A jit decorator was called on an already jitted function "
        #         f"{func}.  If trying to access the original python "
        #         f"function, use the {func}.py_func attribute."
        #     )

        if not inspect.isfunction(func):
            raise TypeError(
                "The decorated object is not a function (got type "
                f"{type(func)})."
            )

        disp = dispatcher(py_func=func, **dispatcher_args)
        # if sigs is not None:
        #     # Register the Dispatcher to the type inference mechanism,
        #     # even though the decorator hasn't returned yet.
        #     from numba.core import typeinfer
        #     with typeinfer.register_dispatcher(disp):
        #         for sig in sigs:
        #             disp.compile(sig)
        #         disp.disable_compile()
        return disp

    return wrapper(func)
