# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Functions for making ``torch.nn.Module`` subclass instances stateless."""

import abc as _abc
from collections import OrderedDict as _OrderedDict
from contextlib import contextmanager as _contextmanager
import typing as _typing
import weakref as _weakref
import warnings as _warnings

import jittor
import jittor as jit

from . import utils as _utils

# ==============================================================================
# Helper functions and attributes for MonkeyPatch modules.
# ==============================================================================

_internal_attrs = {
    
    "_backend",
    "_parameters",
    "_buffers",
    "_backward_hooks",
    "_forward_hooks",
    "_forward_pre_hooks",
    "_state_dict_hooks",
    "_load_state_dict_pre_hooks",
    "_modules",
}

# _internal_attrs = {
    
#     "_backend",
#     "_backward_hooks",
#     "_forward_hooks",
#     "_forward_pre_hooks",
#     "_state_dict_hooks",
#     "_load_state_dict_pre_hooks",
# }


_BufferType = _typing.Dict[str, _typing.Optional[jit.Var]]


@_contextmanager
def _modify_internally(fmodule):
    fmodule._being_modified_internally = True
    yield
    fmodule._being_modified_internally = False


def _patched_parameters(
    self, recurse: bool = True, time: _typing.Optional[int] = None
) -> _typing.Iterable[jit.Var]:
    r"""Returns an iterator over monkey patched module fast parameters.

    Args:
        recurse (bool): if True, then yields fast parameters of this module
            and all submodules. Otherwise, this *still* yields parameters of
            this module and all submodules, and raises a warning. This keyword
            exists only to satisfy API compatibility with
            ``torch.nn.Module.parameters``.
        time (int or None): if None, the most recent fast parameters are
            provided. The int provided stands for the number of steps since the
            module was created. *Note* that the step counter is incremented
            every time parameters are updated, so this may not align with number
            of training or evaluations steps.

    Yields:
        Parameter: module fast weights.
    """
    if getattr(self, "_fast_params", None) is None:
        raise Exception(
            "Tried to get fast weights of a monkey patched module which does "
            "not encapsulate fast weights."
        )

    if not recurse:
        _warnings.warn(
            "Calling parameters with recurse=False on a monkey patched module "
            "still returns all the fast weights of of nested patched modules."
        )

    time = -1 if time is None else time

    if not self.track_higher_grads and time not in (-1, 0):
        raise ValueError(
            "The patched model is not tracking higher gradients. Only the "
            "latest parameters are available."
        )

    return iter(self._fast_params[time])


class _MonkeyPatchBase(_abc.ABC, jit.Module):
    @_abc.abstractmethod
    def __init__(self) -> None:
        self._param_mapping: _typing.List[int] = []
        self._being_modified_internally: bool = True
        self._track_higher_grads: bool = True

    def forward(self):
        raise NotImplementedError(
            "The monkey-patching logic has failed to override self.forward "
            "on the new module, or you tried calling forward on a patched "
            "version of a module which doesn't have forward (e.g. ModuleList)."
        )

    def _expand_params(self, params: _typing.List[jit.Var]) -> _typing.List[jit.Var]:
        expanded = []
        for index in self._param_mapping:
            expanded.append(params[index])
        return expanded

    @property
    def init_fast_params(self):
        if not self.track_higher_grads:
            raise Exception(
                "Cannot get initial parameters when not tracking higher " "gradients."
            )
        return self._fast_params[0]

    @property
    def fast_params(self):
        return None if self._fast_params is None else self._fast_params[-1]

    @fast_params.setter
    def fast_params(self, value):
        value = list(value)
        if self._fast_params is None:
            self._fast_params = []
        if self.track_higher_grads:
            self._fast_params.append(value)
        else:
            self._fast_params[0] = value

    @property
    def track_higher_grads(self):
        return self._track_higher_grads

    @track_higher_grads.setter
    def track_higher_grads(self, value):
        if not isinstance(value, bool):
            raise ValueError("Expected boolean argument. Got: {}.".format(type(value)))
        self._track_higher_grads = value


def buffer_sync(
    module: jit.Module, fmodule: _MonkeyPatchBase, device: _typing.Optional[str] = None
) -> None:
    r"""One off sync (copy) of buffers in ``fmodule`` with those from ``module``."""
    for key, value in module._buffers.items():
        if not isinstance(value, jit.Var):
            fmodule._buffers[key] = value
        elif device is None:
            fmodule._buffers[key] = value.clone().detach()
        else:
            fmodule._buffers[key] = value.clone().detach().to(device)
    for name, child in module._modules.items():
        if name in fmodule._modules:
            buffer_sync(child, fmodule._modules[name], device)
        else:
            raise KeyError(
                "Did not find expected submodule {} of monkey-patched module {}.".format(name, fmodule)
            )

# from collections import deque

# def buffer_sync(module, fmodule, device=None):
#     """One-off sync (copy) of buffers in `fmodule` with those from `module`."""
#     queue = deque([(module, fmodule)])
    
#     while queue:
#         current_module, current_fmodule = queue.popleft()
        
#         # Sync buffers
#         for key, value in getattr(current_module, "_buffers", {}).items():
#             if not isinstance(value, jit.Var):
#                 current_fmodule._buffers[key] = value
#             elif device is None:
#                 current_fmodule._buffers[key] = value.clone().detach()
#             else:
#                 current_fmodule._buffers[key] = value.clone().detach()  # Jittor 没有设备概念

#         # Add child modules to the queue
#         current_modules = getattr(current_module, "_modules", {})
#         current_fmodules = getattr(current_fmodule, "_modules", {})
        
#         for name, child in current_modules.items():
#             if name in current_fmodules:
#                 queue.append((child, current_fmodules[name]))
#             else:
#                 print("fine")
#                 # raise KeyError(
#                 #     f"Did not find expected submodule {name} "
#                 #     f"of monkey-patched module {current_fmodule}."
#                 # )



# def buffer_sync(
#     module: jit.Module, fmodule: _MonkeyPatchBase, device: _typing.Optional[str] = None
# ) -> None:
#     """Synchronize buffers from `module` to `fmodule`."""
#     # 获取 module 和 fmodule 的 buffer
#     module_buffers = getattr(module, "_buffers", None)
#     fmodule_buffers = getattr(fmodule, "_buffers", None)

#     if module_buffers is None or fmodule_buffers is None:
#         raise ValueError("Failed to access _buffers for module or fmodule.")

#     # 同步 buffer
#     for key, value in module_buffers.items():
#         if not isinstance(value, jit.Var):
#             fmodule_buffers[key] = value
#         elif device is None:
#             fmodule.register_buffer(key, value.clone().detach())
#         else:
#             fmodule.register_buffer(key, value.clone().detach().to(device))

#     # 递归同步子模块
#     module_modules = getattr(module, "_modules", None)
#     fmodule_modules = getattr(fmodule, "_modules", None)

#     if module_modules is None or fmodule_modules is None:
#         raise ValueError("Failed to access _modules for module or fmodule.")

#     for name, child in module_modules.items():
#         if name in fmodule_modules:
#             buffer_sync(child, fmodule_modules[name], device)
#         else:
#             raise KeyError(
#                 f"Did not find expected submodule {name} "
#                 f"of monkey-patched module {fmodule}."
#             )





# ==============================================================================
# Helper class to use instead of actual torch.nn.Parameters when patching.
# ==============================================================================


class _ParameterPlaceholder:
    def __init__(self, name: str) -> None:
        self._param_name = name

    def __repr__(self) -> str:
        return 'Parameter placeholder ("{}")'.format(self._param_name)


_ParameterPlaceholder.__name__ = "ParameterPlaceholder"
_ParameterPlaceholder.__qualname__ = "ParameterPlaceholder"

# ==============================================================================
# Helper function for recursively patching submodules.
# ==============================================================================


def _make_functional(
    module: jit.Module,
    params_box: _typing.Sequence[_typing.Optional[_typing.List[jit.Var]]],
    params_offset: int,
    root_patched: _typing.Optional[_MonkeyPatchBase] = None,
) -> _typing.Tuple[int, _MonkeyPatchBase, _typing.Type[_MonkeyPatchBase]]:
    if isinstance(module, _MonkeyPatchBase):
        raise ValueError(
            "Monkey-patching monkey-patched modules is untested uncharted "
            "territory, so we're going to assume it's done in error. If you "
            "are doing this intentionally and need this to be supported, "
            "contact the developers of this library."
        )

    # param_names = list(
    #     name
    #     for name in module._parameters.keys()
    #     if module._parameters[name] is not None
    # )
    param_names = list(
        name
        for name, param in module.named_parameters(False)
        if param is not None
    )
    
    # print('params_name',param_names)
    _ModuleType: _typing.Type[jit.Module] = module.__class__

    # type checking of next line disabled as mypy is iffy with dynamic types
    class MonkeyPatched(_ModuleType, _MonkeyPatchBase):  # type: ignore
        _wrapped_name = type(module).__name__

        def __init__(self, original_params, root) -> None:
            # init_params = {
            # "C": module.__dict__.get("_C", None),
            # "num_classes": module.__dict__.get("_num_classes", None),
            # "layers": module.__dict__.get("_layers", None),
            # "criterion": module.__dict__.get("_criterion", None),
            # }

            # init_params = {
            # "C": module.__dict__.get("_C", None),
            # "num_classes": module.__dict__.get("_num_classes", None),
            # "layers": module.__dict__.get("_layers", None),
            # "criterion": module.__dict__.get("_criterion", None),
            # }

            # # 确保所有必要的参数都存在
            # if None in init_params.values():
            #     missing = [k for k, v in init_params.items() if v is None]
            #     raise ValueError(f"Missing initialization parameters: {missing}")
            # 手动初始化 Jittor Module 的关键属性
            # self._buffers = {}
            # self._modules = {}
            # self._parameters = {}

            # 初始化 jit.Module
            # jit.Module.__init__(self, **init_params)
            jit.Module.__init__(self)
            _MonkeyPatchBase.__init__(self)
            self._root_ref = _weakref.ref(root) if root else None

            self._fast_params = None
            self._param_names = param_names
            # print("_param_names", self._param_names)
            self._original_params = original_params
            # print("_original_params", self._original_params)
            # for pretty printing
            # self._parameters_not_recur = _OrderedDict(
            #     (name, _ParameterPlaceholder(name))
            #     for name in self._param_names
            # )
            # print(self._param_names)
            # self.load_parameters(_OrderedDict(
            #     (name, _ParameterPlaceholder(name))
            #     for name in self._param_names
            # ))
            #     for name in self._param_names)
            # for name in self._param_names:
            #     placeholder = _ParameterPlaceholder(name)
            #     # setattr(self, name, placeholder)
            #     self.safe_setattr(name, placeholder)

        @property
        def direct_submodule_call(self):
            return params_box[0] is None

        @property
        def is_root(self):
            return self._root_ref is None

        @property
        def root(self):
            if self.is_root:
                return self
            else:
                return self._root_ref()

        def safe_setattr(self, name, value):
                """
                Safely set attributes for a Jittor Module, handling Jittor's dynamic properties
                like _modules, _parameters, and _buffers without causing recursion issues.
                """
                def remove_from(*dicts):
                    for d in dicts:
                        if name in d:
                            del d[name]

                # Special handling for _parameters
                # params = getattr(self, "_parameters", None)
                params = self.named_parameters(False)
                # print('params',params)
                if params is not None and name in params:
                    if not isinstance(value, jit.Var):
                        raise TypeError(
                            "Require Tensor as fast weights. "
                            "Got {}".format(type(value).__name__)
                        )

                    if not self._being_modified_internally:
                        # Additional behaviour for when fast weights are being directly modified
                        old_value = self._parameters[name]
                        fast_params = self.root.fast_params[:]
                        if not fast_params:
                            raise Exception(
                                "Cannot assign parameters to patched module which "
                                "does not have implicit fast parameters."
                            )
                        replacement_index = _utils._find_param_in_list(old_value, fast_params)
                        fast_params[replacement_index] = value
                        self.update_params(fast_params)

                    # Change parameters in place, usually during boxed_forward pass
                    # self._parameters[name] = value
                    # setattr(self,name,value)
                    object.__setattr__(self, name, value)
                else:
                    # Special handling for _modules
                    modules = getattr(self, "_modules", None)
                    if isinstance(value, jit.Module):
                        if modules is None:
                            raise AttributeError(
                                "cannot assign module before Module.__init__() call"
                            )
                        remove_from(self.__dict__, self._parameters, self._buffers)
                        # modules[name] = valu
                        self.add_module(name,value)
                    elif modules is not None and name in modules:
                        if value is not None:
                            raise TypeError(
                                (
                                    "cannot assign '{}' as child module '{}'"
                                    "(jit.Module or None expected)"
                                ).format(type(value).__name__, name)
                            )
                        self.add_module(name,value)
                    else:
                        # Special handling for _buffers
                        buffers = getattr(self, "_buffers", None)
                        if buffers is not None and name in buffers:
                            if value is not None and not isinstance(value, jit.Var):
                                raise TypeError(
                                    "cannot assign '{}' as buffer '{}' "
                                    "(jit.Var or None expected)".format(type(value).__name__, name)
                                )
                            # buffers[name] = value
                            self.register_buffer(name, value)
                        else:
                            # Default behavior
                            object.__setattr__(self, name, value)


        # def __setattr__(self, name, value):
        #     def remove_from(*dicts):
        #         for d in dicts:
        #             if name in d:
        #                 del d[name]

        #     # print(f"Attempting to set attribute: {name}, value: {value}")
        #     # if hasattr(self, name):
        #     #     existing_value = getattr(self, name, None)
        #     #     print(f"Existing attribute '{name}': {existing_value}")
        #     # else:
        #     #     print(f"Attribute '{name}' does not exist yet.")
            
        #     # 特殊处理 Jittor 的内部属性
        #     # if name in {"_modules", "_parameters", "_buffers"}:
        #     #     try:
        #     #         object.__setattr__(self, name, value)
        #     #         print(f"Attribute '{name}' set successfully.")
        #     #     except Exception as e:
        #     #         print(f"Failed to set attribute '{name}': {e}")
        #     #     return

        #     params = self.__dict__.get("_parameters")
        #     print('params',params)
        #     if params is not None and name in params:
        #         if not isinstance(value, jit.Var):
        #             raise TypeError(
        #                 "Require Tensor as fast weights. "
        #                 "Got {}".format(type(value).__name__)
        #             )

        #         if not self._being_modified_internally:
        #             # Additional behaviour for when fast weights are being
        #             # directly modified goes here:
        #             old_value = self._parameters[name]
        #             fast_params = self.root.fast_params[:]
        #             if not fast_params:
        #                 raise Exception(
        #                     "Cannot assign parameters to patched module which "
        #                     "does not have implicit fast parameters."
        #                 )
        #             replacement_index = _utils._find_param_in_list(
        #                 old_value, fast_params
        #             )
        #             fast_params[replacement_index] = value
        #             self.update_params(fast_params)

        #         # Change parameters in place, usually during boxed_forward pass
        #         self._parameters[name] = value
        #     else:
        #         # modules = self.__dict__.get("_modules")
        #         modules = getattr(self, "_modules", None)
        #         print("modules", modules)
        #         if isinstance(value, jit.Module):
        #             # print(value)
        #             if modules is None:
        #                 raise AttributeError(
        #                     "cannot assign module before Module.__init__() " "call"
        #                 )
        #             remove_from(self.__dict__, self._parameters, self._buffers)
        #             modules[name] = value
        #             # self.add_module(name, value)
        #         elif modules is not None and name in modules:
        #             if value is not None:
        #                 raise TypeError(
        #                     (
        #                         "cannot assign '{}' "
        #                         "as child module '{}'"
        #                         "(torch.nn.Module or None expected)"
        #                     ).format(type(value).__name__, name)
        #                 )
        #             modules[name] = value
        #         else:
        #             buffers = self.__dict__.get("_buffers")
        #             if buffers is not None and name in buffers:
        #                 if value is not None and not isinstance(value, jit.Var):
        #                     raise TypeError(
        #                         "cannot assign '{}' as buffer '{}' "
        #                         "(torch.Tensor or None expected)".format(
        #                             type(value).__name__, name
        #                         )
        #                     )
        #                 buffers[name] = value
        #             else:
        #                 object.__setattr__(self, name, value)


    MonkeyPatched.__name__ = "InnerFunctional" + type(module).__name__
    MonkeyPatched.__qualname__ = MonkeyPatched.__name__

    fmodule = MonkeyPatched(module.parameters(), root=root_patched)

    # If a root module hasn't been defined yet, this fmodule is the root
    if not root_patched:
        root_patched = fmodule

    # use 1 as dummy list item since we are only counting
    # num_params = len([1 for p in module._parameters.values() if p is not None])
    num_params = len([1 for p,_ in module.named_parameters(False) if _ is not None])
    # print('num_params',num_params)
    # # Copy over all attributes
    # print(fmodule.__dict__.keys())
    # print(fmodule._modules.items())
    for name, attr in module.__dict__.items():
        # print('set_module',name)
        if name in _internal_attrs:
            continue
        # setattr(fmodule, name, attr)
        fmodule.safe_setattr(name, attr)
    # k=0
    # # print(module['_param_mapping'])
    # for name, attr in module.__dict__.items():
    #     k+=1
    #     if name in _internal_attrs:
    #         continue

    #     # setattr(fmodule, name, attr)
    #     fmodule.safe_setattr(name, attr)
    # print(len(module._parameters.keys()))
    # print('new!!!!!!!!!!!!')
    # for k,b  in enumerate(module.parameters()):
    #     print('items ',k ,b.shape)

    # Deal with "None"-style params
    
    with _modify_internally(fmodule):
        # for name, attr in module.__dict__['_parameters'].items():
        for name, attr in module.named_parameters(False):
            if isinstance(attr, jit.Var) and not attr.stop_grad:
                continue
            else:
                # setattr(fmodule, name, attr)
                fmodule.safe_setattr(name, attr)

    child_params_offset = params_offset + num_params
    # print(child_params_offset)
    # k=1
    # for name, child in module._modules.items():
    # for name, child in enumerate(module.modules()):
    #     print(child)
    for name, child in module._modules.items():
        # k+=1
        # print('_make_functional',name)
        # print(type(child).__name__)
        # print('old!!!!!!!!!!!!',k)
        # if name =='classifier':
        #     print(child.__dict__)
        #     print(child.weight.shape)
        # if 'cell' in name:
        #     print('cells test')
        #     print(child._modules.keys())
        # if hasattr(child,'shape'):
        #     print("name",name,child.shape)

        child_params_offset, fchild, _ = _make_functional(
            child, params_box, child_params_offset, root_patched
        )
        # fmodule._modules[name] = fchild
        # setattr(fmodule, name, fchild)
        fmodule.safe_setattr(name, fchild)

    true_forward = getattr(type(module), "execute", None) or getattr(
        type(module), "forward", None
    )
    # true_forward = type(module).forward

    def patched_forward(self, *args, params=None, **kwargs):
        # print('run patched_forward!!!')
        if self.direct_submodule_call:
            # If submodule was called directly, run intialisation that happens
            # at top level call. If *full set of params* is provided here, it
            # will use those. If not, it will fall back on fast weights.
            # In the future, we should be able to support passing only the
            # submodule (+ children) weights here, but that's not simple.
            self.root._refill_params_box(params)

        with _modify_internally(self):
            for name, param in zip(
                self._param_names,
                params_box[0][params_offset : params_offset + num_params],
            ):
                setattr(self, name, param)

            # This snippet deals with torch.nn.{RNN,GRU,LSTM}
            if hasattr(self, "_flat_weights_names"):
                self._flat_weights = [
                    self._parameters[wn] for wn in self._flat_weights_names
                ]

        # Call true_forward after some checks
        with _warnings.catch_warnings():
            # If running RNNs on GPU, surpress the warnings due to flattening
            # not happening here. Maybe we should raise a warning of our own?
            is_RNN = isinstance(module, jit.nn.RNN)
            if is_RNN and jit.flags.use_cuda:
                _warnings.simplefilter("ignore", category=UserWarning)

            return true_forward(self, *args, **kwargs)

    # setattr(MonkeyPatched, "forward", patched_forward)
    setattr(MonkeyPatched, "execute", patched_forward)
    # MonkeyPatched.safe_setattr("execute", patched_forward)

    def flatten_parameters(self):
        return  # no-op

    # This (hopefully) avoids trouble on GPU with torch.nn.{RNN,GRU,LSTM}
    if hasattr(module, "flatten_parameters"):
        setattr(MonkeyPatched, "flatten_parameters", flatten_parameters)
        # MonkeyPatched.safe_setattr("flatten_parameters", flatten_parameters)
    return child_params_offset, fmodule, type(fmodule)


def _update_patched_params(
    fmodule: _MonkeyPatchBase,
    params_box: _typing.Sequence[_typing.List[jit.Var]],
    params_offset: int,
) -> int:
    # num_params = len([1 for p in fmodule._parameters.values() if p is not None])
    num_params = len([1 for p,_ in fmodule.named_parameters(False) if _ is not None])
    child_params_offset = params_offset + num_params
    for name, child in fmodule._modules.items():
        child_params_offset = _update_patched_params(
            child, params_box, child_params_offset
        )

    with _modify_internally(fmodule):
        for name, param in zip(
            fmodule._param_names,
            params_box[0][params_offset : params_offset + num_params],
        ):
            setattr(fmodule, name, param)
    return child_params_offset


# ==============================================================================
# The main function which does the monkey patching.
# ==============================================================================
_EncapsulatorType = _typing.Optional[
    _typing.Callable[[_MonkeyPatchBase, jit.Var], None]
]


def make_functional(
    module: jit.Module, encapsulator: _EncapsulatorType = None
) -> _MonkeyPatchBase:
    r"""Returns a stateless version of an ``nn.Module`` instance."""
    params_box = [None]
    _, fmodule, MonkeyPatched = _make_functional(module, params_box, 0)

    top_name = "Functional" + MonkeyPatched._wrapped_name
    MonkeyPatched.__name__ = MonkeyPatched.__qualname__ = top_name
    ###################
    # MonkeyPatched.boxed_forward = MonkeyPatched.forward
    MonkeyPatched.boxed_forward = MonkeyPatched.execute
    #############
    param_mapping = _utils._get_param_mapping(module, [], [])
    setattr(fmodule, "_param_mapping", param_mapping)

    def _refill_params_box(self, params):
        if params is not None:
            self.fast_params = params  # update view on latest fast params
        elif self.fast_params is None:
            raise ValueError(
                "params keyword must be provided if patched module not "
                "tracking its own fast parameters"
            )

        # Copy fast parameters into params_box for use in boxed_forward
        params_box[0] = self._expand_params(self.fast_params)

    def _patched_forward(self, *args, params=None, **kwargs):
        self._refill_params_box(params)

        output = self.boxed_forward(*args, **kwargs)

        # Clean up
        params_box[0] = None

        return output

    def _update_params(self, params):
        self.fast_params = params
        # print(params)
        # print('lang')
        # print('right',params[-1].shape)
        # print('right',params[-2].shape)
        # print(params[-3].shape)
        # print(params[-4].shape)
        # print(len(params))
        params = self._expand_params(params)
        # print(len(params))
        # print('not right',params[-1].shape)
        # print('not right',params[-2].shape)
        _update_patched_params(self, [params], 0)

    ##############
    setattr(MonkeyPatched, "execute", _patched_forward)
    ##############
    # setattr(MonkeyPatched, "forward", _patched_forward)
    setattr(MonkeyPatched, "parameters", _patched_parameters)
    setattr(MonkeyPatched, "update_params", _update_params)
    setattr(MonkeyPatched, "_refill_params_box", _refill_params_box)
    
    if encapsulator is not None:
        encapsulator(fmodule, module)
    return fmodule


# ==============================================================================
# Convenience functions and decorators for hiding away a lot of the complexity
# of creating patched modules, taking their parameters, and linking patched
# modules to a differentiable optimizer.
# ==============================================================================


def monkeypatch(
    module: jit.Module,
    device: _typing.Optional[str] = None,
    copy_initial_weights: bool = True,
    track_higher_grads: bool = True,
) -> _MonkeyPatchBase:
    r"""Create a monkey-patched stateless version of a module.

    This function produces a monkey-patched version of a module, and returns a
    copy of its parameters for use as fast weights. Where the original module
    or any of its submodules have state (e.g. batch norm), this will be copied
    too, but further updates (e.g. during inner loop training) will cause these
    to diverge without changing the state of the original module.

    Args:
        module: a ``torch.nn.Module`` subclass instance.
        device (optional): a device to cast the fast weights and state to.
        copy_initial_weights: if True, the weights of the patched module are
            copied to form the initial weights of the patched module, and thus
            are not part of the gradient tape when unrolling the patched module.
            If this is set to False, the actual module weights will be the
            initial weights of the patched module. This is useful when doing
            MAML, for example.
        track_higher_grads: if True, during unrolled optimization the graph be
            retained, and the fast weights will bear grad funcs, so as to permit
            backpropagation through the optimization process. Setting this to
            False allows ``monkeypatch`` to be used in "test mode", without
            potentially tracking higher order gradients. This can be useful when
            running the training loop at test time, e.g. in k-shot learning
            experiments, without incurring a significant memory overhead.

    Returns:
        ``fmodule``: a "stateless" version of the original module, for which calls
        to forward take the additional kwarg-only parameter ``params``, which
        should be a list of torch tensors requiring gradients, ideally
        provided by this function (see below) or by an update step from one
        of the optimizers in ``higher.optim``.
    """

    def encapsulator(fmodule: _MonkeyPatchBase, module: jit.Module) -> None:
        if copy_initial_weights:
            params = _utils.get_func_params(module, device=device)
        else:   
            params = [
                p.clone() if device is None else p.clone().to(device)
                for p in module.parameters()
            ]
        # params = [
        #     p.clone() if device is None else p.clone().to(device)
        #     for p in module.parameters()
        # ]
        # print('now')
        # print(len(params))
        # print(params[-2].shape)
        # print(params[-1].shape)
        # print('test',fmodule.classifier.weight.shape)
        # print('test',module.classifier.weight.shape)
        # print('test',len(fmodule.cells))
        # print('test',len(module.cells))
        buffer_sync(module, fmodule, device)
        # print('test',len(fmodule.cells))
        # print('test',len(module.cells))
        # print('test',fmodule.classifier.weight.shape)
        # print('test',module.classifier.weight.shape)
        fmodule.update_params(params)
        # new_params= [
        #         p.clone() if device is None else p.clone().to(device)
        #         for p in fmodule.parameters()
        #     ]
        # print(new_params[-2].shape)
        # print(new_params[-1].shape)
        # print(len(new_params))
        # print('test',fmodule.classifier.weight.shape)
        # print('test',module.classifier.weight.shape)

    fmodule = make_functional(module, encapsulator=encapsulator)
    # print('final_test',module.classifier.weight.shape)
    fmodule.track_higher_grads = track_higher_grads

    return fmodule
