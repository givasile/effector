import effector

print(effector.PDP.plot.__doc__)


# import griffe

# griffe_effector = griffe.load("effector", force_inspection=True)
# print(griffe_effector["global_effect_pdp.PDP.plot"].docstring.value)



# import griffe

# logger = griffe.get_logger("griffe_inspect_specific_objects")


# class InspectSpecificObjects(griffe.Extension):
#     """An extension to inspect just a few specific objects."""

#     def __init__(self, objects: list[str]) -> None:
#         self.objects = objects

#     def on_instance(self, *, obj: griffe.Object, **kwargs) -> None:
#         if obj.path not in self.objects:
#             return

#         try:
#             runtime_obj = griffe.dynamic_import(obj.path)
#         except ImportError as error:
#             logger.warning(f"Could not import {obj.path}: {error}")
#             return

#         if obj.docstring:
#             obj.docstring.value = runtime_obj.__doc__
#         else:
#             obj.docstring = griffe.Docstring(runtime_obj.__doc__)-pppppppppppppppppp
