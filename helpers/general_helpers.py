"""
##################################################
##################################################
## This module contains helper functions used   ##
## by the different elements of the trainer.    ##
##################################################
##################################################
"""

###################
##### IMPORTS #####
###################

# *** General modules imports. *** #

import inspect
import json
import os
import warnings
import sys
from   colorama import Fore
import time


# *** Own modules imports. *** #

# Import here!





#####################
##### FUNCTIONS #####
#####################

def inspect_class_parameters(class_instance):
    """
    This method allows for an easy way to inspect all parameters of a class instance.


    :param class_instance: The class instance whose parameters are meant to be inspected.


    :return: Nothing
    """

    attributes = inspect.getmembers(class_instance, lambda a:not(inspect.isroutine(a)))
    for attribute in attributes:
        if (attribute[0][:2] != '__' and attribute[0][:1] != '_'):
            print(attribute)




def file_len(fname):
    """
    Determines the number of lines in a file.


    :param fname: The path to the file for which the number of lines will be determined.


    :return: An integer that corresponds to the number of lines in the file specified by the parameters fname.
    """

    with open(fname, "r", encoding='utf-8') as f:
        for i, l in enumerate(f):
            pass
    return i + 1




def pretty_dict_json_dump(d, f, indent=3, current_indent=0):
    """
    Assumes keys are function parameter values, so, strings.
    TODO: Improve, comment
    """

    if (current_indent == 0):
        current_indent = indent

    if (len(d) > 0):
        max_key_len = max(len(json.dumps(repr(key))) for key in d)
    else:
        f.write("{\n" + " "*(current_indent-indent) + "}")

    for k_num, key in enumerate(d):
        if (k_num == 0):
            f.write("{\n")

        new_key = json.dumps(repr(key)) + " "*(max_key_len - len(json.dumps(repr(key)))) + " : "
        f.write(" "*current_indent + new_key)

        if (isinstance(d[key], dict)):
            pretty_dict_json_dump(d[key], f, current_indent=current_indent + len(new_key) + indent)
        else:
            json.dump(d[key], f)

        if (k_num == len(d) - 1):
            f.write("\n" + " "*(current_indent-indent) + "}")
        else:
            f.write(",\n")


def pretty_dict_json_load(f):
    """
    Assumes keys are function parameter values, so, strings.
    TODO: Improve, comment
    """

    loaded_dict = {}

    current_line = f.readline().strip().rstrip(',')

    if (current_line == "{"):
        current_line = f.readline().strip().rstrip(',')

    while (current_line != '}'):
        string_from_start = current_line[0] == "\""
        line_temp_on_string = current_line.split("\"")[(1 if string_from_start else 0):]
        line_temp = []
        for ele_num, line_temp_on_string_element in enumerate(line_temp_on_string):
            if ((ele_num + (1 if string_from_start else 0)) % 2 == 0):
                line_temp += line_temp_on_string_element.split()
            else:
                line_temp += ["\"" + line_temp_on_string_element + "\""]

        first_element = json.loads(line_temp[:2][0])[1:-1]
        last_element = ''.join(line_temp[2:])

        if (last_element == "{"):
            loaded_dict[first_element] = pretty_dict_json_load(f)
        else:
            loaded_dict[first_element] = json.loads(last_element)

        current_line = f.readline().strip().rstrip(',')

    return loaded_dict
# The one below is an old implementation that did not allow for space separated strings.
# I think the one above fixes that.
# def pretty_dict_json_load(f):
#     """
#     Assumes keys are function parameter values, so, strings.
#     DOES NOT WORK FOR ELEMENTS WITH SPACE SEPARATED STRINGS!!
#     TODO: Improve, comment
#     """
#
#     loaded_dict = {}
#
#     current_line = f.readline().strip().rstrip(',')
#
#     if (current_line == "{"):
#         current_line = f.readline().strip().rstrip(',')
#
#     while (current_line != '}'):
#         line_temp = current_line.split()
#         try:
#             first_element = json.loads(line_temp[:2][0])[1:-1]
#         except:
#             print(line_temp)
#             raise
#         last_element = ''.join(line_temp[2:])
#         if (last_element == "{"):
#             loaded_dict[first_element] = pretty_dict_json_load(f)
#         else:
#             loaded_dict[first_element] = json.loads(last_element)
#
#         current_line = f.readline().strip().rstrip(',')
#
#     return loaded_dict




def remove_files(top_path, termination, exception_terminations=tuple(), recursive=False):
    """
    TODO
    """
    existing_files = os.listdir(top_path)
    for existing_file in existing_files:
        path_to_file = os.path.join(top_path, existing_file)
        if (os.path.isdir(path_to_file)):
            if (recursive):
                remove_files(top_path + existing_file + '/', termination, exception_terminations, recursive)
        elif (os.path.isfile(path_to_file)):
            if (existing_file.endswith(termination) and
                    not any(existing_file.endswith(term) for term in exception_terminations)):
                os.remove(os.path.join(top_path, existing_file))
        else:
            continue




def isfloat(value):
    """
    Helper function used to determine whether a string is a float or not.
    From: "https://stackoverflow.com/questions/736043/checking-if-a-string-can-be-converted-to-float-in-python"


    :param value: The string to be tested.


    :return: A boolean indicating whether the string is a float or not.
    """

    try:
        float(value)
        return True
    except ValueError:
        return False




def get_path_components(path):
    """
    Function that returns the individual components of a path, i.e. the folders' names in the path (and a file name at
    the end, if one is part of the path).
    From: "https://stackoverflow.com/questions/3167154/how-to-split-a-dos-path-into-its-components-in-python"


    :param path: The path for which we want to get the components.


    :return: A list with the individual path components.
    """

    begin_sep = path[0]  == os.sep
    norm_path = os.path.normpath(path).split(os.sep)[(1 if begin_sep else 0):]
    return ([os.sep] if begin_sep else []) + norm_path




def join_path(path_components):
    """
    Function that given separate components of a path creates a valid path, according to the OS in use.


    :param path: The path components.


    :return: The composed path.
    """

    # TODO: Add os.sep at the end if provided? I think there was a reason for not doing that. Cannot remember. Test.
    return (os.sep if path_components[0] == os.sep else '') + os.path.join(*path_components)




def custom_formatwarning(msg, *args, **kwargs):
    # ignore everything except the message
    return '\n\n' + Fore.RED + "WARNING:\n" + str(msg) + Fore.RESET + '\n\n\n'

warnings.formatwarning = custom_formatwarning

def print_warning(msg):
    sys.stdout.flush()
    time.sleep(0.125)
    warnings.warn(msg)
    sys.stderr.flush()
    time.sleep(0.125)