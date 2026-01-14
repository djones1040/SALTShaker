"""
Configuration parsing utilities for SALT3 training.

This module provides enhanced argument parsing that combines configuration
files with command-line overrides, supporting environment variable expansion
and type conversion.

Classes
-------
FullPaths
    Argparse action that expands environment variables and ~ in paths.
EnvAwareArgumentParser
    ArgumentParser subclass with environment variable expansion.
ConfigWithCommandLineOverrideParser
    Parser that reads defaults from config files with CLI override support.

Functions
---------
boolean_string
    Convert string to boolean for argparse.
nonetype_or_int
    Convert string to None or int for argparse.
generateerrortolerantaddmethod
    Wrapper for error-tolerant argument adding.
"""

import argparse
import configparser
from os import path

import logging

log = logging.getLogger(__name__)

__all__ = [
    "expandvariablesandhomecommaseparated",
    "FullPaths",
    "EnvAwareArgumentParser",
    "ConfigWithCommandLineOverrideParser",
    "boolean_string",
    "nonetype_or_int",
    "generateerrortolerantaddmethod",
]


def expandvariablesandhomecommaseparated(paths):
    """
    Expand environment variables and ~ in comma-separated paths.

    Parameters
    ----------
    paths : str
        Comma-separated list of paths.

    Returns
    -------
    str
        Comma-separated paths with variables expanded.
    """
    return ",".join([path.expanduser(path.expandvars(x)) for x in paths.split(",")])


class FullPaths(argparse.Action):
    """
    Argparse action that expands environment variables and ~ in path arguments.

    Use with ``action=FullPaths`` in add_argument().
    """

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, expandvariablesandhomecommaseparated(values))


def boolean_string(s):
    """
    Convert string to boolean for argparse type conversion.

    Parameters
    ----------
    s : str
        String value ('True', 'False', 'true', 'false', '1', '0').

    Returns
    -------
    bool
        Converted boolean value.

    Raises
    ------
    ValueError
        If string is not a valid boolean representation.
    """
    if s not in {"False", "True", "false", "true", "1", "0"}:
        raise ValueError("Not a valid boolean string")
    return (s == "True") | (s == "1") | (s == "true")


def nonetype_or_int(s):
    """
    Convert string to None or int for argparse type conversion.

    Parameters
    ----------
    s : str
        String value ('None' or integer string).

    Returns
    -------
    None or int
        None if s == 'None', otherwise int(s).
    """
    if s == "None":
        return None
    else:
        return int(s)


class EnvAwareArgumentParser(argparse.ArgumentParser):
    """
    ArgumentParser subclass that expands environment variables in defaults.

    Automatically expands $VAR and ~ in default values for arguments
    using the FullPaths action.
    """

    def add_argument(self, *args, **kwargs):
        if "action" in kwargs:
            action = kwargs["action"]
            if (action == FullPaths or action == "FullPaths") and "default" in kwargs:
                kwargs["default"] = expandvariablesandhomecommaseparated(
                    kwargs["default"]
                )
        return super().add_argument(*args, **kwargs)


class ConfigWithCommandLineOverrideParser(EnvAwareArgumentParser):
    """
    Parser combining config file defaults with command-line overrides.

    Reads default values from a ConfigParser object, allowing command-line
    arguments to override them. Supports multiple key aliases and automatic
    type conversion.

    Examples
    --------
    >>> config = configparser.ConfigParser()
    >>> config.read('training.conf')
    >>> parser = ConfigWithCommandLineOverrideParser()
    >>> parser.add_argument_with_config_default(
    ...     config, 'trainparams', 'maxiter', type=int)
    """

    def addhelp(self):
        default_prefix = "-"
        self.add_argument(
            default_prefix + "h",
            default_prefix * 2 + "help",
            action="help",
            default=argparse.SUPPRESS,
            help="show this help message and exit",
        )

    def add_argument_with_config_default(self, config, section, *keys, **kwargs):
        """
        Add argument with default from config file.

        Parameters
        ----------
        config : ConfigParser
            Configuration parser with loaded config file.
        section : str
            Section name in config file.
        *keys : str
            One or more key names to look for in the section.
        **kwargs
            Additional arguments passed to add_argument().
            Special kwargs:
            - clargformat : str, format for CLI argument name
              (default: '--{key}', use 'prependsection' for '--{section}_{key}')

        Raises
        ------
        KeyError
            If no matching key found and no default provided.
        """
        if "clargformat" in kwargs:
            if kwargs["clargformat"] == "prependsection":
                kwargs["clargformat"] = "--{section}_{key}"
        else:
            kwargs["clargformat"] = "--{key}"

        clargformat = kwargs.pop("clargformat")

        clargs = [clargformat.format(section=section, key=key) for key in keys]

        def checkforflagsinconfig():
            for key in keys:
                if key in config[section]:
                    return key, config[section][key]
            raise KeyError(f"key {key} not found in section {section} of config file")

        try:
            includedkey, kwargs["default"] = checkforflagsinconfig()
        except KeyError:
            if "default" in kwargs:
                pass
            else:
                message = f"Key {keys[0]} not found in section {section}; valid keys include: {', '.join(keys)}"
                if "help" in kwargs:
                    message += f"\nHelp string: {kwargs['help'].format(**kwargs)}"
                raise KeyError(message)
        if "nargs" in kwargs and (
            (type(kwargs["nargs"]) is int and kwargs["nargs"] > 1)
            or (type(kwargs["nargs"] is str and (kwargs["nargs"] in ["+", "*"])))
        ):
            if kwargs["default"] != argparse.SUPPRESS:
                if not "type" in kwargs:
                    kwargs["default"] = kwargs["default"].split(",")
                else:
                    kwargs["default"] = list(
                        map(kwargs["type"], kwargs["default"].split(","))
                    )
                if type(kwargs["nargs"]) is int:
                    try:
                        assert len(kwargs["default"]) == kwargs["nargs"]
                    except:
                        nargs = kwargs["nargs"]
                        numfound = len(kwargs["default"])
                        raise ValueError(
                            f"Incorrect number of arguments in {(config)}, section {section}, key {includedkey}, {nargs} arguments required while {numfound} were found"
                        )
        return super().add_argument(*clargs, **kwargs)


def generateerrortolerantaddmethod(parser):
    """
    Create error-tolerant wrapper for add_argument_with_config_default.

    Returns a function that catches exceptions when adding arguments,
    logging errors instead of raising them.

    Parameters
    ----------
    parser : ConfigWithCommandLineOverrideParser
        Parser to wrap.

    Returns
    -------
    callable
        Wrapped add method that returns True on success, False on error.
    """

    def wraparg(*args, **kwargs):
        try:
            parser.add_argument_with_config_default(*args, **kwargs)
            return True
        except Exception as e:
            log.error("\n".join(e.args))
            return False

    return wraparg
