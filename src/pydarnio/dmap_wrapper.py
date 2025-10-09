"""
Wrappers around the `darn-dmap` API.

Each file type will have one function for calling any type of reading (regular, lax, bytes) or any type of writing
(regular, bytes).
"""
from typing import Union, Optional
import dmap


def read_dispatcher(source: Union[str, bytes], fmt: str, mode: str) -> Union[dict, list[dict], tuple[list[dict], Optional[int]]]:
    """
    Reads in DMAP data from `source`.

    Parameters
    ----------
    source: Union[str, bytes]
        Where to read data from. If input is of type `str`, this is interpreted as the path to a file.
        If input is of type `bytes`, this is interpreted as the raw data itself.
    fmt: str
        DMAP format being read. One of `["dmap", "iqdat", "rawacf", "fitacf", "grid", "map", "snd"]`.
    mode: str
        Mode in which to read the data, either `strict`, `lax`, or `sniff`. In `strict` mode, any corruption
        in the data will raise an error. In `lax` mode, all valid records will be returned in a tuple along with
        the byte index of `source` where the corruption starts. In `sniff` mode, `source` must be a `str`, and
        only the first record will be read.

    Returns
    -------
    If `mode` is `strict`, returns `list[dict]` which is the parsed records.
    If `mode` is `lax`, returns `tuple[list[dict], Optional[int]]`, where the first element is the records which were parsed,
    and the second is the byte index where `source` was no longer a valid record of type `fmt`.
    If `mode` is `sniff`, returns `dict` of the first record.
    """
    if fmt not in ["dmap", "iqdat", "rawacf", "fitacf", "grid", "map", "snd"]:
        raise ValueError(
            f"invalid fmt `{fmt}`: expected one of ['dmap', 'iqdat', 'rawacf', 'fitacf', 'grid', 'map', 'snd']"
        )

    if mode not in ["strict", "lax", "sniff"]:
        raise ValueError(f"invalid mode `{mode}`: expected `strict`, `lax`, or `sniff`")

    if mode == "sniff" and not isinstance(source, str):
        raise TypeError(f"invalid type for `source` {type(source)} in `sniff` mode: expected `str`")
    
    if not isinstance(source, bytes) and not isinstance(source, str):
        raise TypeError(f"invalid type for `source` {type(source)}: expected `str` or `bytes`")

    # Construct the darn-dmap function name dynamically based on parameters:
    # fn_name = [sniff|read]_[fmt][_bytes][_lax]
    # All possibilites for, e.g., a FITACF file:
    #   read_fitacf
    #   read_fitacf_bytes
    #   read_fitacf_lax
    #   read_fitacf_bytes_lax
    #   sniff_fitacf
    fn_name = (
        f"{'sniff' if mode == 'sniff' else 'read'}"
        f"_{fmt}"
        f"{'_bytes' if isinstance(source, bytes) else ''}"
        f"{'_lax' if mode == 'lax' else ''}"
    )

    return getattr(dmap, fn_name)(source)


def write_dispatcher(source: list[dict], fmt: str, outfile: Union[None, str]) -> Union[None, bytes]:
    """
    Writes DMAP data from `source` to either a `bytes` object or to `outfile`.

    Parameters
    ----------
    source: list[dict]
        list of DMAP records as dictionaries.
    fmt: str
        DMAP format being read. One of `["dmap", "iqdat", "rawacf", "fitacf", "grid", "map", "snd"]`.
    outfile: Union[None, str]
        If `None`, returns the data as a `bytes` object. If this is a string, then this is interpreted as a path
        and data will be written to the filesystem. If the file ends in the `.bz2` extension, the data will be
        compressed using bzip2.
    """
    if fmt not in ["dmap", "iqdat", "rawacf", "fitacf", "grid", "map", "snd"]:
        raise ValueError(
            f"invalid fmt `{fmt}`: expected one of ['dmap', 'iqdat', 'rawacf', 'fitacf', 'grid', 'map', 'snd']"
        )
    if outfile is None:
        return getattr(dmap, f"write_{fmt}_bytes")(source)
    elif isinstance(outfile, str):
        getattr(dmap, f"write_{fmt}")(source, outfile)
    else:
        raise TypeError(f"invalid type for `outfile` {type(outfile)}: expected `str` or `None`")


def read_dmap(source: Union[str, bytes], mode: str = "lax") -> Union[dict, list[dict], tuple[list[dict], Optional[int]]]:
    """
    Reads in DMAP data from `source`.

    Parameters
    ----------
    source: Union[str, bytes]
        Where to read data from. If input is of type `str`, this is interpreted as the path to a file.
        If input is of type `bytes`, this is interpreted as the raw data itself.
    mode: str
        Mode in which to read the data, either "lax" (default), "strict", or "sniff". 
        In "lax" mode, all valid records will be returned in a tuple along with the byte index of `source` where the 
        corruption starts. 
        In "strict" mode, any corruption in the data will raise an error. 
        In "sniff" mode, `source` must be a path, and only the first record will be read.

    Returns
    -------
    If `mode` is `lax`, returns `tuple[list[dict], Optional[int]]`, where the first element is the records which were parsed,
    and the second is the byte index where `source` was no longer a valid record of type `fmt`.
    If `mode` is `strict`, returns `list[dict]` which is the parsed records.
    If `mode` is `sniff`, returns `dict`, which is the first record.
    """
    return read_dispatcher(source, "dmap", mode)


def read_iqdat(source: Union[str, bytes], mode: str = "lax") -> Union[dict, list[dict], tuple[list[dict], Optional[int]]]:
    """
    Reads in IQDAT data from `source`.

    Parameters
    ----------
    source: Union[str, bytes]
        Where to read data from. If input is of type `str`, this is interpreted as the path to a file.
        If input is of type `bytes`, this is interpreted as the raw data itself.
    mode: str
        Mode in which to read the data, either "lax" (default), "strict", or "sniff". 
        In "lax" mode, all valid records will be returned in a tuple along with the byte index of `source` where the 
        corruption starts. 
        In "strict" mode, any corruption in the data will raise an error. 
        In "sniff" mode, `source` must be a path, and only the first record will be read.

    Returns
    -------
    If `mode` is `lax`, returns `tuple[list[dict], Optional[int]]`, where the first element is the records which were parsed,
    and the second is the byte index where `source` was no longer a valid record of type `fmt`.
    If `mode` is `strict`, returns `list[dict]` which is the parsed records.
    If `mode` is `sniff`, returns `dict`, which is the first record.
    """
    return read_dispatcher(source, "iqdat", mode)


def read_rawacf(source: Union[str, bytes], mode: str = "lax") -> Union[dict, list[dict], tuple[list[dict], Optional[int]]]:
    """
    Reads in RAWACF data from `source`.

    Parameters
    ----------
    source: Union[str, bytes]
        Where to read data from. If input is of type `str`, this is interpreted as the path to a file.
        If input is of type `bytes`, this is interpreted as the raw data itself.
    mode: str
        Mode in which to read the data, either "lax" (default), "strict", or "sniff". 
        In "lax" mode, all valid records will be returned in a tuple along with the byte index of `source` where the 
        corruption starts. 
        In "strict" mode, any corruption in the data will raise an error. 
        In "sniff" mode, `source` must be a path, and only the first record will be read.

    Returns
    -------
    If `mode` is `lax`, returns `tuple[list[dict], Optional[int]]`, where the first element is the records which were parsed,
    and the second is the byte index where `source` was no longer a valid record of type `fmt`.
    If `mode` is `strict`, returns `list[dict]` which is the parsed records.
    If `mode` is `sniff`, returns `dict`, which is the first record.
    """
    return read_dispatcher(source, "rawacf", mode)


def read_fitacf(source: Union[str, bytes], mode: str = "lax") -> Union[dict, list[dict], tuple[list[dict], Optional[int]]]:
    """
    Reads in FITACF data from `source`.

    Parameters
    ----------
    source: Union[str, bytes]
        Where to read data from. If input is of type `str`, this is interpreted as the path to a file.
        If input is of type `bytes`, this is interpreted as the raw data itself.
    mode: str
        Mode in which to read the data, either "lax" (default), "strict", or "sniff". 
        In "lax" mode, all valid records will be returned in a tuple along with the byte index of `source` where the 
        corruption starts. 
        In "strict" mode, any corruption in the data will raise an error. 
        In "sniff" mode, `source` must be a path, and only the first record will be read.

    Returns
    -------
    If `mode` is `lax`, returns `tuple[list[dict], Optional[int]]`, where the first element is the records which were parsed,
    and the second is the byte index where `source` was no longer a valid record of type `fmt`.
    If `mode` is `strict`, returns `list[dict]` which is the parsed records.
    If `mode` is `sniff`, returns `dict`, which is the first record.
    """
    return read_dispatcher(source, "fitacf", mode)


def read_grid(source: Union[str, bytes], mode: str = "lax") -> Union[dict, list[dict], tuple[list[dict], Optional[int]]]:
    """
    Reads in GRID data from `source`.

    Parameters
    ----------
    source: Union[str, bytes]
        Where to read data from. If input is of type `str`, this is interpreted as the path to a file.
        If input is of type `bytes`, this is interpreted as the raw data itself.
    mode: str
        Mode in which to read the data, either "lax" (default), "strict", or "sniff". 
        In "lax" mode, all valid records will be returned in a tuple along with the byte index of `source` where the 
        corruption starts. 
        In "strict" mode, any corruption in the data will raise an error. 
        In "sniff" mode, `source` must be a path, and only the first record will be read.

    Returns
    -------
    If `mode` is `lax`, returns `tuple[list[dict], Optional[int]]`, where the first element is the records which were parsed,
    and the second is the byte index where `source` was no longer a valid record of type `fmt`.
    If `mode` is `strict`, returns `list[dict]` which is the parsed records.
    If `mode` is `sniff`, returns `dict`, which is the first record.
    """
    return read_dispatcher(source, "grid", mode)


def read_map(source: Union[str, bytes], mode: str = "lax") -> Union[dict, list[dict], tuple[list[dict], Optional[int]]]:
    """
    Reads in MAP data from `source`.

    Parameters
    ----------
    source: Union[str, bytes]
        Where to read data from. If input is of type `str`, this is interpreted as the path to a file.
        If input is of type `bytes`, this is interpreted as the raw data itself.
    mode: str
        Mode in which to read the data, either "lax" (default), "strict", or "sniff". 
        In "lax" mode, all valid records will be returned in a tuple along with the byte index of `source` where the 
        corruption starts. 
        In "strict" mode, any corruption in the data will raise an error. 
        In "sniff" mode, `source` must be a path, and only the first record will be read.

    Returns
    -------
    If `mode` is `lax`, returns `tuple[list[dict], Optional[int]]`, where the first element is the records which were parsed,
    and the second is the byte index where `source` was no longer a valid record of type `fmt`.
    If `mode` is `strict`, returns `list[dict]` which is the parsed records.
    If `mode` is `sniff`, returns `dict`, which is the first record.
    """
    return read_dispatcher(source, "map", mode)


def read_snd(source: Union[str, bytes], mode: str = "lax") -> Union[dict, list[dict], tuple[list[dict], Optional[int]]]:
    """
    Reads in SND data from `source`.

    Parameters
    ----------
    source: Union[str, bytes]
        Where to read data from. If input is of type `str`, this is interpreted as the path to a file.
        If input is of type `bytes`, this is interpreted as the raw data itself.
    mode: str
        Mode in which to read the data, either "lax" (default), "strict", or "sniff". 
        In "lax" mode, all valid records will be returned in a tuple along with the byte index of `source` where the 
        corruption starts. 
        In "strict" mode, any corruption in the data will raise an error. 
        In "sniff" mode, `source` must be a path, and only the first record will be read.

    Returns
    -------
    If `mode` is `lax`, returns `tuple[list[dict], Optional[int]]`, where the first element is the records which were parsed,
    and the second is the byte index where `source` was no longer a valid record of type `fmt`.
    If `mode` is `strict`, returns `list[dict]` which is the parsed records.
    If `mode` is `sniff`, returns `dict`, which is the first record.
    """
    return read_dispatcher(source, "snd", mode)


def write_dmap(source: list[dict], outfile: Union[None, str] = None) -> Union[None, bytes]:
    """
    Writes DMAP data from `source` to either a `bytes` object or to `outfile`.

    Parameters
    ----------
    source: list[dict]
        list of DMAP records as dictionaries.
    outfile: Union[None, str]
        If `None`, returns the data as a `bytes` object. If this is a string, then this is interpreted as a path
        and data will be written to the filesystem. If the file ends in the `.bz2` extension, the data will be
        compressed using bzip2.
    """
    return write_dispatcher(source, "dmap", outfile)


def write_iqdat(source: list[dict], outfile: Union[None, str] = None) -> Union[None, bytes]:
    """
    Writes IQDAT data from `source` to either a `bytes` object or to `outfile`.

    Parameters
    ----------
    source: list[dict]
        list of IQDAT records as dictionaries.
    outfile: Union[None, str]
        If `None`, returns the data as a `bytes` object. If this is a string, then this is interpreted as a path
        and data will be written to the filesystem. If the file ends in the `.bz2` extension, the data will be
        compressed using bzip2.
    """
    return write_dispatcher(source, "iqdat", outfile)


def write_rawacf(source: list[dict], outfile: Union[None, str] = None) -> Union[None, bytes]:
    """
    Writes RAWACF data from `source` to either a `bytes` object or to `outfile`.

    Parameters
    ----------
    source: list[dict]
        list of RAWACF records as dictionaries.
    outfile: Union[None, str]
        If `None`, returns the data as a `bytes` object. If this is a string, then this is interpreted as a path
        and data will be written to the filesystem. If the file ends in the `.bz2` extension, the data will be
        compressed using bzip2.
    """
    return write_dispatcher(source, "rawacf", outfile)


def write_fitacf(source: list[dict], outfile: Union[None, str] = None) -> Union[None, bytes]:
    """
    Writes FITACF data from `source` to either a `bytes` object or to `outfile`.

    Parameters
    ----------
    source: list[dict]
        list of FITACF records as dictionaries.
    outfile: Union[None, str]
        If `None`, returns the data as a `bytes` object. If this is a string, then this is interpreted as a path
        and data will be written to the filesystem. If the file ends in the `.bz2` extension, the data will be
        compressed using bzip2.
    """
    return write_dispatcher(source, "fitacf", outfile)


def write_grid(source: list[dict], outfile: Union[None, str] = None) -> Union[None, bytes]:
    """
    Writes GRID data from `source` to either a `bytes` object or to `outfile`.

    Parameters
    ----------
    source: list[dict]
        list of GRID records as dictionaries.
    outfile: Union[None, str]
        If `None`, returns the data as a `bytes` object. If this is a string, then this is interpreted as a path
        and data will be written to the filesystem. If the file ends in the `.bz2` extension, the data will be
        compressed using bzip2.
    """
    return write_dispatcher(source, "grid", outfile)


def write_map(source: list[dict], outfile: Union[None, str] = None) -> Union[None, bytes]:
    """
    Writes MAP data from `source` to either a `bytes` object or to `outfile`.

    Parameters
    ----------
    source: list[dict]
        list of MAP records as dictionaries.
    outfile: Union[None, str]
        If `None`, returns the data as a `bytes` object. If this is a string, then this is interpreted as a path
        and data will be written to the filesystem. If the file ends in the `.bz2` extension, the data will be
        compressed using bzip2.
    """
    return write_dispatcher(source, "map", outfile)


def write_snd(source: list[dict], outfile: Union[None, str] = None) -> Union[None, bytes]:
    """
    Writes SND data from `source` to either a `bytes` object or to `outfile`.

    Parameters
    ----------
    source: list[dict]
        list of SND records as dictionaries.
    outfile: Union[None, str]
        If `None`, returns the data as a `bytes` object. If this is a string, then this is interpreted as a path
        and data will be written to the filesystem. If the file ends in the `.bz2` extension, the data will be
        compressed using bzip2.
    """
    return write_dispatcher(source, "snd", outfile)
