import os

import numpy


def read_behaviour(fpath, delimiter=",", missing=None, auto_typing=True, \
    string_vars=None):
    
    """Reads a text file, and returns a dict with parsed data. The dict will
    contain one key for every variable, which will point to a NumPy array
    with one entry for every trial.
    
    Arguments
    
    fpath           -   String. Full path to a data file.
    
    Keyword Arguments
    
    delimiter       -   String. Delimiter for the data file. Default = ","
    
    missing         -   List. List of values that code for missing data, or
                        None if no such values exist. Note that all values
                        should be strings, as this is what the data will
                        initially be read as (missing data is converted before
                        auto-typing occurs). Default = None.
    
    auto_typing     -   Bool. When True, variables will automatically be
                        converted to float where possible. Default = True
    
    string_vars     -   List. List of variable names that should not be
                        converted to numbers, but should remain string; or
                        None to auto-convert when possible. Default = None
    
    Returns

    data            -   Dict. One key for every variable, each pointing to a 
                        NumPy array with shape=(N,) where N is the number of
                        trials. Can also return None if the data file was
                        empty, or when it only contained a header.
    """
    
    # Convert None input arguments to empty lists where necessary. (This is
    # so that 'in' statements and for loops will work.)
    if missing is None:
        missing = []
    if string_vars is None:
        string_vars = []
    
    # Check whether the file exists.
    if not os.path.isfile(fpath):
        raise Exception("ERROR: File does not exist at path '%s'" % (fpath))
    
    # Read the raw data.
    if ("Hearts" in os.path.basename(fpath)) or (("Flowers" in os.path.basename(fpath))):
        # More robust file reading, because I fucked it up in Unity.
        with open(fpath, "r") as f:
            lines = f.readlines()
        # Return nothing if nothing or only the header was logged.
        if len(lines) < 2:
            return None
        # Grab the header.
        header = lines.pop(0)
        header = header.replace("\n","").split(",")
        # Create an empty NumPy array to hold data.
        raw = numpy.zeros((len(header),len(lines)+1), dtype="|S20")
        raw[:,0] = numpy.array(header)
        # Go through all the lines.
        for i, l in enumerate(lines):
            # String to list.
            l = l.replace("\n","").split(",")
            # Check if the length of the lines matches the length of the
            # header. If it's off by -1, it's because the RT wasn't logged in
            # some cases.
            if len(l) == len(header) - 1:
                # Add "None" for the RT.
                l.insert(header.index("RT"), "NaN")
            # Convert "None" to "NaN" in RT column, so it will be picked up as
            # missing automatically.
            if l[header.index("RT")] == "None":
                l[header.index("RT")] = "NaN"
            # Save the data.
            raw[:,i+1] = numpy.array(l)
#
#        try:
#            raw = numpy.loadtxt(fpath, delimiter=delimiter, unpack=True, dtype=str, skiprows=1)
#        except StopIteration:
#            return None
#        header = ["phase", "trial", "stimulus", "stim_location", "pressed", "RT", "correct", "timeout"]
#        if raw.shape[0] == 7:
#            header.remove("RT")
#        header = numpy.array(header, dtype=raw.dtype).reshape(raw.shape[0],1)
#        raw = numpy.hstack((header, raw))

    else:
        raw = numpy.loadtxt(fpath, delimiter=delimiter, unpack=True, dtype=str)
    
    # Check if the file is completely empty.
    if raw.shape[0] == 0:
        print("WARNING: Empty file at path '%s'" % (fpath))
        return None

    # Check if the file only has a header.
    elif len(raw.shape) == 1:
        print("WARNING: Header-only file at path '%s'" % (fpath))
        return None
    
    # Create an empty dict to store the data in.
    data = {}
    # Count the number of variables and the number of trials.
    n_vars, n_trials = raw.shape
    n_trials -= 1
    # Parse the raw data.
    for i in range(n_vars):
        # Get the variable name (first row in this column), and the values
        # for each trial (rest of the rows).
        var = raw[i,0]
        val = raw[i,1:]
        # Convert missing values into NaN by looping through each missing
        # value indicator, and converting it into a NaN. The string "NaN" is
        # recognised as the float numpy.NaN when calling val.astype(float),
        # so our auto-typing should work after replacing missing with "NaN".
        for m in missing:
            # Compare the read values (strings) against the stringified
            # missing value.
            val[val==str(m)] = "NaN"
            # NOTE: This can generate a future warning in NumPy version 1.14.5,
            # as the comparison isn't clear in strict Pythonic interpretation,
            # i.e. should it return a single Boolean (False, because the
            # NumPy array and the string m are not the same), or should it
            # return a NumPy array of Booleans (True for elements that are
            # equal to m and False when they are not.) The option below is
            # future proof. Uncomment if necessary:
            #val[val == numpy.array(val.shape[0]*[m], dtype=val.dtype)] = "NaN"
        # Optionally auto-detect the type.
        if auto_typing and (var not in string_vars):
            try:
                val = val.astype(float)
            except:
                pass
        # Store the variable name and values in the data dict.
        data[var] = numpy.copy(val)
    
    return data


def rename_data_files(dir_path, old_name, new_name, recursive=True):
    
    """Renames files in dir_path (optionally recursively). This function
    replaces a component of a file name (old_name) with a different component
    (new_name), e.g. to turn all files with "Q1.questionnaire" in their name
    to files with "Q1_questionnaire" in their name.
    
    Arguments
    
    dir_path        -   String. Full path to a directory.
    
    old_name        -   String. Should occur within the file names that you
                        aim to change, but does not have to be the whole file
                        name.
    
    new_name        -   String. Replaces occurences of old_name in file names
                        where old_name occurs.
    
    Keyword Arguments
    
    recursive       -   Bool. True if this function should work on files in
                        dir_path and all its sub-directories (and their sub-
                        directories, and so on), or False if only on files
                        occuring in dir_path should be changed. Default = True
    """

    # Create a list of all files in the directory.
    all_files = []
    if recursive:
        for dp, dn, filenames in os.walk(dir_path):
            for fname in filenames:
                all_files.append(os.path.join(dp, fname))
    else:
        for fname in os.listdir(dir_path):
            all_files.append(os.path.join(dir_path, fname))
    
    # Loop through all files, and rename where necessary.
    for i, fpath in enumerate(all_files):
        # Split path and file name.
        fname = os.path.basename(fpath)
        name, ext = os.path.splitext(fname)
        # Check if the target file name is in the current file name.
        if old_name in name:
            # Replace the target string within the file name.
            name = name.replace(old_name, new_name)
            # Construct the new path.
            new_path = os.path.join(os.path.dirname(fpath), name+ext)
            # Rename the file.
            os.rename(fpath, new_path)


def combine_data_files(path_list, output_path, pp_key="ppname", delimiter=",", \
    exclude=None):
    
    """Combines the data from several files into one BIG DATA file. Data files
    should be formatted with a single header row that contains variable names,
    and with each further row representing the data of 1 participant. Each
    column should be a variable. Data should be separated by a delimiter;
    which one is up to the user (see delimiter keyword).
    
    Arguments
    
    path_list       -   List. List of full paths to all data files that
                        should be merged.

    output_path     -   String. Full path to the output file. If a file exists
                        at this path, it will be overwritten. If a file does
                        not exist, it will be created.
    
    Keyword Arguments
    
    pp_key          -   String. Variable name for the participant name or
                        number. This should be the same across all files.
                        Default = "ppname"

    delimiter       -   String. Delimiter for the data file. Default = ","
    
    exclude         -   List. List of strings of participant names/numbers
                        who should be excluded from the combined data file.
                        Can also be None to not exclude any participant.
                        Default = None
    """
    
    # Loop through all files.
    all_data = {}
    for fpath in path_list:
        # Get the name of the current file.
        name, ext = os.path.splitext(os.path.basename(fpath))
        # Read the data, and add it to the dict. Note that this is completely
        # agnostic about what type the data is; everything is considered a
        # string to not mess with any formatting.
        all_data[name] = read_behaviour(fpath, delimiter=delimiter, \
            missing=None, auto_typing=False, string_vars=None)
    
    # Find all participant and variable names.
    all_participants = []
    all_variables = {}
    for task_name in all_data.keys():
        # Add the participants to the list.
        all_participants.extend(all_data[task_name][pp_key])
        # Add all variables in this task to the list.
        all_variables[task_name] = all_data[task_name].keys()
        all_variables[task_name].remove(pp_key)
    # Remove double entries in the participant list.
    ppnames = numpy.sort(numpy.unique(all_participants))
    n_participants = len(ppnames)
    
    # Create a dict with a key for variable, pointing to a NumPy array with
    # shape (n_participants,).
    data = {}
    data[pp_key] = ppnames
    for task_name in all_variables.keys():
        for var_name in all_variables[task_name]:
            # Create a new entry for this variable.
            data["%s-%s" % (task_name.upper(), var_name)] = \
                numpy.zeros(n_participants, dtype="|S100")
            # Loop through all participants.
            for i, ppname in enumerate(ppnames):
                # Find whether the participant had data in this task.
                sel = numpy.where(all_data[task_name][pp_key] == ppname)[0]
                # Add the data to the dict.
                if len(sel) == 1:
                    data["%s-%s" % (task_name.upper(), var_name)][i] = \
                        all_data[task_name][var_name][sel[0]]
                elif len(sel) > 1:
                    raise Exception("ERROR: Duplicate entry in task '%s' for participant '%s'" \
                        % (task_name, ppname))
                else:
                    data["%s-%s" % (task_name.upper(), var_name)][i] = "NaN"
    
    # Optionally exclude participants.
    excluded = []
    if exclude is not None:
        for ppname in data[pp_key]:
            if ppname in exclude:
                excluded.append(ppname)
    
    # Write data to file.
    with open(output_path, "w") as f:
        # Create a header.
        header = data.keys()
        header.remove(pp_key)
        header.sort()
        header.insert(0, pp_key)
        # Write header to file.
        f.write(delimiter.join(map(str, header)))
        # Loop through all participants.
        for i, ppname in enumerate(data[pp_key]):
            if ppname in excluded:
                print("Excluding participant '%s'" % (ppname))
                continue
            line = []
            for var in header:
                line.append(data[var][i])
            f.write("\n" + delimiter.join(map(str, line)))
    