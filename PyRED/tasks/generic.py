import os
import copy

import numpy

from PyRED.files import read_behaviour


class TaskLoader:
    """A generic class to define some of the general functionality used across
    the different RED task-specific loaders.
    """
    
    def __init__(self):
        
        """Overwrite the __init__ function with your own, unless nothing needs
        to be initialised.
        """

        # No NotImplementedError should be raised here, as it is a perfectly
        # legitimate option to not implement an __init__ function in a child
        # class, for example when no preparation needs to be done.
        return

    
    def load_from_file(self, file_path, delimiter=",", missing=None, \
            auto_typing=True, string_vars=None):
        
        """Loads data from a single file. Overwrite this function in a child
        class to allow for custom processing of data after it has been loaded.

        Arguments
        
        file_path       -   String. Path to the file that needs to be loaded.
        
        Keyword arguments
        
        delimiter       -   String. Delimiter for the data file. Default = ","
        
        missing         -   List. List of values that code for missing data, or
                            None if no such values exist. Note that all values
                            should be strings, as this is what the data will
                            initially be read as (missing data is converted before
                            auto-typing occurs). Default = None.
        
        auto_typing     -   Bool. When True, variables will automatically be
                            converted to float where possible. Default = True
        
        Returns
        
        data            -   Whatever PyRED.files.read_behaviour returns.
        """
        
        return read_behaviour(file_path, delimiter=delimiter, \
            missing=missing, auto_typing=auto_typing, string_vars=string_vars)

    
    def load_from_directory(self, dir_path, task_name, recursive=True, \
            delimiter=",", missing=None, auto_typing=True, string_vars=None, \
            callback=None, callback_args=None, callback_kwargs=None):
        
        """Loads the files from which the name contains a particular string,
        and adds them to an internal dict of loaded data. Participant names
        and testing dates are automatically recognised from the file name.
        
        Arguments
        
        dir_path        -   String. Path to the directory that contains files
                            that need to be loaded.
        
        task_name       -   String. Name of the task that needs to be present
                            in the data files. The names of data files are
                            assumed to be in the format
                            "taskname_ppname_yyyy-mm-dd-HH-MM-SS"
        
        Keyword arguments
        
        recursive       -   Bool. Indicates whether files in sub-directories 
                            should be loaded too. Default = True
        
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
        
        callback        -   Function. Pass a function to be run directly after
                            loading the data, for example to check whether
                            participants answered correctly against an external
                            source of data. Can also be None, if no callback
                            is required. Default = None.
        
        callback_args   -   List. Passed as arguments to the callback function.
                            Can also be None, if no arguments need to be
                            passed. Default = None.
        
        callback_kwargs -   Dict. Passed as keyword arguments to the callback
                            function. Can also be None, if no arguments need
                            to be passed. Default = None.
        """
        
        # Check if a data dict already exists within this instance.
        if not hasattr(self, "raw"):
            # Create a new data dict.
            self.raw = {}
        
        # Check if the directory exists.
        if not os.path.isdir(dir_path):
            raise Exception("ERROR: Directory not found at '%s'" % (dir_path))
        
        # Create a list of all files in the directory.
        all_files = []
        if recursive:
            for dp, dn, filenames in os.walk(dir_path):
                for fname in filenames:
                    all_files.append(os.path.join(dp, fname))
        else:
            for fname in os.listdir(dir_path):
                all_files.append(os.path.join(dir_path, fname))
        
        # Loop through the files.
        for i, fpath in enumerate(all_files):
            
            # Get the name of the file.
            fname = os.path.basename(fpath)
            
            # Skip non-files.
            if not os.path.isfile(fpath):
                continue

            # Check whether the current file is likely from the intended task.
            if task_name not in fname:
                # Skip to the next file.
                continue

            # Parse the file name. The first step is to separate the name from
            # the file extension.
            name, ext = os.path.splitext(fname)
            # The second step is to grab the date of the name.
            testdate = copy.deepcopy(name[-19:-9])
            testtime = copy.deepcopy(name[-8:])
            # The third step is to find the participant name by finding the
            # last occuring underscore ("_") in the remaining name after
            # removing the test date.
            # Remove the task date and time.
            name = name[:-20]
            # Find the first underscore position in the reversed name.
            i = name[-1::-1].find("_")
            # Separate the participant name and the task name.
            ppname = copy.deepcopy(name[-i:])
            taskname = name[:-i-1]

            # Check if this is indeed the intended task name.
            if taskname != task_name:
                continue
            
            # Load the data via a custom file read function.
            self.raw[ppname] = self.load_from_file(fpath, \
                delimiter=delimiter, missing=missing, \
                auto_typing=auto_typing, string_vars=string_vars)
        
        # Run the callback function, if one was passed.
        if callback is None:
            return
        else:
            if callback_args is None:
                callback_args = []
            callback(*callback_args, **callback_kwargs)
    

    def process_raw_data(self):
        
        """Overwrite the process_raw_data function with your own.
        """
        
        raise NotImplementedError("generic.TaskLoader.process_raw_data needs to be overwritten in a child class")
    
    
    def write_processed_data_to_file(self, file_path, pp_key="ppname"):
        
        """Writes all processed data (stored in the self.data dict to a file.
        
        Arguments
        
        file_path       -   String. Path that data needs to be written to.
        
        Keyword Arguments
        
        pp_key          -   String. Key that points to participant names in
                            the self.data dict. Default = "ppname"
        """
        
        # Check if processed data exists.
        if not hasattr(self, "data"):
            raise Exception("ERROR: No data was loaded or processed yet. Make sure to call load_from_directory and process_raw_data first")
        
        # Write an empty file if no data was loaded.
        if self.data is None:
            f = open(file_path, "w")
            f.close()
        
        # Get all variables.
        header = self.data.keys()
        # Check if the participant key is present in the variables.
        if pp_key not in header:
            raise Exception("ERROR: Participant key '%s' does not appear in the self.data dict" \
                % (pp_key))
        # Temporarily remove the key from the list.
        else:
            header.remove(pp_key)
        # Sort the variables alphabetically.
        header.sort()
        # Add the participant key at the start of the variable list.
        header.insert(0, pp_key)
        
        # Attempt to autodetect the type of file.
        name, ext = os.path.splitext(os.path.basename(file_path))
        if ext in [".txt", ".tsv"]:
            sep = "\t"
        elif ext in [".csv"]:
            sep = ","
        else:
            sep = ","
        
        # Open a new file.
        with open(file_path, "w") as f:
            # Write a header to the file.
            f.write(sep.join(map(str, header)))
            # Loop through all participants.
            for i, ppname in enumerate(self.data[pp_key]):
                # Get all data for this participant.
                line = []
                for var in header:
                    line.append(self.data[var][i])
                # Write this participant's data to the file.
                f.write("\n" + sep.join(map(str, line)))
