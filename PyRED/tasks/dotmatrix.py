import os
import copy
import time

import numpy

from PyRED.files import read_behaviour
from PyRED.tasks.generic import TaskLoader


class DotMatrixLoader(TaskLoader):
    
    """Class to process files from the RED Dot Matrix task.
    """
    
    def __init__(self, data_dir, output_path=None, task_name="Dot_Matrix"):
        
        """Initialises a new DotMatrixLoader instance to read and process data
        from files generated by the RED dot matrix task.
        
        Arguments
        
        data_dir            -   String. Path to the directory that contains
                                data files that need to be loaded.

        Keyword Arguments
        
        output_path         -   String. Path to the file in which processed
                                data needs to be stored, or None to not write
                                the data to file. Default = None

        task_name           -   String. Name of the task that needs to be
                                present in the data files. The names of data
                                files are assumed to be in the format
                                "taskname_ppname_yyyy-mm-dd-HH-MM-SS"
                                Default = "Dot_Matrix"

        """
        
        # Load all data.
        self.load_from_directory(data_dir, task_name)
        self.process_raw_data()
        if not (output_path is None):
            self.write_processed_data_to_file(output_path)

    

    def load_from_file(self, file_path, delimiter=",", missing=None, \
            auto_typing=True, string_vars=None):
        
        """Loads data from a single file. This function overwrites the parent's
        load_from_file function to allow for the checking of answers.

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
        
        # Load the data from a file.
        raw = read_behaviour(file_path, delimiter=",", missing=None, \
            auto_typing=True, string_vars=None)
        
        # If the file is empty, return None.
        if raw is None:
            return None
        
        return raw
    
    
    def process_raw_data(self):
        
        """Computes the variables that need to be computed from this task, and
        stores them in the self.data dict. This has one key for every variable
        of interest, and each of these keys points to a NumPy array with shape
        (N,) where N is the number of participants.
        
        The processed data comes from the self.raw dict, so make sure that
        self.load_from_directory is run before this function is.
        """
        
        # Define some variables of interest.
        vor = ["n_items", "n_correct", "p_correct", "span", "median_RT", \
            "mean_RT", "stdev_RT", "scaled_stdev_RT"]
        
        # Get all participant names, or return straight away if no data was
        # loaded yet.
        if hasattr(self, "raw"):
            participants = self.raw.keys()
            participants.sort()
        else:
            self.data = None
            return

        # Count the number of participants.
        n = len(participants)
        
        # Create a data dict for each variable of interest.
        self.data = {}
        self.data["ppname"] = []
        for var in vor:
            self.data[var] = numpy.zeros(n, dtype=float) * numpy.NaN
        
        # Loop through all participants.
        for i, ppname in enumerate(participants):
            # Add the participant name.
            self.data["ppname"].append(copy.deepcopy(ppname))
            # Skip empty datasets.
            if self.raw[ppname] is None:
                continue
            # Compute stuff relevant to this task.
            self.data["n_items"][i]     = len(self.raw[ppname]["Trial"])
            self.data["n_correct"][i]   = numpy.sum(self.raw[ppname]["Accuracy"])
            self.data["p_correct"][i]   = float(self.data["n_correct"][i]) \
                / float(self.data["n_items"][i])
            self.data["span"][i]        = numpy.nanmax(self.raw[ppname]["Span"])
            self.data["median_RT"][i]   = numpy.nanmedian(self.raw[ppname]["RT"])
            self.data["mean_RT"][i]     = numpy.nanmean(self.raw[ppname]["RT"])
            self.data["stdev_RT"][i]    = numpy.nanstd(self.raw[ppname]["RT"])
        # Compute a scaled standard deviation of the response time, scaled to the
        # median response time to remove the correlation between the two.
        self.data["scaled_stdev_RT"] = self.data["stdev_RT"] / self.data["median_RT"]
