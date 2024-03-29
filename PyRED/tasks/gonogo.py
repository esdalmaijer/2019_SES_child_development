import os
import copy
import time

import numpy
from scipy.stats import norm

from PyRED.files import read_behaviour
from PyRED.tasks.generic import TaskLoader


class GoNoGoLoader(TaskLoader):
    
    """Class to process files from the RED Go/NoGo task.
    """
    
    def __init__(self, data_dir, output_path=None, task_name="GoNoGoTask"):
        
        """Initialises a new GoNoGoLoader instance to read and process data
        from files generated by the RED go / no go task.
        
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
                                Default = "GoNoGoTask"

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
            auto_typing=True, string_vars=["stimtype", "timeout"])
        # If the file is empty, return None.
        if raw is None:
            return None
        
        # Convert the timeout to Booleans.
        raw["timeout"] = raw["timeout"] == "True"
        
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
        vor = ["hits", "misses", "false_alarms", "correct_rejections", \
            "hit_rate", "miss_rate", "FA_rate", "CR_rate", "dprime", "beta", \
            "criterion", "Ad", "RT_hit", "RT_FA", "RT_SD_hit", "RT_SD_FA", \
            "RT_SD_scaled_hit", "RT_SD_scaled_FA", "points", "duration"]
        
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

            # Select only data from the first four minutes.
            sel = self.raw[ppname]["time"] <= 240.0
            # Count the number of target and distractor trials.
            n_targets = numpy.sum( \
                (self.raw[ppname]["stimtype"] == "target").astype(int))
            n_distractors = numpy.sum( \
                (self.raw[ppname]["stimtype"] == "distractor").astype(int))
            # Compute stuff relevant to this task.
            h =  (self.raw[ppname]["timeout"][sel] == False) & \
                (self.raw[ppname]["stimtype"][sel] == "target")
            m =  (self.raw[ppname]["timeout"][sel] == True)  & \
                (self.raw[ppname]["stimtype"][sel] == "target")
            cr = (self.raw[ppname]["timeout"][sel] == True)  & \
                (self.raw[ppname]["stimtype"][sel] == "distractor")
            fa = (self.raw[ppname]["timeout"][sel] == False) & \
                (self.raw[ppname]["stimtype"][sel] == "distractor")
            self.data["hits"][i] = numpy.sum(h.astype(int))
            self.data["misses"][i] = numpy.sum(m.astype(int))
            self.data["false_alarms"][i] = numpy.sum(fa.astype(int))
            self.data["correct_rejections"][i] = numpy.sum(cr.astype(int))
            # Compute rates.
            if n_targets > 0:
                self.data["hit_rate"][i] = float(self.data["hits"][i]) \
                    / float(n_targets)
                self.data["miss_rate"][i] = float(self.data["misses"][i]) \
                    / float(n_targets)
            if n_distractors > 0:
                self.data["FA_rate"][i] = float(self.data["false_alarms"][i]) \
                    / float(n_distractors)
                self.data["CR_rate"][i] = float(self.data["correct_rejections"][i]) \
                    / float(n_distractors)
            # Prevent inf in computations.
            if self.data["hit_rate"][i] == 1:
                hit_rate = self.data["hit_rate"][i] - (0.5 / float(self.data["hits"][i]+self.data["misses"][i]))
            elif self.data["hit_rate"][i] == 0:
                hit_rate = 0.5 / float(self.data["hits"][i]+self.data["misses"][i])
            else:
                hit_rate = self.data["hit_rate"][i]
            if self.data["FA_rate"][i] == 1:
                fa_rate = self.data["FA_rate"][i] - (0.5 / float(self.data["correct_rejections"][i]+self.data["false_alarms"][i]))
            elif self.data["FA_rate"][i] == 0:
                fa_rate = 0.5 / float(self.data["correct_rejections"][i]+self.data["false_alarms"][i])
            else:
                fa_rate = self.data["FA_rate"][i]
            # Compute signal-detection theory metrics.
            self.data["dprime"][i] = norm.ppf(hit_rate) - norm.ppf(fa_rate)
            self.data["beta"][i] = numpy.exp( (norm.ppf(fa_rate)**2 - norm.ppf(hit_rate)**2) / 2.0)
            self.data["criterion"][i] = - (norm.ppf(hit_rate) + norm.ppf(fa_rate)) / 2.0
            self.data["Ad"][i] = norm.cdf(self.data["dprime"][i] / numpy.sqrt(2))
            # Compute response times.
            self.data["RT_hit"][i] = numpy.nanmedian(self.raw[ppname]["RT"][sel][h])
            self.data["RT_SD_hit"][i] = numpy.nanstd(self.raw[ppname]["RT"][sel][h])
            self.data["RT_FA"][i] = numpy.nanmedian(self.raw[ppname]["RT"][sel][fa])
            self.data["RT_SD_FA"][i] = numpy.nanstd(self.raw[ppname]["RT"][sel][fa])
            self.data["points"][i] = self.raw[ppname]["pointtotal"][sel][-1]
            self.data["duration"][i] = self.raw[ppname]["time"][sel][-1]

        # Compute a scaled standard deviation of the response time, scaled to the
        # median response time to remove the correlation between the two.
        self.data["RT_SD_scaled_hit"] = self.data["RT_SD_hit"] / self.data["RT_hit"]
        self.data["RT_SD_scaled_FA"] = self.data["RT_SD_FA"] / self.data["RT_FA"]
