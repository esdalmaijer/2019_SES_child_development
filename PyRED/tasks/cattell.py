import os
import copy

import numpy

from PyRED.files import read_behaviour
from PyRED.tasks.generic import TaskLoader


class CattellLoader(TaskLoader):
    
    """Class to process files from the RED Cattell task.
    """
    
    def __init__(self, data_dir, output_path=None, task_name="cattell_test", \
            answer_file=None):
        
        """Initialises a new CattellLoader instance to read and process data
        from files generated by the RED fluid reasoning task.
        
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
                                Default = "cattell_test"

        answer_file         -   String. Path to the file where the task's
                                answers are defined, or None to use the answer
                                sheet that is included with this library.
                                Default = None
        """
        
        # Define the default answer_file.
        if answer_file is None:
            answer_file = os.path.join( \
                os.path.dirname(os.path.abspath(__file__)), \
                "cattell_test_answers.txt")
        # Throw an error if the answer sheet doesn't exist at the expected 
        # location.
        if not os.path.isfile(answer_file):
            raise Exception("ERROR: Could not find the answer file at '%s'" % \
                (answer_file))
        # Load the answers from a text file.
        raw = numpy.loadtxt(answer_file, dtype=str, delimiter=',', unpack=True)
        # Put the answers in a dict (two keys: one for sentences and one for
        # the associated correct responses).
        r = {}
        for i in range(len(raw)):
            r[raw[i][0]] = raw[i][1:]
        # Put the answers in a dict with one key for every sentence, pointing
        # to the correct response for that item.
        self._answers = {}
        for i in range(len(r["item"])):
            self._answers[r["item"][i]] = r["answer"][i]
        
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
            auto_typing=True, string_vars=["item", "event", "option"])
        
        # If the file is empty, return None.
        if raw is None:
            return None
        
        # Loop through every index in the raw data, as each row in the data
        # file represented an event rather than a trial. Hence, we will need
        # to figure out what happened in each trial, i.e. what performance,
        # response times, and the number of re/de-selects was like.
    
        # Loop through the rest of the file. We start counting at 1, as the
        # header is at index 0.
        current_item = None
        item_start = 0.0
        itemstats = {"selects":0, "deselects":0}
        ppstats = {"items":[], "rts":[], "selects":[], "correct":[]}
        for i in range(len(raw["time"])):
    
            # Get the current data.
            timestamp = raw["time"][i]
            item = raw["item"][i]
            event = raw["event"][i]
            option = raw["option"][i].replace("Option ", "")
            
            # Set the new current item if the current item is None.
            if current_item is None:
                # Set the current item.
                current_item = copy.deepcopy(item)
            # If the current item somehow doesn't overlap with the item in this
            # event, something weird happened!
            elif item != current_item:
                current_item = copy.deepcopy(item)
                item_start = copy.deepcopy(timestamp)
                rt = numpy.NaN
            
            # Handle the input according to the event.
            if event == 'confirm':
                # Compute the response time.
                rt = timestamp - item_start
                # Compare the option against the answers.
                correct = self._answers[item] == option
                # Store the same data in a temporary dict.
                if 'example' not in current_item:
                    ppstats['items'].append(copy.deepcopy(item))
                    ppstats['rts'].append(copy.deepcopy(rt))
                    ppstats['correct'].append(copy.deepcopy(correct))
                    ppstats['selects'].append(copy.deepcopy(itemstats['selects']))
                # Unset the current item.
                current_item = None
                # Log the start of this trial as the confirmation of the previous.
                item_start = copy.deepcopy(timestamp)
                # Reset the item stats.
                itemstats = {'selects':0, 'deselects':0}
    
            elif event == 'select':
                itemstats['selects'] += 1
    
            elif event == 'deselect':
                itemstats['deselects'] += 1
        
        # Store the individual item responses in a condensed raw dict.
        condensed = {}
        condensed["item"] = numpy.array(ppstats["items"])
        condensed["RT"] = numpy.array(ppstats["rts"], dtype=float)
        condensed["correct"] = numpy.array(ppstats["correct"], dtype=bool)
        condensed["n_selections"] = numpy.array(ppstats["selects"], dtype=int)
                
        return condensed
    
    
    def process_raw_data(self):
        
        """Computes the variables that need to be computed from this task, and
        stores them in the self.data dict. This has one key for every variable
        of interest, and each of these keys points to a NumPy array with shape
        (N,) where N is the number of participants.
        
        The processed data comes from the self.raw dict, so make sure that
        self.load_from_directory is run before this function is.
        """
        
        # TODO: Also include data on a per-item level. (Might be nice for
        # more detailed questions on fluid reasoning.)
        
        # Define some variables of interest.
        vor = ["n_items", "n_correct", "p_correct", "mean_n_selections", \
            "median_RT", "mean_RT", "stdev_RT", "scaled_stdev_RT"]
        
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
            self.data["n_items"][i] = len(self.raw[ppname]["item"])
            self.data["n_correct"][i]   = numpy.sum(self.raw[ppname]["correct"])
            self.data["p_correct"][i]   = float(self.data["n_correct"][i]) \
                / float(self.data["n_items"][i])
            self.data["mean_n_selections"][i] = numpy.nanmean(self.raw[ppname]["n_selections"])
            self.data["median_RT"][i]   = numpy.nanmedian(self.raw[ppname]["RT"])
            self.data["mean_RT"][i]     = numpy.nanmean(self.raw[ppname]["RT"])
            self.data["stdev_RT"][i]    = numpy.nanstd(self.raw[ppname]["RT"])
        # Compute a scaled standard deviation of the response time, scaled to the
        # median response time to remove the correlation between the two.
        self.data["scaled_stdev_RT"] = self.data["stdev_RT"] / self.data["median_RT"]
