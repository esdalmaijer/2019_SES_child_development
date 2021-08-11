import os
import copy
import random

import numpy
import scipy.stats
import matplotlib
from matplotlib import pyplot
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn import decomposition
from sklearn.cluster import DBSCAN

from factor_analyzer import FactorAnalyzer

from PyRED.cluster import clustering, cluster_comparison, convenience_clustering, \
    correlation_matrix, dim_reduction, plot_averages, plot_clusters, \
    plot_samples, plot_silhouette, preprocess
from PyRED.graph import create_graph, partial_corr, plot_correlation_matrix, \
    plot_graph


# Set options for what to run.
DO_PCA = True
DO_FACTOR = True
DO_CLUSTERING = True
DO_NETWORK = True
OVERWRITE_NETWORK_BOOTSTRAP = False
N_BOOTSTRAP_ITERATIONS = 3000
BOOTSTRAPPED_CI_ALPHA = 0.05
DO_NETWORK_CLUSTERING = True

# Set the random seed.
numpy.random.seed(19)

# This is the number of factors that is to be extracted from the data. This
# is a predefined value, as it is expected you have inspected the outcome of
# the parallel analysis previously, and have decided on its basis.
PARALLEL_N_FACTORS = 5
# Set the exact type of analysis: "PCA" (principle component analysis) or "FA"
# (factor analysis).
PARALLEL_FACTOR_TYPE = "FA"
# The number of iterations in the parallel analysis.
PARALLEL_N_PERMUTATIONS = 1000
# Set to True to permute within each feature in the parallel analysis, and to
# False to sample from standard normal distributions.
PARALLEL_PERMUTE = True
# Set to False to overwrite existing parallel analysis data.
PARALLEL_OVERWRITE_EXISTING = False

# Set the clustering methods.
CLUSTER_DIM_REDUCTION = "MDS"
CLUSTER_METHOD = "KMEANS"
# Set the network clustering methods.
# (Set "NUMBER" to the best-fitting solution. This needs to be done manually,
# after the convenience clustering on the factors has run. Soz!)
NETWORK_CLUSTER_NUMBER = 1
NETWORK_DIM_REDUCTION = "MDS"
NETWORK_CLUSTER_METHOD = "WARD"
# Groups variables as thematic clusters, or set as None. This will only work
# if NETWORK_CLUSTER_NUMBER == 1.
THEMATIC_CLUSTERS = { \
    "Mental health":  ["Anxiety", "Depression"], \
    "Cognition":  ["Number sense", "Spatial STM", "Verbal STM", "Search", \
        "Fluid reasoning", "Inhibition", "Speed"], \
    "Attitude":  ["Grit", "Conscientiousness", "Growth mindset", \
        "School liking", "Class distraction"], \
    "Education":  ["Reading", "Sums"], \
    "Environment":  ["Affluence", "Deprivation", "Calm home"], \
    }
# Themes are the keys to THEMATIC_CLUSTERS, but in the order that they should
# be drawn. This gives control over the colour scheme.
THEMES = ["Environment", "Education", "Attitude", "Cognition", "Mental health"]

# Factor loadings (PCA and varimax rotation) as determined by an externally
# run analysis. The FACTOR_CLUSTERS argument overrules any other colouring
# argument, and sets node colours according to the factor they most align with.
FACTOR_CLUSTERS = True
# The following is a fallback for if the factor analysis is not run. The names
# are based on manual inspection and interpretation, and the values copied over
# from a previous factor analysis. The names will be retained, but the values
# overwritten if DO_FACTOR == True
FACTORS = ["Cognition", "Attitude", "Mental Health", "Speed", "SES"]
FACTOR_SCORES = { \
    "Affluence":            [ 0.056,  0.024,  0.034,  0.145,  0.794], \
    "Anxiety":              [-0.023, -0.044,  0.805, -0.061, -0.019], \
    "Calm home":            [-0.042,  0.547, -0.247,  0.050,  0.146], \
    "Class distraction":    [-0.207, -0.347,  0.491,  0.054,  0.025], \
    "Conscientiousness":    [-0.086,  0.754,  0.044,  0.082, -0.060], \
    "Depression":           [-0.229, -0.196,  0.760,  0.011, -0.058], \
    "Deprivation":          [-0.312, -0.051,  0.119,  0.212, -0.571], \
    "Fluid reasoning":      [ 0.640,  0.084, -0.206,  0.014,  0.179], \
    "Grit":                 [ 0.199,  0.713,  0.044,  0.002,  0.180], \
    "Growth mindset":       [ 0.163,  0.582, -0.125, -0.028, -0.006], \
    "Inhibition":           [ 0.583,  0.104, -0.028, -0.105, -0.351], \
    "Number sense":         [ 0.629,  0.021, -0.075,  0.171,  0.033], \
    "Reading":              [ 0.592,  0.082,  0.031,  0.265,  0.091], \
    "School liking":        [ 0.106,  0.598, -0.256,  0.003, -0.180], \
    "Search":               [ 0.164, -0.002, -0.220,  0.726, -0.037], \
    "Spatial STM":          [ 0.707,  0.074, -0.056,  0.095, -0.009], \
    "Speed":                [ 0.228,  0.076,  0.200,  0.730,  0.063], \
    "Sums":                 [ 0.766,  0.071, -0.067,  0.186,  0.097], \
    "Verbal STM":           [ 0.579,  0.045, -0.185, -0.061,  0.241], 
    }
    

# Set the multiple-comparisons correction method. Choose from None, "holm", or
# "bonferroni".
CORRELATION_ALPHA = 0.05
CORRELATION_CORRECTION = "holm"
PARTIAL_CORRELATION_CORRECTION = None

# Choose the type of test ("factor" for factor analysis, of "ica" for 
# independent component analysis).
ANALYSIS_TYPE = { \
    "que": "custom", \
    "cog": "custom", \
    "edu": None, \
    }

# Choose the number of factors to extract.
N_FACTORS = { \
    "que": 11, \
    "cog": 7, \
    "edu": 2, \
    }

# Choose the factor labels.
# (NOTE: Only do this after visual inspection of the factor loadings, or if
# you are going by a pre-determined thematic combination of variables.)
FACTORLABELS = { \
    "que": ["Depression", "Anxiety", "Calm home", "Affluence", \
        "Deprivation", "Grit", "Conscientiousness", "Growth mindset", \
        "School liking", "Class distraction"], \
    "cog": ["Fluid reasoning", "Spatial STM", "Verbal STM", "Search", \
        "Number sense", "Inhibition", "Speed"], \
    "edu": ["Reading", "Sums"], \
    }
# Choose what factors to reverse. (This allows higher severity to map onto the
# same direction, i.e. positive or negative. Only set these after inspecting
# the factor loadings, or if you have pre-determined variable mappings.)
REVERSE_FACTOR = { \
    "que": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], \
    "cog": [1, 1, 1, 1, 1, 1, 1], \
    "edu": [1, 1], \
    }
# What type of imputation to use, if any. Choose from:
#   None (all NaNs are ignored), mean, median, most_frequent, or knn
IMPUTATION = "knn"
# Maximum proportion of missing values allowed to still be included.
MAX_MISSING_PROP = 0.2

# What pre-processing to use. Choose from:
#   normalise, standardise, min-max, and None for no pre-processing
PREPROCESS = "standardise"

# The name of the dataset that we'd like to test.
DATASET = 'synthetic_data'
DATAEXT = 'csv'

# All cognitive factors to take on board.
COGVARS = [ \
    # Search organisation with markings.
    "CANCELLATION_MARKED-bestR", \
    "CANCELLATION_MARKED-intersect_rate", \
    "CANCELLATION_MARKED-interdist", \
    # Processing speed.
    "CANCELLATION_MARKED-intertime", \
    "CANCELLATION_UNMARKED-intertime", \
    # Inhibition.
    "GONOGO-dprime", \
    # Number sense.
    "ANS-p_correct", \
    # Fluid intelligence.
    "CATTELL_1-n_correct", \
    "CATTELL_2-n_correct", \
    # Short-term memory spans.
    "DIGIT_SPAN-n_correct", \
    "DOT_MATRIX-n_correct", \
    ]

# All questionnaire factors to use in the analysis.
QVARS = [ \
    # Deprivation
    "DEPRIVATION-normalised_IDACI_Rank", \
    # RCADS depression
    "QUESTIONNAIRE_2-Q5_resp", "QUESTIONNAIRE_2-Q6_resp", \
    "QUESTIONNAIRE_2-Q7_resp", "QUESTIONNAIRE_2-Q8_resp", \
    "QUESTIONNAIRE_2-Q9_resp"    , \
    # RCADS anxiety
    "QUESTIONNAIRE_2-Q10_resp", "QUESTIONNAIRE_2-Q11_resp", \
    "QUESTIONNAIRE_2-Q12_resp", "QUESTIONNAIRE_2-Q13_resp", \
    "QUESTIONNAIRE_2-Q14_resp", \
    # Household chaos
    "QUESTIONNAIRE_1-Q1_resp", "QUESTIONNAIRE_1-Q4_resp", \
    "QUESTIONNAIRE_1-Q18_resp", "QUESTIONNAIRE_1-Q19_resp", \
    "QUESTIONNAIRE_1-Q2_resp", "QUESTIONNAIRE_1-Q3_resp", \
    "QUESTIONNAIRE_1-Q17_resp", \
    # Family affluence
    "QUESTIONNAIRE_1-Q5_resp", "QUESTIONNAIRE_1-Q9_resp", \
    "QUESTIONNAIRE_1-Q10_resp", \
    "QUESTIONNAIRE_1-Q6_resp", "QUESTIONNAIRE_1-Q7_resp", \
    "QUESTIONNAIRE_1-Q8_resp", \
    # Grit
    "QUESTIONNAIRE_2-Q1_resp", "QUESTIONNAIRE_2-Q2_resp", \
    "QUESTIONNAIRE_2-Q3_resp", "QUESTIONNAIRE_2-Q4_resp", \
    # Conscientiousness
    "QUESTIONNAIRE_1-Q13_resp", "QUESTIONNAIRE_1-Q14_resp", \
    "QUESTIONNAIRE_1-Q16_resp", \
    # Growth mindset
    "QUESTIONNAIRE_2-Q15_resp", \
    # School liking
    "QUESTIONNAIRE_3-Q1_resp", \
    "QUESTIONNAIRE_3-Q2_resp", \
    # Class distraction
    "QUESTIONNAIRE_1-Q15_resp", \
    ]
# Total number of questions in each quiz (not necessarily all used here).
N_QUESTIONS = { \
    1: 19, \
    2: 16, \
    3: 12, \
    }

# Variables to correlate each factor with.
EDUVARS = [ \
    # Educational attainment.
    "READING-n_correct", \
    "SUMS-n_correct", \
    ]

# All question labels, ordered by task number, then question number
ALL_QUESTION_LABELS = [ \
        # Questionnaire 1
        [
        "regular bedtime",
        "hear think at home",
        "TV on at home",
        "calm house atmosphere",
        
        "N computers",
        "family car",
        "own bedroom",
        "dishwasher",
        
        "N holidays abroad",
        "N bathrooms",
        "N adults at home",
        "N siblings",
        
        "check homework",
        "play after homework",
        "distracted in class",
        "tidy bedroom",
        
        "messy home",
        "relaxing home",
        "quiet home",
        ],
    
        # Questionnaire 2
        [
        "finish class work",
        "try harder after bad grade",
        "work until work is right",
        "persist in classwork",
        
        "nothing is fun",
        "sad or empty",
        "very tired",
        "do not want to move",
        "appetite problems", 
        
        "worry bad things",
        "worry something bad to myself",
        "worry what will happen",
        "worry something awful family",
        "think about death",
        
        "always can be more clever",
        "cannot change talent"
        ],
    
        # Questionnaire 3
        [
        "like coming to school",
        "wish stay home from school",
        
        "hours reading alone",
        "hours reading with a parent",
        
        "N books",
        
        "hours on computer",
        "hours watching videos",
        "hours chatting online",
        "hours computer for learning",
        "hours playing games",
        "hours computer for creativity",
        
        "personal mobile phone",
        ],
    ]

# Axis labels for different variables.
VARLABELS = {
    # Fluid intelligence.
    'CATTELL_1-n_correct':"Fluid intelligence (Cattell series)", \
    'CATTELL_2-n_correct':"Fluid intelligence (Cattell classification)", \
    'CATTELL-n_correct':"Fluid intelligence (Cattell)", \
    # Short-term memory.
    'DIGIT_SPAN-n_correct': "Verbal short-term memory (digit span)", \
    'DOT_MATRIX-n_correct': "Spatial short-term memory (dot matrix)", \
    # Educational fluency measures.
    'READING-n_correct': "Reading fluency", \
    'SUMS-n_correct': "Maths fluency", \
    # Processing speed
    'CANCELLATION_MARKED-intertime': "Inter-detection time (marked targets)", \
    'CANCELLATION_UNMARKED-intertime': "Inter-detection time (unmarked targets)", \
    # Search organisation.
    'CANCELLATION_MARKED-bestR': "Search organisation (best R)", \
    'CANCELLATION_MARKED-intersect_rate': "Search organisation (intersection rate)", \
    'CANCELLATION_MARKED-interdist': "Inter-cancellation distance", \
    # Deprivation scores
    "GONOGO-dprime": "Inhibition (Go/NoGo d')", \
    # Deprivation scores
    "ANS-p_correct": "Approximate number sense (accuracy)", \
    # Deprivation scores
    "DEPRIVATION-normalised_IDACI_Rank": "Child-affecting deprivation", \
    }

# Construct variable labels for the questionnaire items.
for i in range(1,4):
    for j in range(1, N_QUESTIONS[i]+1):
        VARLABELS["QUESTIONNAIRE_{}-Q{}_resp".format(i,j)] = ALL_QUESTION_LABELS[i-1][j-1]

# PLOT SETTINGS
FONTSIZE = { \
    "title":            30, \
    "axtitle":          24, \
    "legend":           14, \
    "bar":              14, \
    "label":            20, \
    "ticklabels":       12, \
    "annotation":       14, \
    }
# Define a list of output colours (this is the Tango colour scheme).
COLS = {    "butter": [    '#fce94f',
                           '#edd400',
                           '#c4a000'],
            "orange": [    '#fcaf3e',
                           '#f57900',
                           '#ce5c00'],
            "chocolate": [ '#e9b96e',
                           '#c17d11',
                           '#8f5902'],
            "chameleon": [ '#8ae234',
                           '#73d216',
                           '#4e9a06'],
            "skyblue": [   '#729fcf',
                           '#3465a4',
                           '#204a87'],
            "plum":     [  '#ad7fa8',
                           '#75507b',
                           '#5c3566'],
            "scarletred":[ '#ef2929',
                           '#cc0000',
                           '#a40000'],
            "aluminium": [ '#eeeeec',
                           '#d3d7cf',
                           '#babdb6',
                           '#888a85',
                           '#555753',
                           '#2e3436'],
        }
# Assign colours to variables for the radial bar plot.
PLOTCOLS = { \
    "Depression":       COLS["orange"][2], \
    "Anxiety":          COLS["orange"][1], \

    "Affluence":        COLS["scarletred"][1], \
    "Deprivation":      COLS["scarletred"][2], \

    "Grit":             COLS["chocolate"][1], \
    "Conscientiousness":COLS["chocolate"][2], \
    "Growth mindset":   COLS["chocolate"][0], \

    "Calm home":        COLS["butter"][2], \
    "School liking":    COLS["butter"][0], \
    "Class distraction":COLS["butter"][1], \

    "Fluid reasoning":  COLS["skyblue"][1], \
    "Spatial STM":      COLS["skyblue"][2], \
    "Verbal STM":       COLS["skyblue"][2], \
    "Number sense":     COLS["skyblue"][0], \
    "Search":           COLS["plum"][1], \
    "Speed":            COLS["plum"][2], \
    "Inhibition":       COLS["plum"][0], \
    
    "Reading":          COLS["chameleon"][1], \
    "Sums":             COLS["chameleon"][2], \
    }


# # # # #
# CUSTOM FACTORS

# Factors can be based on external analysis, or questionnaire sub-scales.
CUSTOM_FACTOR = { \
    "que": { \
        # RCADS Depression
        0: {"pos": ["QUESTIONNAIRE_2-Q5_resp", "QUESTIONNAIRE_2-Q6_resp", \
                    "QUESTIONNAIRE_2-Q7_resp", "QUESTIONNAIRE_2-Q8_resp", \
                    "QUESTIONNAIRE_2-Q9_resp"], \
            "neg": [],
            },
        # RCADS Anxiety
        1: {"pos": ["QUESTIONNAIRE_2-Q10_resp", "QUESTIONNAIRE_2-Q11_resp", \
                    "QUESTIONNAIRE_2-Q12_resp", "QUESTIONNAIRE_2-Q13_resp", \
                    "QUESTIONNAIRE_2-Q14_resp"], \
            "neg": [],
            },
        # Household chaos
        2: {"pos": ["QUESTIONNAIRE_1-Q1_resp", "QUESTIONNAIRE_1-Q4_resp", \
                    "QUESTIONNAIRE_1-Q18_resp", "QUESTIONNAIRE_1-Q19_resp"],
            "neg": ["QUESTIONNAIRE_1-Q2_resp", "QUESTIONNAIRE_1-Q3_resp", \
                    "QUESTIONNAIRE_1-Q17_resp"],
            },
        # Family affluence
        3: {"pos": ["QUESTIONNAIRE_1-Q5_resp", "QUESTIONNAIRE_1-Q9_resp", \
                    "QUESTIONNAIRE_1-Q10_resp"], \
            "neg": ["QUESTIONNAIRE_1-Q6_resp", "QUESTIONNAIRE_1-Q7_resp", \
                    "QUESTIONNAIRE_1-Q8_resp"],
            },
        # Deprivation index
        4: {"pos": ["DEPRIVATION-normalised_IDACI_Rank"], \
            "neg": [],
            },
        # Grit
        5: {"pos": ["QUESTIONNAIRE_2-Q1_resp", "QUESTIONNAIRE_2-Q2_resp", \
                    "QUESTIONNAIRE_2-Q3_resp", "QUESTIONNAIRE_2-Q4_resp"],
            "neg": [],
            },
        # Conscientiousness
        6: {"pos": ["QUESTIONNAIRE_1-Q13_resp", "QUESTIONNAIRE_1-Q14_resp", \
                    "QUESTIONNAIRE_1-Q16_resp"],
            "neg": [],
            },
        # Growth mindset
        7: {"pos": ["QUESTIONNAIRE_2-Q15_resp"],
            "neg": [],
            },
        # School liking
        8: {"pos": ["QUESTIONNAIRE_3-Q1_resp"],
            "neg": ["QUESTIONNAIRE_3-Q2_resp"],
            },
        # Distracted in class
        9: {"pos": ["QUESTIONNAIRE_1-Q15_resp"], \
            "neg": [],
             }
        }, \

    "cog": { \
        # Fluid reasoning
        0: {"pos": ["CATTELL_1-n_correct", "CATTELL_2-n_correct"], \
            "neg": [],
            },
        # Spatial short-term memory
        1: {"pos": ["DOT_MATRIX-n_correct"], \
            "neg": [],
            },
        # Verbal short-term memory
        2: {"pos": ["DIGIT_SPAN-n_correct"], \
            "neg": [],
            },
        # Search organisation
        3: {"pos": ["CANCELLATION_MARKED-bestR"], \
            "neg": ["CANCELLATION_MARKED-intersect_rate", \
                "CANCELLATION_MARKED-interdist"],
            },
        # Number sense
        4: {"pos": ["ANS-p_correct"], \
            "neg": [], \
            },
        # Inhibition
        5: {"pos": ["GONOGO-dprime"], \
            "neg": [], \
            },
        # Processing soeed
        6: {"pos": [], \
            "neg": ["CANCELLATION_MARKED-intertime", \
                "CANCELLATION_UNMARKED-intertime"], \
            },
        }, \
    }


# # # # #
# LOAD DATA

# This directory, and the data directory.
DIR = os.path.dirname(os.path.abspath(__file__))
DATADIR = os.path.join(DIR, "data_all_in_one")

# Output directory.
OUTDIR = os.path.join(DIR, 'red_overview_output')
if not os.path.isdir(OUTDIR):
    os.mkdir(OUTDIR)
COUTDIR = os.path.join(OUTDIR, 'cluster_comparisons')
if not os.path.isdir(COUTDIR):
    os.mkdir(COUTDIR)

# The data file.
DATAFILE = os.path.join(DATADIR, '{}.{}'.format(DATASET, DATAEXT))

# Load data.
print("Loading data file '{}'".format(DATASET))
if not os.path.isfile(DATAFILE):
    raise Exception("ERROR: File does not exist '{}'".format(DATAFILE))
if DATAEXT == "csv":
    delim = ','
elif DATAEXT == "tsv":
    delim = '\t'
else:
    delim = None
raw = numpy.loadtxt(DATAFILE, dtype=str, unpack=True, delimiter=delim)
data = {}
for i in range(raw.shape[0]):
    var = raw[i,0]
    empty = raw[i,1:] == ''
    raw[i,1:][empty] = 'nan'
    if var == "ppname":
        val = raw[i,1:]
    else:
        try:
            val = raw[i,1:].astype(float)
        except:
            val = raw[i,1:]
    data[var] = val

# Compute additional variables.
data["CATTELL-n_correct"] = data["CATTELL_1-n_correct"] + \
    data["CATTELL_2-n_correct"]
# Z-score the deprivation rank (divide rank score by 32844, then use
# scipy.stats.norm.ppf). Also make sure that the ranks are reversed, so that
# higher values indicate more deprivation.
data["DEPRIVATION-normalised_rank"] = scipy.stats.norm.ppf( \
    (32884.0-data["DEPRIVATION-Index_of_Multiple_Deprivation_Rank"]) / 32884.0)
data["DEPRIVATION-normalised_IDACI_Rank"] = scipy.stats.norm.ppf( \
    (32884.0-data["DEPRIVATION-IDACI_Rank"]) / 32884.0)

# Combine all variables in a single list.
VARS = []
VARS.extend(COGVARS)
VARS.extend(QVARS)
VARS.extend(EDUVARS)
varnames = {}
varnames["cog"] = COGVARS
varnames["que"] = QVARS
varnames["edu"] = EDUVARS

# Stuff the data in a matrix.
n_features = len(VARS)
n_subjects = len(data[VARS[0]])
# Construct X.
X_raw = numpy.zeros((n_subjects, n_features), dtype=float)
for i, var in enumerate(VARS):
    X_raw[:,i] = data[var]


# # # # #
# REVIEWER REQUESTS

# DEPRIVATION INDICES
# Output a few things to a text file, in response to a reviewer question.
with open(os.path.join(OUTDIR, "deprivation_data_for_reviewer.txt"), "w") as f:
    f.write("DEPRIVATION INDEX DETAILS")

    for varname in ["DEPRIVATION-Index_of_Multiple_Deprivation_Rank", \
            "DEPRIVATION-normalised_rank", "DEPRIVATION-IDACI_Rank", \
            "DEPRIVATION-normalised_IDACI_Rank"]:
        f.write("\n\n{} \n\tM={} \n\tSD={} \n\trange=[{}, {}]".format( \
            varname, \
            numpy.round(numpy.nanmean(data[varname]), 2), \
            numpy.round(numpy.nanstd(data[varname]), 2), \
            numpy.round(numpy.nanmin(data[varname]), 2), \
            numpy.round(numpy.nanmax(data[varname]), 2), \
            ))
        dep_notnan = (numpy.isnan(data["DEPRIVATION-normalised_rank"]) | \
            numpy.isnan(data["DEPRIVATION-normalised_IDACI_Rank"])) == False
        dep_r, dep_p = scipy.stats.pearsonr( \
            data["DEPRIVATION-normalised_rank"][dep_notnan], \
            data["DEPRIVATION-normalised_IDACI_Rank"][dep_notnan])

    f.write("\n\nPearson correlation IMD and IDACI: R={}, p={}".format( \
        numpy.round(dep_r, 2), dep_p))

# PRE-STANDARDISATION SCORES
with open(os.path.join(OUTDIR, "descriptives_for_reviewer.tsv"), "w") as f:

    header = ["varname", "M", "SD", "min", "max"]
    f.write("\t".join(header))

    # Compute the raw scores for the cognitive tests.
    for varname in ["CATTELL_1-n_correct", "CATTELL_2-n_correct", \
        "DOT_MATRIX-n_correct", "DIGIT_SPAN-n_correct", \
        "CANCELLATION_MARKED-bestR", "CANCELLATION_MARKED-intersect_rate", \
        "CANCELLATION_MARKED-interdist", "ANS-p_correct", "GONOGO-dprime", \
        "CANCELLATION_MARKED-intertime", "CANCELLATION_UNMARKED-intertime", \
        "READING-n_correct", "SUMS-n_correct"]:

        line = [ \
            varname, \
            numpy.round(numpy.nanmean(data[varname]), 2), \
            numpy.round(numpy.nanstd(data[varname]), 2), \
            numpy.round(numpy.nanmin(data[varname]), 2), \
            numpy.round(numpy.nanmax(data[varname]), 2), \
            ]
        f.write("\n" + "\t".join(map(str, line)))
    
    # Do the same for the questionnaires.
    for i, varname in enumerate(FACTORLABELS["que"]):
        
        scores = numpy.zeros(data["ppname"].shape[0], dtype=numpy.float32)
        for varlbl in CUSTOM_FACTOR["que"][i]["pos"]:
            scores += data[varlbl]
        for varlbl in CUSTOM_FACTOR["que"][i]["neg"]:
            scores -= data[varlbl]
        line = [ \
            varname, \
            numpy.round(numpy.nanmean(scores), 2), \
            numpy.round(numpy.nanstd(scores), 2), \
            numpy.round(numpy.nanmin(scores), 2), \
            numpy.round(numpy.nanmax(scores), 2), \
            ]
        f.write("\n" + "\t".join(map(str, line)))


# # # # #
# PREPROCESS

# Remove any infinite values.
X_raw[numpy.isinf(X_raw)] = numpy.NaN

# PREPROCESSING
print("Pre-processing with scaling method '{}' and imputation method '{}'".format( \
    PREPROCESS, IMPUTATION))
X_original, include = preprocess(X_raw, mode=PREPROCESS, impute=IMPUTATION, \
    max_miss_prop=MAX_MISSING_PROP)
ppnames = data["ppname"][include]

# Split the data into cognition, questionnaire, and educational outcomes.
X = {}
X["cog"] = X_original[:, :len(COGVARS)]
X["que"] = X_original[:, len(COGVARS):len(COGVARS)+len(QVARS)]
X["edu"] = X_original[:, len(COGVARS)+len(QVARS):]


# # # # #
# SAVE SAMPLE DETAILS

# Store outcome in a text file, using the following string.
sample_description = ""

# Count the number of datasets with postcode data. These are all the children
# for who we obtained data from schools.
n_postcode = X_raw.shape[0] - numpy.sum(numpy.isnan( \
    X_raw[VARS.index("DEPRIVATION-normalised_IDACI_Rank")]))
sample_description += \
    "Postcodes from {} children were obtained from all schools.\n\n".format( \
    n_postcode)

# Count the number of complete datasets.
n_nans = numpy.sum(numpy.isnan(X_raw), 1)
unique_numbers = numpy.sort(numpy.unique(n_nans))
for n_nan in unique_numbers:
    n_nan_count = numpy.sum(n_nans==n_nan)
    sample_description += \
        "{} cases with {} missing values (p_missing={})\n".format( \
        n_nan_count, n_nan, float(n_nan)/float(n_features))

# Count the number of remaining datasets after preprocessing.
n_remaining = X_original.shape[0]
sample_description += "\n{} datasets included in analysis".format(n_remaining)

# Count the proportion of children with <70% reading performance.
notnan = numpy.isnan(data["READING-p_correct"]) == False
p_low_read = float(numpy.sum(data["READING-p_correct"][notnan] < 0.7)) \
    / float(numpy.sum(notnan))
sample_description += "\n\n{} percent of children score lower than 70 percent correct in the reading task".format( \
    p_low_read*100.0)

# Deprivation details.
notnan = numpy.isnan(data["DEPRIVATION-normalised_IDACI_Rank"]) == False
sample_description += "\n\nIDACI z-score M={}, SD={}, min={}, max={}".format( \
    numpy.mean(data["DEPRIVATION-normalised_IDACI_Rank"][notnan]), \
    numpy.std(data["DEPRIVATION-normalised_IDACI_Rank"][notnan]), \
    numpy.min(data["DEPRIVATION-normalised_IDACI_Rank"][notnan]), \
    numpy.max(data["DEPRIVATION-normalised_IDACI_Rank"][notnan]))

print("Sampling details:\n{}\n\n".format(sample_description))

# Write the description to file.
with open(os.path.join(OUTDIR, "sample_description.txt"), "w") as f:
    f.write(sample_description)


# # # # #
# DECOMPOSITION

# Run through all separate datasets.
X_decomp = {}
for datatype in X.keys():
    print("\nDecompositioning datatype '{}' with method '{}'".format( \
        datatype, ANALYSIS_TYPE[datatype]))
    # Initialise a new factor analysis.
    if ANALYSIS_TYPE[datatype] is None:
        model = None
        X_decomp[datatype] = numpy.copy(X[datatype])
    elif ANALYSIS_TYPE[datatype] is "custom":
        model = None
        components = list(CUSTOM_FACTOR[datatype].keys())
        components.sort()
        n_components = len(components)
        X_decomp[datatype] = numpy.zeros((X[datatype].shape[0],n_components), \
            dtype=float)
        for i in components:
            for var in CUSTOM_FACTOR[datatype][i]["pos"]:
                X_decomp[datatype][:,i] += X[datatype][:,varnames[datatype].index(var)]
            for var in CUSTOM_FACTOR[datatype][i]["neg"]:
                X_decomp[datatype][:,i] -= X[datatype][:,varnames[datatype].index(var)]
    elif ANALYSIS_TYPE[datatype].lower() == "factor":
        model = decomposition.FactorAnalysis(n_components=N_FACTORS[datatype])
    elif ANALYSIS_TYPE[datatype].lower() == "ica":
        model = decomposition.FastICA(n_components=N_FACTORS[datatype])
    elif ANALYSIS_TYPE[datatype].lower() == "pca":
        model = decomposition.PCA(n_components=N_FACTORS[datatype])
    else:
        raise Exception("ERROR: Unknown decomposition type '{}'".format( \
            ANALYSIS_TYPE[datatype]))

    # Apply the decompositioning (if any).
    if model is not None:
        # Perform the factor analysis on the data.
        model.fit(X[datatype])
        # Transform the data.
        X_decomp[datatype] = model.transform(X[datatype])
    
    # Reverse the components when necessary.
    for i in range(X_decomp[datatype].shape[1]):
        if REVERSE_FACTOR[datatype][i] == -1:
            X_decomp[datatype][:,i] *= -1.0

    # Plot the correlation matrix for the components and the original data.
    plotvarnames = varnames[datatype]
    if datatype == "que":
        plotvarnames = []
        for var in QVARS:
            plotvarnames.append(VARLABELS[var])
    if CORRELATION_CORRECTION is None:
        p = CORRELATION_ALPHA
    elif CORRELATION_CORRECTION == "bonferroni":
        p = CORRELATION_ALPHA / float(X_decomp[datatype].shape[1] * X[datatype].shape[1])
    elif CORRELATION_CORRECTION == "holm":
        print("WARNING: Can't do Holm-Bonferroni correction for this correlation matrix. Using alpha={} instead.".format( \
            CORRELATION_ALPHA))
        p = CORRELATION_ALPHA
    else:
        raise Exception("ERROR: Unrecognised multiple-comparisons correction type '{}'".format( \
            CORRELATION_CORRECTION))
    correlation_matrix(X_decomp[datatype], X[datatype], \
        varnames=plotvarnames, sig=p, vlim=1.0, ax=None, \
        savepath=os.path.join(OUTDIR, "cormat_{}.png".format(datatype)))
    # Plot another correlation matrix for educational outcomes.
    if CORRELATION_CORRECTION is None:
        p = CORRELATION_ALPHA
    elif CORRELATION_CORRECTION == "bonferroni":
        p = CORRELATION_ALPHA / float(X_decomp[datatype].shape[1] * X[datatype].shape[1])
    elif CORRELATION_CORRECTION == "holm":
        print("WARNING: Can't do Holm-Bonferroni correction for this correlation matrix. Using alpha={} instead.".format( \
            CORRELATION_ALPHA))
        p = CORRELATION_ALPHA
    else:
        raise Exception("ERROR: Unrecognised multiple-comparisons correction type '{}'".format( \
            CORRELATION_CORRECTION))
    correlation_matrix(X_decomp[datatype], X["edu"], \
        varnames=EDUVARS, sig=p, vlim=1.0, ax=None, \
        savepath=os.path.join(OUTDIR, "cormat_{}_edu.png".format(datatype)))

# Count the components for each datatype.
n_components = {}
for datatype in X_decomp.keys():
    n_components[datatype] = X_decomp[datatype].shape[1]


# # # # #
# JOIN DATA

# Join the data together into a single matrix.
print("\nJoining and standardising decomposed scores.")
X_tot = numpy.hstack((X_decomp["edu"], X_decomp["cog"], X_decomp["que"]))
X_tot, include = preprocess(X_tot, mode="standardise", impute=None)
X_tot = X_tot.astype(numpy.float32)
ppnames = ppnames[include]

# Create a list of labels.
varlabels = []
for i in range(X_tot.shape[1]):
    # Choose the appropriate name for this subplot.
    if i < n_components["edu"]:
        lbl = FACTORLABELS["edu"][i]
    elif (i >= n_components["edu"]) & (i < \
        n_components["edu"]+n_components["cog"]):
        lbl = FACTORLABELS["cog"][i-n_components["edu"]]
    elif i >= n_components["edu"]+n_components["cog"]:
        lbl = FACTORLABELS["que"][i-n_components["edu"]-n_components["cog"]]
    varlabels.append(lbl)


# # # # #
# STORE PRE-PROCESSED DATA

print("\nStoring pre-processed data.")

# Store the pre-processed data in a text file.
with open(os.path.join(OUTDIR, "preprocessed_data.csv"), "w") as f:
    header = []
    X_keys = list(X.keys())
    X_keys.sort()
    for k in X_keys:
        all_vars = {"cog":COGVARS, "que":QVARS, "edu":EDUVARS}
        for j, var in enumerate(all_vars[k]):
            header.append(var)
    header.extend(varlabels)
    f.write(",".join(map(str, header)))
    for i in range(X_original.shape[0]):
        line = []
        for k in X_keys:
            all_vars = {"cog":COGVARS, "que":QVARS, "edu":EDUVARS}
            for j, var in enumerate(all_vars[k]):
                line.append(X[k][i,j])
        for j in range(X_tot.shape[1]):
            line.append(X_tot[i,j])
        f.write("\n" + ",".join(map(str, line)))


# # # # #
# DESCRIPTIVES BY SCHOOL

print("\nComputing per-school and per-class descriptives.")

# First two numbers in ppname are school, third is class, last three are a
# random pupil ID.
schools = numpy.zeros(ppnames.shape[0], dtype=int)
for i, ppname in enumerate(ppnames):
    schools[i] = int(ppname[0:2])

with open(os.path.join(OUTDIR, "descriptives_per_school.csv"), "w") as f:
    header = ["var", "school", "M", "SD", "n", "t", "p"]
    f.write(",".join(map(str, header)))
    for vi, var in enumerate(varlabels):
        for si, school in enumerate(numpy.unique(schools)):
            sel = schools==school
            t, p = scipy.stats.ttest_ind(X_tot[sel,vi], \
                X_tot[sel==False,vi], equal_var=False)
            line = [var, school, numpy.mean(X_tot[sel,vi]), \
                numpy.std(X_tot[sel,vi]), numpy.sum(sel.astype(int)), t, p]
            f.write("\n" + ",".join(map(str, line)))


# # # # #
# FACTOR ANALYSIS

if DO_PCA:
    
    print("\nRunning principle component analysis to compute eigenvalues.")
    
    # Run a PCA to compute eigenvalues and proportion of data explained for
    # each grouping of measured variables.
    pca = decomposition.PCA(n_components=len(varlabels), svd_solver="full")
    pca.fit(X_tot)
    
    # Create a text file with the necessary output to create a table.
    with open(os.path.join(OUTDIR, "PCA_loadings.tsv"), "w") as f:
        for i, varname in enumerate(varlabels):
            line = [varname]
            line.extend(list(pca.components_[:,i]))
            f.write("\t".join(map(str, line)) + "\n")
        line = ["eigenvalue"]
        line.extend(list(pca.explained_variance_))
        f.write("\t".join(map(str, line)) + "\n")
        line = ["p_explained"]
        line.extend(list(pca.explained_variance_ratio_))
        f.write("\t".join(map(str, line)) + "\n")


if DO_FACTOR:
    
    print("\nRunning factor analysis with parallel analysis.")
    
    # Choose either regular PCA, or factor analysis with varimax rotation.
    if PARALLEL_FACTOR_TYPE == "FA":
        fa = FactorAnalyzer(n_factors=X_tot.shape[1], rotation="varimax")
    elif PARALLEL_FACTOR_TYPE == "PCA":
        fa = decomposition.PCA(n_components=X_tot.shape[1])
    # Fit the decomposer.
    fa.fit(X_tot)
    
    # Compute eigenvalues.
    if PARALLEL_FACTOR_TYPE == "FA":
        eigenvalues, _ = fa.get_eigenvalues()
        sumsqloadings, p_var, cum_p_var = fa.get_factor_variance()
        uniqueness = fa.get_uniquenesses()
        # Write to file.
        with open(os.path.join(OUTDIR, "factor_results_full.tsv"), "w") as f:
            header = ["feature"] + \
                ["factor_{}".format(str(i+1).rjust(2,"0")) \
                for i in range(len(eigenvalues))] + \
                ["uniqueness"]
            f.write("\t".join(map(str,header)))
            for i, var in enumerate(varlabels):
                line = [var] + list(fa.loadings_[i,:]) + [uniqueness[i]]
                f.write("\n" + "\t".join(map(str,line)))
            line = ["eigenvalue"] + list(eigenvalues) + [""]
            f.write("\n" + "\t".join(map(str,line)))
            line = ["SumSqLoad"] + list(sumsqloadings) + [""]
            f.write("\n" + "\t".join(map(str,line)))
            line = ["p_var_explained"] + list(p_var) + [""]
            f.write("\n" + "\t".join(map(str,line)))
            line = ["cum_var_explained"] + list(cum_p_var) + [""]
            f.write("\n" + "\t".join(map(str,line)))
                
    elif PARALLEL_FACTOR_TYPE == "PCA":
        eigenvalues = numpy.zeros(X_tot.shape[1], dtype=numpy.float32)
        cov_matrix = numpy.dot(X_tot.T, X_tot) / float(X_tot.shape[0])
        for ei in range(fa.components_.shape[0]):
            eigenvector = fa.components_[ei,:]
            eigenvalues[ei] = numpy.dot(eigenvector.T, \
                numpy.dot(cov_matrix, eigenvector))

    # Compute proportion of explained variance.
    explained_var = eigenvalues / float(X_tot.shape[1])

    # File name for the memoory-mapped array.
    memmap_fpath = os.path.join(OUTDIR, "memmap_parallel_analysis.dat")
    memmap_shape = (PARALLEL_N_PERMUTATIONS, X_tot.shape[1])
    
    # Load previous parallel analysis results.
    if (os.path.isfile(memmap_fpath)) and (not PARALLEL_OVERWRITE_EXISTING):
        par = numpy.memmap(memmap_fpath, mode="r", dtype=numpy.float32, \
            shape=memmap_shape)
        if PARALLEL_FACTOR_TYPE == "FA":
            ssql = numpy.memmap(memmap_fpath.replace(".dat", \
                "sumsqloading.dat"), mode="r", dtype=numpy.float32, \
                shape=memmap_shape)

    # Run a parallel analysis.
    else:
        par = numpy.memmap(memmap_fpath, mode="w+", dtype=numpy.float32, \
            shape=memmap_shape)
        if PARALLEL_FACTOR_TYPE == "FA":
            ssql = numpy.memmap(memmap_fpath.replace(".dat", \
                "sumsqloading.dat"), mode="w+", dtype=numpy.float32, \
                shape=memmap_shape)
        for i in range(par.shape[0]):

            # Generate a new distribution.
            if PARALLEL_PERMUTE:
                X_rand = numpy.copy(X_tot)
                for j in range(X_rand.shape[1]):
                    numpy.random.shuffle(X_rand[:,j])
            else:
                X_rand = numpy.random.randn(X_tot.shape[0], X_tot.shape[1])
            
            # Run PCA or factor analysis with varimax rotation.
            if PARALLEL_FACTOR_TYPE == "FA":
                fa = FactorAnalyzer(n_factors=X_rand.shape[1], \
                    rotation="varimax")
            elif PARALLEL_FACTOR_TYPE == "PCA":
                fa = decomposition.PCA(n_components=X_rand.shape[1])
            fa.fit(X_rand)
            
            # Save the eigenvalues.
            cov_matrix = numpy.dot(X_rand.T, X_rand) / float(X_rand.shape[0])
            if PARALLEL_FACTOR_TYPE == "FA":
                par[i,:], _ = fa.get_eigenvalues()
                ssql[i,:], _, _ = fa.get_factor_variance()
            elif PARALLEL_FACTOR_TYPE == "PCA":
                for ei in range(X_rand.shape[1]):
                    par[i,ei] = numpy.dot(fa.components_[ei,:].T, \
                        numpy.dot(cov_matrix, fa.components_[ei,:]))
    
    # Sort the parallel analysis results.
    par_sorted = numpy.copy(par)
    par_sorted = numpy.sort(par_sorted, axis=0)
    if PARALLEL_FACTOR_TYPE == "FA":
        ssql_sorted = numpy.copy(ssql)
        ssql_sorted = numpy.sort(ssql_sorted, axis=0)
    
    # Plot the real eigenvalues and those from the parallel analysis.
    if PARALLEL_FACTOR_TYPE == "FA":
        fig, axes = pyplot.subplots(nrows=1, ncols=2, figsize=(14.0,6.0), \
            dpi=300.0)
        plot_vars = ["eigen", "sqload"]
    else:
        fig, axes = pyplot.subplots(nrows=1, ncols=1, figsize=(7.0,6.0), \
            dpi=300.0)
        axes = [axes]
        plot_vars = ["eigen"]
    # Choose the data label.
    if PARALLEL_PERMUTE:
        rand_label = "Permuted data"
    else:
        rand_label = "Simulated data"
    # Plot all variables.
    for vi, var_type in enumerate(plot_vars):
        # Get the variables.
        if var_type == "eigen":
            val = eigenvalues
            par_val = par_sorted
            ylbl = "Eigenvalues"
        elif var_type == "sqload":
            val = sumsqloadings
            par_val = ssql_sorted
            ylbl = "Sum of squared loadings"
        else:
            raise Exception("Unrecognised variable '{}'".format(var_type))
        # Choose the appropriate axis.
        ax = axes[vi]
        # Create the x values.
        x_ = range(1, val.shape[0]+1)
        # Plot the real data.
        ax.plot(x_, val, "o-", lw=3, color="#000000", alpha=1.0, label="Data")
        # Plot the parallel random or shuffled data.
        for p in [0.5, 0.9, 0.95, 0.99]:
            alpha = 0.5 + (1.0 - p)
            i = int(numpy.floor(p*par_val.shape[0]))
            ax.plot(x_, par_val[i,:], ":", lw=3, color="#FF69B4", alpha=alpha, \
                label="{} ({}".format(rand_label, int(round(100*p))) + \
                r"$^{th}$ percentile)")
        # Plot Kaiser's rule at eigenvalue==1.
        if var_type == "eigen":
            ax.axhline(y=1.0, lw=3, ls="--", color="#000000", alpha=0.5, \
                label="Kaiser's rule")
        # Finish the plot.
        ax.set_xlabel("Factors", fontsize=FONTSIZE["label"])
        ax.set_xticks(x_)
        ax.set_xticklabels(x_, fontsize=FONTSIZE["ticklabels"])
        ax.set_ylabel(ylbl, fontsize=FONTSIZE["label"])
        y_ticks = range(0, 1 + int(numpy.ceil(numpy.max(eigenvalues))))
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_ticks, fontsize=FONTSIZE["ticklabels"])
        ax.legend(loc="upper right", fontsize=FONTSIZE["legend"])
    # Save and close the plot.
    fig.savefig(os.path.join(OUTDIR, "factor_screeplot_{}.png".format( \
        PARALLEL_FACTOR_TYPE)))
    pyplot.close(fig)
    
    
    # FACTOR ANALYSIS
    # Run the factor analysis with the pre-defined number of factors.
    if PARALLEL_FACTOR_TYPE == "FA":
        fa = FactorAnalyzer(n_factors=PARALLEL_N_FACTORS, rotation="varimax")
    elif PARALLEL_FACTOR_TYPE == "PCA":
        fa = decomposition.PCA(n_components=PARALLEL_N_FACTORS)
    # Fit the decomposer.
    fa.fit(X_tot)
    
    # Overwrite factor loadings.
    FACTOR_SCORES = {}
    for i, var in enumerate(varlabels):
        FACTOR_SCORES[var] = list(fa.loadings_[i,:])
    
    # Compute eigenvalues.
    if PARALLEL_FACTOR_TYPE == "FA":
        eigenvalues, _ = fa.get_eigenvalues()
        sumsqloadings, p_var, cum_p_var = fa.get_factor_variance()
        uniqueness = fa.get_uniquenesses()
        
        # Write to file.
        with open(os.path.join(OUTDIR, "factor_results_{}-factor.tsv".format( \
            PARALLEL_N_FACTORS)), "w") as f:
            header = ["feature"] + \
                ["factor_{}".format(str(i+1).rjust(2,"0")) \
                for i in range(len(p_var))] + \
                ["uniqueness"]
            f.write("\t".join(map(str,header)))
            for i, var in enumerate(varlabels):
                line = [var] + list(fa.loadings_[i,:]) + [uniqueness[i]]
                f.write("\n" + "\t".join(map(str,line)))
            line = ["eigenvalue"] + list(eigenvalues[:PARALLEL_N_FACTORS]) + [""]
            f.write("\n" + "\t".join(map(str,line)))
            line = ["p_var_explained"] + list(eigenvalues[:PARALLEL_N_FACTORS] \
                / eigenvalues.shape[0]) + [""]
            f.write("\n" + "\t".join(map(str,line)))
            line = ["SumSqLoad"] + list(sumsqloadings)
            f.write("\n" + "\t".join(map(str,line)))
            line = ["p_common_var_explained"] + list(p_var)
            f.write("\n" + "\t".join(map(str,line)))
            line = ["cum_common_var_explained"] + list(cum_p_var)
            f.write("\n" + "\t".join(map(str,line)))


# # # # #
# CLUSTERING

if DO_CLUSTERING:

    print("\nCluster analysis on individuals:")

    # Reduce the dimensionality of the data.
    print("\tApplying dimensionality reduction '{}'".format(CLUSTER_DIM_REDUCTION))
    if CLUSTER_DIM_REDUCTION is not None:
        X_tot_reduced = dim_reduction(X_tot, n_components=2, mode=CLUSTER_DIM_REDUCTION)
    else:
        X_tot_reduced = numpy.copy(X_tot)
    # Cluster the reduced data.
    print("\tApplying clustering method '{}'".format(CLUSTER_METHOD))
    convenience_clustering(X_tot_reduced, CLUSTER_METHOD, 10, \
        os.path.join(OUTDIR, "clustering_{}_{}".format(CLUSTER_DIM_REDUCTION, CLUSTER_METHOD)), \
        varnames=None, X_original=X_tot, varnames_original=varlabels)
    
    print("\nCluster analysis on features:")
    
    # Reduce the dimensionality of the data.
    print("\tApplying dimensionality reduction '{}'".format(CLUSTER_DIM_REDUCTION))
    if CLUSTER_DIM_REDUCTION is not None:
        X_tot_reduced = dim_reduction(X_tot.T, n_components=2, mode=CLUSTER_DIM_REDUCTION)
    else:
        X_tot_reduced = numpy.copy(X_tot.T)
    # Custer the reduced data.
    print("\tApplying clustering method '{}'".format(CLUSTER_METHOD))
    convenience_clustering(X_tot_reduced, CLUSTER_METHOD, 10, \
        os.path.join(OUTDIR, "clustering_features_{}_{}".format( \
        CLUSTER_DIM_REDUCTION, CLUSTER_METHOD)), varnames=None, \
        X_original=X_tot.T, varnames_original=None)


# # # # #
# NETWORK ANALYSIS

if DO_NETWORK:
    
    print("\nRunning network analysis...")

    # CORRELATION AND PARTIAL CORRELATION
    print("Computing full and partial correlations between all features.")

    # Create a list of nodes.
    nodes = copy.deepcopy(varlabels)

    # Compute the partial correlations for the whole sample, without
    # regularisation.
    pc, pc_p, a = partial_corr(X_tot, alpha=0, average_triangles=True)

    # Compute a correlation matrix.
    fc = numpy.zeros(pc.shape, dtype=float)
    fc_p = numpy.zeros(pc.shape, dtype=float)
    for i in range(X_tot.shape[1]):
        for j in range(X_tot.shape[1]):
            fc[i,j], fc_p[i,j] = scipy.stats.pearsonr(X_tot[:,i], X_tot[:,j])

    # Create Boolean matrices for upper and lower half.
    upper = numpy.zeros(pc.shape, dtype=bool)
    lower = numpy.zeros(pc.shape, dtype=bool)
    for i in range(pc.shape[0]):
        for j in range(pc.shape[1]):
            # Upper half (top-right).
            if i < j:
                upper[i,j] = True
            # Lower half (bottom-left).
            elif i > j:
                lower[i,j] = True

    # Combine full and partial correlation within a single matrix.
    comb = numpy.zeros(pc.shape, dtype=float)
    comb[upper] = fc[upper]
    comb[lower] = pc[lower]
    comb_p = numpy.ones(comb.shape, dtype=float)
    comb_p[upper] = fc_p[upper]
    comb_p[lower] = pc_p[lower]
    
    # Count the number of tests.
    n_correlations = ((pc.shape[0] * pc.shape[1]) - len(nodes)) // 2
    
    # No correction (OMG SO RECKLESS!!).
    print("Implementing (partial) correlation correction for multiple comparisons ({})".format( \
        CORRELATION_CORRECTION))
    if CORRELATION_CORRECTION is None:
        alpha = numpy.ones(comb.shape, dtype=float) * CORRELATION_ALPHA
    # Bonferroni correction.
    elif CORRELATION_CORRECTION == "bonferroni":
        alpha = numpy.ones(comb.shape, dtype=float) * (CORRELATION_ALPHA / float(n_correlations))
    # Holm-Bonferroni correction.
    elif CORRELATION_CORRECTION == "holm":
        # Rank-order correlation p-values.
        fc_rank = numpy.argsort(fc_p[upper], axis=None)
        pc_rank = numpy.argsort(pc_p[lower], axis=None)
        # Awkward way to re-organise p-values back into a single matrix.
        comb_rank = numpy.ones(comb.shape, dtype=float) * (n_correlations - 1)
        for i in range(comb.shape[0]):
            for j in range(comb.shape[1]):
                # p values of 1 get the highest possible rank.
                if comb_p[i,j] == 1.0:
                    comb_rank[i,j] = n_correlations - 1
                # Find matching p-value.
                else:
                    if i < j:
                        comb_rank[i,j] = fc_rank[fc_p[upper]==comb_p[i,j]][0]
                    elif i > j:
                        comb_rank[i,j] = pc_rank[pc_p[upper]==comb_p[i,j]][0]
        # Define alpha according to Holm-Bonferroni rule.
        alpha = CORRELATION_ALPHA / (n_correlations - comb_rank)
    else:
        raise Exception("ERROR: Unrecognised multiple-comparisons correction type '{}'".format( \
            CORRELATION_CORRECTION))
    
    if PARTIAL_CORRELATION_CORRECTION is None:
        print("Overruling partial correlation correction.")
        alpha[lower] = CORRELATION_ALPHA
    
    # Correct correlations.
    comb_sig = numpy.copy(comb)
    comb_sig[comb_p>alpha] = 0.0
    
    print("Plotting correlation matrices.")
    # Plot the correlation matrix.
    file_path = os.path.join(OUTDIR, "correlation_matrix_pearson.png")
    plot_correlation_matrix(fc, file_path, varlabels=nodes, \
        cbar_label=r"Correlation $R$", vmin=-0.5, vmax=0.5, \
        cmap="RdBu_r", dpi=300.0)
    # Plot the partial correlation matrix.
    file_path = os.path.join(OUTDIR, "correlation_matrix_partial.png")
    plot_correlation_matrix(pc, file_path, varlabels=nodes, \
        cbar_label=r"Partial correlation $R$", vmin=-0.5, vmax=0.5, \
        cmap="RdBu_r", dpi=300.0)
    # Plot corrected correlation and partial correlation in the same plot.
    file_path = os.path.join(OUTDIR, "correlation_matrix_combined_corrected.png")
    plot_correlation_matrix(comb_sig, file_path, varlabels=nodes, \
        cbar_label=r"(Partial) Correlation $R$", vmin=-0.5, vmax=0.5, \
        cmap="RdBu_r", dpi=300.0)


    # BOOTSTRAPPING

    # Run through all iterations (only if we need to create or overwrite
    # existing bootstrap data).
    if OVERWRITE_NETWORK_BOOTSTRAP or not os.path.isfile(os.path.join(OUTDIR, "memmap_pc_bs.dat")):
        print("Running {} bootstrapping iterations...".format(N_BOOTSTRAP_ITERATIONS))

        # Create new 
        pc_bs = numpy.memmap(os.path.join(OUTDIR, "memmap_pc_bs.dat"), \
            dtype=numpy.float32, mode="w+", \
            shape=(pc.shape[0], pc.shape[1], N_BOOTSTRAP_ITERATIONS))
        pc_bs_p = numpy.memmap(os.path.join(OUTDIR, "memmap_pc_bs_p.dat"), \
            dtype=numpy.float32, mode="w+", \
            shape=(pc.shape[0], pc.shape[1], N_BOOTSTRAP_ITERATIONS))
    
        for i in range(N_BOOTSTRAP_ITERATIONS):
            if (i+1) % 200 == 0:
                print("\titeration {}".format(i+1))
            # Choose samples with replacement (i.e. potentially repeating indices).
            indices = numpy.random.rand(X_tot.shape[0]) * (X_tot.shape[0]-1)
            indices = numpy.round(indices, decimals=0).astype(int)
            # Compute the partial correlations for the current bootstrap sample.
            pc_bs[:,:,i], pc_bs_p[:,:,i], a = partial_corr(X_tot[indices,:], alpha=0)

    # Load existing bootstrap data.
    else:
        print("Loading existing bootstrapping data.")
        pc_bs = numpy.memmap(os.path.join(OUTDIR, "memmap_pc_bs.dat"), \
            dtype=numpy.float32, mode="r", \
            shape=(pc.shape[0], pc.shape[1], N_BOOTSTRAP_ITERATIONS))
        pc_bs_p = numpy.memmap(os.path.join(OUTDIR, "memmap_pc_bs_p.dat"), \
            dtype=numpy.float32, mode="r", \
            shape=(pc.shape[0], pc.shape[1], N_BOOTSTRAP_ITERATIONS))

    # Compute the confidence intervals, which would be the 2.5th and 97.5th
    # percentiles at alpha=0.05.
    print("Computing bootstrapped confidence intervals.")
    pc_bs_sorted = numpy.copy(pc_bs)
    pc_bs_sorted.sort(axis=2)
    low_i = int(numpy.round(((BOOTSTRAPPED_CI_ALPHA/2.0) * (pc_bs_sorted.shape[2]-1)), decimals=0))
    high_i = int(numpy.round(((1.0-BOOTSTRAPPED_CI_ALPHA/2.0) * (pc_bs_sorted.shape[2]-1)), decimals=0))
    ci_low = pc_bs_sorted[:,:,low_i]
    ci_high = pc_bs_sorted[:,:,high_i]
    ci_sig = ((ci_low<0) & (ci_high<0)) | ((ci_low>0) & (ci_high>0))
    
    # Plot the high and low confidence intervals in a correlation-matrix type
    # graph, but only for confidence intervals that do not include 0.
    ci_plot_m = numpy.zeros(ci_sig.shape, dtype=float)
    ci_plot_m[upper] = ci_high[upper]
    ci_plot_m[lower] = ci_low[lower]
    ci_plot_m[ci_sig==False] = 0.0
    file_path = os.path.join(OUTDIR, "bootstrapped_partial_correlation_ci.png")
    plot_correlation_matrix(ci_plot_m, file_path, varlabels=nodes, \
        cbar_label=r"Partial correlation $R$", vmin=-0.3, vmax=0.3, \
        cmap="RdBu_r", dpi=300.0)
    
    # Construct a list of connections and their average weights and confidence
    # intervals.
    print("Plotting connection weight estimates with confidence intervals for each possible connection.")
    connection = {"name":[], "weight":[], "m":[], "median":[], "ci_low":[], "ci_high":[]}
    for i in range(ci_low.shape[0]):
        for j in range(i+1, ci_low.shape[1]):
            connection["name"].append("{} - {}".format(nodes[i], nodes[j]))
            connection["weight"].append(pc[i,j])
            connection["m"].append(numpy.mean(pc_bs[i,j,:]))
            connection["median"].append(numpy.median(pc_bs[i,j,:]))
            connection["ci_low"].append(ci_low[i,j])
            connection["ci_high"].append(ci_high[i,j])
    # Combine CI into a single array with shape (2,N) with absolute +/- values
    # from the median, for plotting.
    connection["ci"] = numpy.zeros((2,len(connection["name"])), dtype=float)
    connection["ci"][0,:] = numpy.array(connection["ci_low"])
    connection["ci"][1,:] = numpy.array(connection["ci_high"])
    connection["ci"] = numpy.abs(connection["ci"] - connection["weight"])
    # Get the sorting order for the connections.
    csi = numpy.argsort(connection["weight"])
    
    # Plot all connections and their confidence intervals.
    fig, ax = pyplot.subplots(nrows=1, ncols=1, figsize=(5.0, 20.0), dpi=300.0)
    fig.subplots_adjust(left=0.5, bottom=0.02, right=0.95, top=0.99, wspace=0.27, hspace=0.01)
    x_ticks = numpy.arange(-0.3, 0.51, 0.1)
    y_ticks = range(1, csi.shape[0]+1)
    y_tick_labels = []
    for i, si in enumerate(csi):
        if (connection["ci_high"][si] < 0) & (connection["ci_low"][si] < 0):
            col = COLS["skyblue"][2]
        elif (connection["ci_high"][si] > 0) & (connection["ci_low"][si] > 0):
            col = COLS["scarletred"][2]
        else:
            col = COLS["aluminium"][2]
        ax.errorbar(connection["weight"][si], y_ticks[i], \
            xerr=numpy.abs(connection["ci"][:,si]).reshape(2,1), fmt='o', \
            elinewidth=1, capsize=1, color=col)
        y_tick_labels.append(connection["name"][si])
    ax.axvline(0.0, linestyle="dashed", color="k")
    ax.hlines(y_ticks, x_ticks[0], x_ticks[-1], colors="k", alpha=0.3, linestyles="dotted", linewidths=0.5)
    ax.set_xlim(x_ticks[0], x_ticks[-1])
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(map(str, numpy.round(x_ticks, decimals=1)), fontsize=10)
#    ax.set_xlabel("Connection weight estimate", fontsize=14)
    ax.set_ylim(y_ticks[0]-1, y_ticks[-1]+1)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_tick_labels, fontsize=9, rotation=10)
    fig.savefig(os.path.join(OUTDIR, "ordered_connections.png"))
    pyplot.close(fig)

    # Compute the probability of each connection existing.
    print("Computing and plotting connection probability.")
    p = numpy.nanmean((pc_bs_p < 0.05).astype(float), axis=2)
    # Compute the number of connections for each iteration.
    n_connection = numpy.sum((pc_bs_p[lower,:] < 0.05).astype(int), axis=0)
    p_connection = n_connection.astype(float) / float(numpy.sum(lower.astype(int)))
    i_95 = int(N_BOOTSTRAP_ITERATIONS * 0.95) - 1
    chance = numpy.sort(p_connection)[i_95]
    # Zero out all connection probabilities under chance.
    p_sig = numpy.copy(p)
    p_sig[p<chance] = 0.0
    
    # Create a text file to report the number of connections per iteration.
    with open(os.path.join(OUTDIR, "connection_probability.tsv"), "w") as f:
        f.write("\t".join(map(str, ["iteration", "percentile", "n_connections", "p_connection"])))
        for counter, i in enumerate(numpy.argsort(n_connection)):
            line = [i, numpy.round(100.0*(float(counter)/float(N_BOOTSTRAP_ITERATIONS)), decimals=2), n_connection[i], p_connection[i]]
            f.write("\n" + "\t".join(map(str, line)))
    
    # Plot the connection probabilities.
    p_plot_m = numpy.zeros(p.shape, dtype=float)
    p_plot_m[upper] = p[upper]
    p_plot_m[lower] = p_sig[lower]
    file_path = os.path.join(OUTDIR, "bootstrapped_connection_probability.png")
    plot_correlation_matrix(p_plot_m, file_path, varlabels=nodes, \
        cbar_label=r"Connection probability $p$", vmin=0, vmax=1.0, \
        cmap="viridis", dpi=300.0)


    # NETWORK PLOTS
    
    print("\nProducing network graphs.")

    # Compute the MDS and UMAP projection of node positions.
    print("Computing MDS and UMAP projections for node positions.")
    # Seed the random number generator to make sure that the scaling
    # shows the same layout. (MDS isn't deterministic, so while the
    # same result will come out in terms of inter-sample distances,
    # those samples will be rotated in different ways each time.)
    numpy.random.seed(1)
    # Apply dimensionality reduction on the transposed (subject, factor)
    # matrix.
    xy = {}
    for projection in ["MDS", "UMAP"]:
        xy[projection] = dim_reduction(X_tot.T, n_components=2, mode=projection)
        # Cluster the features (overrules thematic node grouping).
        if NETWORK_CLUSTER_NUMBER > 1:
            y = clustering(xy[projection], mode=NETWORK_CLUSTER_METHOD, \
                n_clusters=NETWORK_CLUSTER_NUMBER)
        # Scale the positions to be in the range [-1,1].
        spread = numpy.nanmax(xy[projection], axis=0) - numpy.nanmin(xy[projection], axis=0)
        xy[projection] = 1.0 - (((xy[projection] - numpy.nanmin(xy[projection], axis=0)) / spread) * 2.0)

    # Loop through two main groups:
    # The first is the cleanest network, with only connections whose
    # bootstrapped confidence interval does not contain 0.
    # The second is a network with L1 regularised network connections, for
    # which the LASSO alpha (lambda) parameter is determined through
    # cross-validation within each partial correlation.
    print("Creating network plot figures.")
    for graph_type in ["ci", "cv"]:

        # Create a graph for the current nodes and connection weights.
        if graph_type == "ci":
            connection_weights = numpy.copy(pc)
            connection_weights[ci_sig==False] = 0.0
        elif graph_type == "cv":
            connection_weights, _, _ = partial_corr(X_tot, alpha="cv", average_triangles=True)
        graph = create_graph(connection_weights, nodes, ignore_value=0)
        
        # Set node groupings for individual colouring.
        if (THEMATIC_CLUSTERS is not None) and (FACTOR_CLUSTERS == False):
            y = numpy.zeros((len(nodes)), dtype=int)
            for i, theme in enumerate(THEMES):
                for j in range(len(nodes)):
                    if nodes[j] in THEMATIC_CLUSTERS[theme]:
                        y[j] = i
            node_colours = numpy.zeros(len(nodes))
            for i, lbl in enumerate(graph.nodes()):
                node_colours[i] = y[nodes.index(lbl)]
        else:
            y = numpy.zeros((len(nodes)), dtype=int)
            node_colours = None
        
        # Plot graphs.
        var = { \
            "mental":  ["Anxiety", "Depression"], \
            "cognition":  ["Number sense", "Spatial STM", "Verbal STM", "Search", \
                "Fluid reasoning", "Inhibition", "Speed"], \
            "attitude":  ["Grit", "Conscientiousness", "Growth mindset"], \
            "edu":  ["Reading", "Sums"], \
            "ses":  ["Affluence", "Deprivation"], \
            "misc":  ["Calm home", "School liking", "Class distraction"], \
            }
    
        for graph_layout in ["spring", "circle", "spectral", "UMAP", "MDS", \
            "MDS_mental", "MDS_cognition", "MDS_attitude", "MDS_edu", "MDS_ses", "MDS_misc"]:
            
            print("\tPlotting graph type '{}' with layout '{}'".format( \
                graph_type, graph_layout))
    
            # Split the graph_layout if necessary.
            if "_" in graph_layout:
                graph_layout, var_group = graph_layout.split("_")
                voi = var[var_group]
            else:
                voi = None
    
            # Construct the file path.
            if voi is None:
                file_path = os.path.join(OUTDIR, "{}_network_plot_{}.png".format( \
                    graph_type, graph_layout))
            else:
                file_path = os.path.join(OUTDIR, "{}_network_plot_{}_{}.png".format( \
                    graph_type, graph_layout, var_group))
    
            # Save node positions in the expected format (dict).
            if graph_layout in ["MDS", "UMAP"]:
                pos = {}
                for i, lbl in enumerate(nodes):
                    pos[lbl] = xy[graph_layout][i,:]
            else:
                import networkx
                if graph_layout == "spring":
                    pos = networkx.spring_layout(graph)
                elif graph_layout == "circle":
                    pos = networkx.circular_layout(graph)
                elif graph_layout == "spectral":
                    pos = networkx.spectral_layout(graph)
            
            # Set graph details.
            if (graph_layout == "MDS") and (voi is None):
                dpi = 900.0
            else:
                dpi = 300.0

            # Draw all nodes.
            if FACTOR_CLUSTERS:
                draw_nodes = False
                # Create a new figure.
                fig, ax = pyplot.subplots(figsize=(11.2,10.0), dpi=dpi, \
                    nrows=1, ncols=1)
                fig.subplots_adjust(left=0.03, bottom=0.03, right=0.9, \
                    top=0.97)
            else:
                draw_nodes = True
                ax = None
            
            # Plot the network graph.
            plot_graph(graph, nodes, file_path, graph_layout=graph_layout, \
                pos=pos, node_grouping=node_colours, variables_of_interest=voi, \
                vmin=-0.3, vmax=0.3, cmap=pyplot.cm.RdBu_r, node_col="#c5f1c5", \
                dpi=dpi, ax=ax, draw_nodes=draw_nodes)
            
            # Manually draw nodes, then save and close.
            if FACTOR_CLUSTERS:
                # Tertiary colours.
                # ORANGE
                # #FF7F00 	
                # ROSE / BRIGHT PINK
                # #FF007F 	
                # CHARTREUSE
                # #7FFF00 	
                # VIOLET
                # #7F00FF 	
                # SPRING GREEN
                # #00FF7F 	
                # AZURE
                # #007FFF
                # Set colour maps for all factors.
                if len(FACTORS) == 5:
                    factor_cols = {
                        "Cognition":        "#7FFF00",  # chartreuse
                        "Attitude":         "#007FFF",  # azure
                        "Mental Health":    "#FF007F",  # rose
                        "Speed":  "#00FF7F",  # spring green
                        "SES":              "#FF7F00",  # orange
                        }
                elif len(FACTORS) == 3:
                    factor_cols = {
                        "Cognition":        "cyan",
                        "Attitude":         "magenta",
                        "Mental Health":    "yellow",
                        }
                # Get the highest possible loading sum.
                l_max = numpy.max([numpy.sum(numpy.abs(FACTOR_SCORES[lbl])) \
                    for lbl in FACTOR_SCORES.keys()])
                # Loop through all factors.
                for fi, factor_name in enumerate(FACTORS):
                    # Grab all alphas (absolute node weight for this factor).
                    cols = []
                    x_pos = []
                    y_pos = []
                    for i, lbl in enumerate(graph.nodes()):
                        c = list(matplotlib.colors.to_rgba(factor_cols[factor_name]))
                        c[-1] = numpy.abs(FACTOR_SCORES[lbl][fi])/l_max
                        cols.append(tuple(c))
                        x_pos.append(pos[lbl][0])
                        y_pos.append(pos[lbl][1])
                    # Draw all nodes.
                    ax.scatter(x_pos, y_pos, s=2000, marker="o", c=cols)
                # Draw nodes outside of view for the benefit of the legend.
                x_pos = ax.get_xlim()[0] - 100
                y_pos = ax.get_ylim()[0] - 100
                ax.scatter(x_pos, y_pos, s=50, marker="o", c="#FFFFFF", \
                    alpha=0.0, label="Factors")
                for factor_name in FACTORS:
                    ax.scatter(x_pos, y_pos, s=100, marker="o", \
                        c=factor_cols[factor_name], alpha=1.0, \
                        label=factor_name)
                ax.legend(loc="lower left", fontsize=FONTSIZE["legend"])
                # Save and close figure.
                fig.savefig(file_path)
                pyplot.close(fig)
    
    # NETWORKS AT DIFFERENT ALPHA (LAMBDA) VALUES
    print("\nComputing and plotting networks at different alpha (lambda) values.")
    # Then take the highest correlation; this will be the maximum
    # alpha value.
    max_alpha = numpy.max(numpy.abs(fc[upper]))
    # Create numbers along a log-linear space.
    alphas = numpy.logspace(0, 1, num=10)
    # Transform the numbers to be between 0 and the maximum alpha.
    alphas = (alphas - alphas[0]) / (alphas[-1]-alphas[0]) * max_alpha
    # Round the alphas, so their value matches the reported value.
    alphas = numpy.round(alphas, decimals=3)
    # Convert to a list, so that we can add the cross-validation option too.
    alphas = list(alphas)
    alphas.append("cv")
    # Create plots for different regularisation levels.
    for alpha in alphas:
        if type(alpha) != str:
            str_alpha = "{}".format(str(numpy.round(alpha, decimals=3)).ljust(5,"0"))
        else:
            str_alpha = alpha
        print("\tConstructing network with alpha={}.".format(str_alpha))
        # Compute the weights.
        connection_weights, p, a = partial_corr(X_tot, alpha=alpha, cv_folds=5, average_triangles=True)
        # Create the graph.
        graph = create_graph(connection_weights, nodes, ignore_value=0)
        # Plot the network.
        file_path = os.path.join(OUTDIR, "network_plot_MDS_alpha-{}.png".format(str_alpha))
        plot_graph(graph, nodes, file_path, graph_layout=None, pos=pos, \
            node_grouping=node_colours, variables_of_interest=None, \
            vmin=-0.3, vmax=0.3, cmap=pyplot.cm.RdBu_r, node_col="#c5f1c5")


if DO_NETWORK_CLUSTERING:
    
    print("\nRunning network feature clustering...")

    # GRAPH NODE CLUSTERING
    numpy.random.seed(1)
    # Apply dimensionality reduction on the transposed (subject, factor)
    # matrix.
    xy = dim_reduction(X_tot.T, n_components=2, mode=NETWORK_DIM_REDUCTION)
    # Cluster the features.
    convenience_clustering(xy, NETWORK_CLUSTER_METHOD, 10, \
        os.path.join(OUTDIR, "network_feature_clusters_{}_{}".format( \
        NETWORK_DIM_REDUCTION, NETWORK_CLUSTER_METHOD)))
