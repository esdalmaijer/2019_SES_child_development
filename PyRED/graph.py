import os
import copy

import numpy
import scipy.stats
import matplotlib
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable

from sklearn.linear_model import Lasso, LassoCV, LinearRegression

import networkx


# Colours for plotting from the Tango colour scheme. They start repeating
# after 20
PLOTCOLS = {\
    -1:'#000000', # black

    0:'#4e9a06', # green
    1:'#204a87', # blue
    2:'#5c3566', # purple
    3:'#c4a000', # yellow
    4:'#8f5902', # chocolate
    5:'#ce5c00', # orange
    6:'#a40000', # red

    7:'#73d216', # green
    8:'#3465a4', # blue
    9:'#75507b', # purple
    10:'#edd400', # yellow
    11:'#c17d11', # chocolate
    12:'#f57900', # orange
    13:'#cc0000', # red

    14:'#8ae234', # green
    15:'#729fcf', # blue
    16:'#ad7fa8', # purple
    17:'#fce94f', # yellow
    18:'#e9b96e', # chocolate
    19:'#fcaf3e', # orange
    20:'#ef2929', # red
    }
for i in range(21, 201):
    PLOTCOLS[i] = copy.deepcopy(PLOTCOLS[i % 21])


# # # # #
# GRAPH FUNCTIONS

def create_graph(connection_weights, node_labels=None, ignore_value=0):
    
    """Create a networkx.Graph instance with the node labels and connection
    weights passed to this function. This is very much a convenience function,
    and as such only a very light wrapper around networkx functionality.
    
    Arguments
    
    connection_weights  -   A numpy.array with shape (N,N), where N is the
                            number of nodes. This would usually be a (partial)
                            correlation matrix. The matrix is expected to be
                            symmetrical across the diagonal.
    
    Keyword Arguments
    
    node_labels         -   A list of strings that indicate the names of nodes
                            in the same order as connection_weights. Pass None
                            to assign numbers. Default = None
    
    ignore_value        -   Value that should be ignore if it appears in
                            connection_weights. Default = 0
    
    Returns
    
    graph               -   A networkx.Graph instance
    """
    
    # Compute the number of features.
    n_features, _ = connection_weights.shape

    # Create node labels if none were provided.
    if node_labels is None:
        node_labels = map(str, range(1, n_features+1))

    # Create an empty graph.
    graph = networkx.Graph()

    # Add all the nodes.
    for i, lbl in enumerate(node_labels):
        graph.add_node(lbl)

    # Add all the edges.
    for i, lbl_i in enumerate(node_labels):
        for j, lbl_j in enumerate(node_labels):
            # Only do this for one half of the weight matrix, as the other
            # half contains the same data.
            if i > j:
                # Only include weights that should not be ignored.
                include_weight = True
                if numpy.isnan(ignore_value):
                    if numpy.isnan(connection_weights[i,j]):
                        include_weight = False
                else:
                    if connection_weights[i,j] == ignore_value:
                        include_weight = False

                if include_weight:
                    lbl_j = node_labels[j]
                    graph.add_edge(lbl_i, lbl_j, weight=connection_weights[i,j])
    
    return graph


def partial_corr(X, alpha=0, cv_folds=5, average_triangles=True):
    
    """Computes partial correlations, with optional L1 regularisation through
    LASSO.
    
    Arguments
    
    X                   -   A numpy.array with shape (N,M), where N is the
                            number of observations, and M is the number of
                            features. Is is good practice to standardise the
                            values that go into a regression.
    
    Keyword Arguments
    
    alpha               -   Float that multiplies the L1 term (sometimes
                            referred to as lambda). Two special cases exits:
                            pass "cv" for cross-validated selection of alpha,
                            or pass 0 for non-regularised linear regression.
                            Default = 0
    
    cv_folds            -   Integer that determines the number of folds used
                            in cross-validation when alpha="cv". Default = 5
    
    average_triangles   -   Bool indicating whether the outcomes of R_ij and
                            R_ji should be averaged (sometimes they turn out
                            subtly differently). Default = True
    
    returns
    
    [r, p, a]
    r                   -   A numpy.array of shape (M,M), where M is the number
                            of observations in the passed data (X.shape[1]).
                            The values are floats between -1 and 1, and reflect
                            the correlation coefficient.
    p                   -   A numpy.array of shape (M,M), where M is the number
                            of observations in the passed data (X.shape[1]).
                            The values are p values for the R values.
    a                   -   A numpy.array of shape (M,M), where M is the number
                            of observations in the passed data (X.shape[1]).
                            The values are alpha values, which will only be
                            different in the case of cross-validation to find
                            the optimal alpha level. (When alpha="cv".)
    """

    # Count the number of observations and features.
    n_obs, n_feat = X.shape
    
    # Start with an empty correlation matrix.
    r = numpy.zeros((n_feat, n_feat), dtype=float) * numpy.NaN

    # If we're using cross-validation, save the alpha levels.
    a = numpy.ones((n_feat, n_feat), dtype=float)
    if alpha == "cv":
        a *=  numpy.NaN
    else:
        a *= alpha
    
    # Loop through all features along one axis.
    for i in range(n_feat):

        # Create a Boolean vector to mask everything but i.
        sel_i = numpy.ones(n_feat, dtype=bool)
        sel_i[i] = False
        # Perform a LASSO regularisation with cross validation to find
        # the optimal regularisation parameter (alpha in scikit-learn).
        if alpha == "cv":
            model_i = LassoCV(cv=cv_folds, fit_intercept=True)
        # Perform a regular linear regression when alpha==0.
        elif alpha == 0:
            model_i = LinearRegression(fit_intercept=True)
        # Perform a LASSO with a single alpha (lambda) value.
        else:
            model_i = Lasso(alpha=alpha, fit_intercept=True)
        # Run the chosen type of linear regression.
        model_i.fit(X[:,sel_i], X[:,i])
        # Compute the predicted outcomes on the basis of the current fit.
        y_i_pred = model_i.intercept_ + numpy.sum(model_i.coef_ * X[:,sel_i], axis=1)
        # Compute the residuals for the regression.
        e_i = X[:,i] - y_i_pred

        # Loop through all features from the current one along the other axis.
        for j in range(i, n_feat):
            
            # Skip auto-correlations, as they are 1 by definition.
            if i == j:
                r[i,j] = 1.0
                if alpha == "cv":
                    a[i,j] = 0.0
                continue
            
            # Create a Boolean vector to mask everything but j.
            sel_j = numpy.ones(n_feat, dtype=bool)
            sel_j[j] = False
            # Perform a LASSO regularisation with cross validation to find
            # the optimal regularisation parameter (alpha in scikit-learn).
            if alpha == "cv":
                model_j = LassoCV(cv=cv_folds, fit_intercept=True)
            # Perform a regular linear regression when alpha==0.
            elif alpha == 0:
                model_j = LinearRegression(fit_intercept=True)
            # Perform a LASSO with a single alpha (lambda) value.
            else:
                model_j = Lasso(alpha=alpha, fit_intercept=True)
            # Run the chosen type of linear regression.
            model_j.fit(X[:,sel_j], X[:,j])
            # Compute the predicted outcomes on the basis of the current fit.
            y_j_pred = model_j.intercept_ + numpy.sum(model_j.coef_ * X[:,sel_j], axis=1)
            # Compute the residuals for the regression.
            e_j = X[:,j] - y_j_pred
            
            # Find the right beta for i and j.
            if i > j:
                b_i = model_i.coef_[j]
                b_j = model_j.coef_[i-1]
            else:
                b_i = model_i.coef_[j-1]
                b_j = model_j.coef_[i]
            
            # Compute R.
            r_ij = (b_i * numpy.std(e_j)) / numpy.std(e_i)
            r_ji = (b_j * numpy.std(e_i)) / numpy.std(e_j)
            
            # Save in the matrix, either imbalanced or averaged.
            if average_triangles:
                r[i,j] = (r_ij+r_ji) / 2.0
                r[j,i] = (r_ij+r_ji) / 2.0
            else:
                r[i,j] = r_ij
                r[j,i] = r_ji
            
            # Save the fitted alphas.
            if alpha == "cv":
                a[i,j] = model_i.alpha_
                a[j,i] = model_j.alpha_

    # Compute a t value for every R, so we can compute p values. There will be
    # R values of 1 (e.g. on the diagonal), which would result in divisions
    # by 0. They should be assigned t values of infinite, which will in turn
    # result in p values of 0.
    t = numpy.zeros(r.shape, dtype=float)
    numerator = r * numpy.sqrt(n_obs - 2)
    denominator = numpy.sqrt(1.0 - r**2)
    t[denominator>0] = numerator[denominator>0] / denominator[denominator>0]
    t[denominator==0] = numpy.inf
    # Compute a p value for every t value.
    p = 2.0 * (1.0 - scipy.stats.t.cdf(numpy.abs(t), df=n_obs-1))

    return r, p, a


# # # # #
# PLOTTING FUNCTIONS

def plot_correlation_matrix(M, file_path, varlabels=None, \
    cbar_label="Correlation (R)", vmin=-1.0, vmax=1.0, cmap="RdBu_r", dpi=300.0):
    

    # Count the number of features.
    n_features, _ = M.shape

    # Create variable labels if none were passed.
    if varlabels is None:
        varlabels = map(str, range(1, n_features+1))

    # Create a new figure.
    fig, ax = pyplot.subplots(nrows=1, ncols=1, figsize=(11.2,10.0), dpi=dpi)
    # Add enough space around the sides to write variable names in.
    fig.subplots_adjust(left=0.12, bottom=0.18, right=0.95, top=0.99, wspace=0.01, hspace=0.01)
    im = ax.imshow(M, vmin=vmin, vmax=vmax, cmap=cmap, \
        aspect="equal", interpolation="none", origin="upper")
    # Annotate R values for every correlation.
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if M[i,j] != 0.0:
                ax.annotate("%.2f" % (numpy.round(M[i,j], decimals=2)), \
                    (j-0.4,i+0.1), color="#FFFFFF", alpha=1.0, fontsize=10)
    # Add colourbar.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
    cbar.set_ticks(numpy.arange(vmin, vmax+0.01, 0.1))
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label(cbar_label, fontsize=20)
    # Draw a line to make the separation clear.
    x = [-1, M.shape[0]]
    y = [-1, M.shape[0]]
    ax.plot(x, y, "--", color="black", alpha=0.2, lw=3)
    # Finish the plot.
    ax.set_xlim(-0.5, M.shape[1]-0.5)
    ax.set_ylim(M.shape[0]-0.5, -0.5)
    ax.set_yticks(range(len(varlabels)))
    ax.set_yticklabels(varlabels, fontsize=12)
    ax.set_xticks(range(len(varlabels)))
    ax.set_xticklabels(varlabels, rotation="vertical", fontsize=12)
    # Save and close the figure.
    fig.savefig(file_path)
    pyplot.close(fig)


def plot_graph(graph, node_labels, file_path, graph_layout="spring", pos=None, \
    node_grouping=None, variables_of_interest=None, vmin=-0.3, vmax=0.3, \
    cmap=pyplot.cm.RdBu_r, node_col="#c5f1c5", dpi=300.0, ax=None, \
    draw_nodes=True):
   
    # If no custom positions were passed, generate positions for the nodes.
    if pos is None:
        if graph_layout == "spring":
            pos = networkx.spring_layout(graph)
        elif graph_layout == "circle":
            pos = networkx.circular_layout(graph)
        elif graph_layout == "spectral":
            pos = networkx.spectral_layout(graph)
        else:
            raise Exception("Unrecognised graph_layout '%s'; please use custom xy values, or use one of: 'spring', 'circle', 'spectral'" \
                % (graph_layout))

    # Create a new figure.
    if ax is None:
        save_and_close = True
        fig, ax = pyplot.subplots(figsize=(11.2,10.0), dpi=dpi, nrows=1, \
            ncols=1)
        fig.subplots_adjust(left=0.03, bottom=0.03, right=0.9, top=0.97)
    else:
        save_and_close = False
        fig = ax.get_figure()
    # Draw the edges. All edges are coloured according to their weight.
    if variables_of_interest is None:
        colors = [d["weight"] for (u,v,d) in graph.edges(data=True)]
    else:
        # Edges connected to the factors of interest are coloured by their
        # weight, and the other edges are set to 0. (This produces a light
        # grey line for those edges.)
        colors = []
        for u,v,d in graph.edges(data=True):
            if (v in variables_of_interest) or (u in variables_of_interest):
                colors.append(d["weight"])
            else:
                colors.append(0.0)
    im = networkx.draw_networkx_edges(graph, pos, edge_color=colors, alpha=0.5, \
        edge_cmap=cmap, edge_vmin=vmin, edge_vmax=vmax, width=7, ax=ax)
    # The following is necessary, as otherwise it somehow jumps back to 
    # viridis (at least on my Python 3 install).
    im.set_cmap(cmap)
    im.set_clim([vmin, vmax])
    # Draw the individual nodes.
    if draw_nodes:
        if node_grouping is not None:
            # Draw the nodes to illustrate cluster membership.
            networkx.draw_networkx_nodes(graph, pos, node_color=node_grouping, \
                node_size=1500, cmap=pyplot.cm.Pastel2_r, ax=ax)
        else:
            # Pastel pink: #ffd1dc
            # Pastel orange: #ffb347
            # Pastel navy: #779ecb
            # Pastel(ish, light) green: #c5f1c5
            networkx.draw_networkx_nodes(graph, pos, node_color=node_col, ax=ax)
    # Draw the node labels.
    networkx.draw_networkx_labels(graph, pos, font_family="ubuntu", \
        font_color="k", font_size=16, ax=ax)
    # Finish the plot.
    ax.set_xlim([-1.35,1.2])
    ax.set_ylim([-1.1,1.1])
    ax.set_xticks([])
    ax.set_yticks([])
    # Add colourbar (but only if there are any colours).
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    if len(colors) > 0:
        cbar = fig.colorbar(im, cax=cax, orientation='vertical')
        cbar.set_ticks(numpy.arange(vmin, vmax+0.01, 0.1))
        cbar.ax.tick_params(labelsize=12)
        cbar.set_label("Connection strength (partial correlation)", fontsize=20)
        # When plotting a colourbar for plots with transparency, lines appear
        # in the colourbar. The following lines correct that by resetting the
        # transparency, and then redrawing the colour bar.
        cbar.set_alpha(1.0)
        cbar.draw_all()
    # Save the figure.
    if save_and_close:
        fig.savefig(file_path)
        pyplot.close(fig)
    
