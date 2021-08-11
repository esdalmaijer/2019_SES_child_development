import numpy
import scipy.stats
from sklearn.linear_model import LinearRegression, LogisticRegression


def mediation_analysis(X, M, y):
    
    """Runs a mediation analysis of X on y, mediated by M. Steps:
    
    y = intercept_0 + c * X
    y = intercept_1 + c' * X + b * M
    M = intercept_3 + a * X
    
    after which follows that unmediated direct effect c is the sum of mediated
    effect c' and the product of a and b: c = a*b + c'
    
    Arguments
    
    X   -   NumPy array with shape (N,M) for the predictors, where N is the
            number of observations, and M the number of features. This is/are
            the independent variable(s).
    M   -   NumPy array with shape (N,M) for the mediators, where N is the
            number of observations, and M the number of features. This is/are
            the mediating variable(s).
    y   -   NumPy array with shape (N,M) for the predictors, where N is the
            number of observations, and M the number of features. This is/are
            the dependent variable(s).
    """
    
    # MEDIATORS ONLY MODEL
    # Run the linear regression of X and y.
    lr_0 = LinearRegression(fit_intercept=True, normalize=False)
    lr_0.fit(M, y)
    m = lr_0.coef_
    # Compute t and p for each beta.
    t_0_, p_0_ = compute_t_values(lr_0.coef_, M, y)
    # Compute confidence intervals for each beta.
    ci_0 = compute_confidence_intervals(lr_0.coef_, M, y, 0.05)
    # Compute r and p for the whole model.
    r_sq_0 = lr_0.score(M, y)
    r_0 = numpy.sqrt(r_sq_0)
    t_0 = r_0 * numpy.sqrt((y.shape[0]-2.0) / (1.0-r_sq_0))
    p_0 = 2.0 * (1.0 - scipy.stats.t.cdf(numpy.abs(t_0), y.shape[0]-1))
    
    # PREDICTORS ONLY MODEL
    # Run the linear regression of X and y.
    lr_1 = LinearRegression(fit_intercept=True, normalize=False)
    lr_1.fit(X, y)
    c = lr_1.coef_
    # Compute t and p for each beta.
    t_1_, p_1_ = compute_t_values(lr_1.coef_, X, y)
    # Compute confidence intervals for each beta.
    ci_1 = compute_confidence_intervals(lr_1.coef_, X, y, 0.05)
    # Compute r and p for the whole model.
    r_sq_1 = lr_1.score(X, y)
    r_1 = numpy.sqrt(r_sq_1)
    t_1 = r_1 * numpy.sqrt((y.shape[0]-2.0) / (1.0-r_sq_1))
    p_1 = 2.0 * (1.0 - scipy.stats.t.cdf(numpy.abs(t_1), y.shape[0]-1))
    
    # FULL MODEL (PREDICTORS AND MEDIATORS)
    # Combine independent and mediating variables.
    XM = numpy.hstack((X,M))
    # Run the regression of X+M and y.
    lr_2 = LinearRegression(fit_intercept=True, normalize=False)
    lr_2.fit(XM, y)
    c_prime = lr_2.coef_[:X.shape[1]]
    b = lr_2.coef_[X.shape[1]:]
    # Compute t and p for each beta.
    t_2_, p_2_ = compute_t_values(lr_2.coef_, XM, y)
    # Compute confidence intervals for each beta.
    ci_2 = compute_confidence_intervals(lr_2.coef_, XM, y, 0.05)
    # Compute r and p for the whole model.
    r_sq_2 = lr_2.score(XM, y)
    r_2 = numpy.sqrt(r_sq_2)
    t_2 = r_2 * numpy.sqrt((y.shape[0]-2.0) / (1.0-r_sq_2))
    p_2 = 2.0 * (1.0 - scipy.stats.t.cdf(numpy.abs(t_2), y.shape[0]-1))
    
    # MEDIATOR MODEL (predicting the mediators with the predictors)
    lr_3 = LinearRegression(fit_intercept=True, normalize=False)
    lr_3.fit(X, M)
    a = lr_3.coef_ # this has shape (M.shape[1], X.shape[1])
    # Compute the predicted mediators, and compute t and p for each beta.
    # I'm not using the compute_t_values helper function, because the
    # dimensions of multivariable X and M make my head hurt.
    # some reshaping.)
    df = y.shape[0] - 2
    m_pred = numpy.zeros(M.shape)
    e_med = numpy.zeros(a.shape)
    t_3_ = numpy.zeros(a.shape)
    p_3_ = numpy.zeros(a.shape)
    for i in range(m_pred.shape[1]):
        m_pred[:,i] = numpy.sum(X * lr_3.coef_[i,:], axis=1)
        e_med[i,:] = numpy.sqrt(numpy.sum((M[:,i]-m_pred[:,i])**2 , axis=0) \
            / (df * numpy.sum((X-numpy.mean(X,axis=0))**2, axis=0)))
        t_3_[i,:] = lr_3.coef_[i,:] / e_med[i,:]
        p_3_[i,:] = 2.0 * (1.0 - scipy.stats.t.cdf(numpy.abs(t_3_[i,:]), df))
    # Compute r and p for the whole model.
    r_sq_3 = lr_3.score(X, M)
    r_3 = numpy.sqrt(r_sq_3)
    t_3 = r_3 * numpy.sqrt((y.shape[0]-2.0) / (1.0-r_sq_3))
    p_3 = 2.0 * (1.0 - scipy.stats.t.cdf(numpy.abs(t_3), y.shape[0]-1))

    # Run a Sobel test for mediation.
    t_ab, p_ab, t_ab_all, p_ab_all = sobel_test(a, b, XM, lr_2.coef_, y, \
        X, M, lr_3.coef_)
    
    # Put data into a somewhat sensible structure. Sort of.
    result_dict = { \
        "c": c, \
        "c_prime": c_prime, \
        "b": b, \
        "a": a, \
        "m": m, \
        "M_y": { \
            "m": m, \
            "r_sq": r_sq_0, \
            "t_model": t_0, \
            "p_model": p_0, \
            "betas": lr_0.coef_, \
            "t_betas": t_0_, \
            "p_betas": p_0_, \
            }, \
        "X_y": { \
            "c": c, \
            "r_sq": r_sq_1, \
            "t_model": t_1, \
            "p_model": p_1, \
            "betas": lr_1.coef_, \
            "t_betas": t_1_, \
            "p_betas": p_1_, \
            }, \
        "XM_y": { \
            "b": b, \
            "c_prime": c_prime, \
            "r_sq": r_sq_2, \
            "t_model": t_2, \
            "p_model": p_2, \
            "betas": lr_2.coef_, \
            "t_betas": t_2_, \
            "p_betas": p_2_, \
            }, \
        "X_M": { \
            "a": a, \
            "c_prime": c_prime, \
            "r_sq": r_sq_3, \
            "t_model": t_3, \
            "p_model": p_3, \
            "betas": lr_3.coef_, \
            "t_betas": t_3_, \
            "p_betas": p_3_, \
            }, \
        "sobel": { \
            "t": t_ab, \
            "p": p_ab, \
            "t_all": t_ab_all, \
            "p_all": p_ab_all, \
            }
        }
    
    return result_dict
    

def compute_error(betas, X, y):
    
    """Internal use."""
    
    df = y.shape[0] - 2
    e = numpy.sqrt( \
        numpy.sum((y.reshape(y.shape[0],1) - X*betas)**2, axis=0) \
        / (df * numpy.sum((X-numpy.mean(X,axis=0))**2, axis=0)))
    
    return e

def compute_t_values(betas, X, y):
    
    """Internal use."""
    
    df = y.shape[0] - 2
    e = compute_error(betas, X, y)
    t = betas / e
    p = 2.0 * (1.0 - scipy.stats.t.cdf(numpy.abs(t), df))
    
    return t, p
    
def compute_confidence_intervals(betas, X, y, alpha):
    
    """Internal use."""
    
    df = y.shape[0] - 2
    e = compute_error(betas, X, y)
    ci_low = betas - e * scipy.stats.t.ppf((alpha/2.0), df)
    ci_high = betas - e * scipy.stats.t.ppf((1-(alpha/2.0)), df)
    
    return ci_low, ci_high

def sobel_test(a, b, XM, XM_y_betas, y, X, M, X_M_betas):
    
    """Internal use."""

    # Compute residuals and error for the "full" regression of X+M on y.
    res_full = y.reshape(y.shape[0],1) - XM*XM_y_betas.reshape(1,XM_y_betas.shape[0])
    b_var = numpy.var(res_full, axis=0) / numpy.sum((XM-numpy.mean(XM))**2, axis=0)
    # Truncate so that only the mediator-to-y mappings remain.
    b_var = b_var[X.shape[1]:]
    if len(b.shape) == 1:
        b = b.reshape((b.shape[0],1))
    if len(b_var.shape) == 1:
        b_var = b_var.reshape((b_var.shape[0],1))

    # Compute residuals and error for the regression of X on M.
    m_pred = numpy.zeros(M.shape)
    for i in range(m_pred.shape[1]):
        m_pred[:,i] = numpy.sum(X * X_M_betas[i,:], axis=1)
    res_med = M - m_pred
    a_var = numpy.zeros(a.shape)
    for i in range(a_var.shape[0]):
        a_var[i,:] = numpy.var(res_med[:,i], axis=0) / numpy.sum((X-numpy.mean(X))**2, axis=0)
    
    # Compute degrees of freedom.
    df = y.shape[0] - 2
    # Compute the product of coefficients and its error.
    ab = a*b
    e_ab = numpy.sqrt(a**2 * b_var + b**2 * a_var)
    # Compute t values through Sobel's method, and associated p values.
    t_ab = ab / e_ab
    p_ab = 2.0 * (1.0 - scipy.stats.t.cdf(numpy.abs(t_ab), df))
    # Sobel test for summed ab and pooled errors.
    t_ab_all = numpy.sum(a*b) / numpy.mean(e_ab)
    p_ab_all = 2.0 * (1.0 - scipy.stats.t.cdf(numpy.abs(t_ab_all), df))

    return t_ab, p_ab, t_ab_all, p_ab_all
