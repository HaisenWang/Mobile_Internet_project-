import warnings
from math import sqrt, floor
import numpy as np
from scipy import linalg
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.covariance import LedoitWolf
from sklearn.utils import deprecated
import signal_0
from extmath import is_spd
from nilearn.input_data import NiftiMapsMasker
from nilearn import plotting
from nilearn import datasets

def _check_square(matrix):

    if matrix.ndim != 2 or (matrix.shape[0] != matrix.shape[-1]):

        raise ValueError('Expected a square matrix, got array of shape'

                         ' {0}.'.format(matrix.shape))

def _check_spd(matrix):

    if not is_spd(matrix, decimal=7):

        raise ValueError('Expected a symmetric positive definite matrix.')

def _form_symmetric(function, eigenvalues, eigenvectors):

    return np.dot(eigenvectors * function(eigenvalues), eigenvectors.T)

def _map_eigenvalues(function, symmetric):

    eigenvalues, eigenvectors = linalg.eigh(symmetric)

    return _form_symmetric(function, eigenvalues, eigenvectors)

def _geometric_mean(matrices, init=None, max_iter=10, tol=1e-7):

    n_features = matrices[0].shape[0]

    for matrix in matrices:

        _check_square(matrix)

        if matrix.shape[0] != n_features:

            raise ValueError("Matrices are not of the same shape.")

        _check_spd(matrix)

    matrices = np.array(matrices)

    if init is None:

        gmean = np.mean(matrices, axis=0)

    else:

        _check_square(init)

        if init.shape[0] != n_features:

            raise ValueError("Initialization has incorrect shape.")

        _check_spd(init)

        gmean = init

    norm_old = np.inf

    step = 1.

    for n in range(max_iter):

        vals_gmean, vecs_gmean = linalg.eigh(gmean)

        gmean_inv_sqrt = _form_symmetric(np.sqrt, 1. / vals_gmean, vecs_gmean)

        whitened_matrices = [gmean_inv_sqrt.dot(matrix).dot(gmean_inv_sqrt)

                             for matrix in matrices]

        logs = [_map_eigenvalues(np.log, w_mat) for w_mat in whitened_matrices]

        logs_mean = np.mean(logs, axis=0)

        if np.any(np.isnan(logs_mean)):

            raise FloatingPointError("Nan value after logarithm operation.")

        norm = np.linalg.norm(logs_mean)

        vals_log, vecs_log = linalg.eigh(logs_mean)

        gmean_sqrt = _form_symmetric(np.sqrt, vals_gmean, vecs_gmean)

        gmean = gmean_sqrt.dot(

            _form_symmetric(np.exp, vals_log * step, vecs_log)).dot(gmean_sqrt)

        if norm < norm_old:

            norm_old = norm

        elif norm > norm_old:

            step = step / 2.

            norm = norm_old

        if tol is not None and norm / gmean.size < tol:

            break

    if tol is not None and norm / gmean.size >= tol:

        warnings.warn("Maximum number of iterations {0} reached without "

                      "getting to the requested tolerance level "

                      "{1}.".format(max_iter, tol))

    return gmean

@deprecated("Function 'sym_to_vec' has been renamed to "

            "'sym_matrix_to_vec' and will be removed in future releases. ")

def sym_to_vec(symmetric, discard_diagonal=False):

    return sym_matrix_to_vec(symmetric=symmetric,

                             discard_diagonal=discard_diagonal)





def sym_matrix_to_vec(symmetric, discard_diagonal=False):

    if discard_diagonal:

        tril_mask = np.tril(np.ones(symmetric.shape[-2:]), k=-1).astype(

            np.bool)

        return symmetric[..., tril_mask]

    scaling = np.ones(symmetric.shape[-2:])

    np.fill_diagonal(scaling, sqrt(2.))

    tril_mask = np.tril(np.ones(symmetric.shape[-2:])).astype(np.bool)

    return symmetric[..., tril_mask] / scaling[tril_mask]





def vec_to_sym_matrix(vec, diagonal=None):

    n = vec.shape[-1]

    n_columns = (sqrt(8 * n + 1) - 1.) / 2

    if diagonal is not None:

        n_columns += 1



    if n_columns > floor(n_columns):

        raise ValueError(

            "Vector of unsuitable shape {0} can not be transformed to "

            "a symmetric matrix.".format(vec.shape))



    n_columns = int(n_columns)

    first_shape = vec.shape[:-1]

    if diagonal is not None:

        if diagonal.shape[:-1] != first_shape or diagonal.shape[-1] != n_columns:

            raise ValueError("diagonal of shape {0} incompatible with vector "

                             "of shape {1}".format(diagonal.shape, vec.shape))

    sym = np.zeros(first_shape + (n_columns, n_columns))

    skip_diagonal = (diagonal is not None)

    mask = np.tril(np.ones((n_columns, n_columns)), k=-skip_diagonal).astype(

        np.bool)

    sym[..., mask] = vec

    sym.swapaxes(-1, -2)[..., mask] = vec

    mask.fill(False)

    np.fill_diagonal(mask, True)

    if diagonal is not None:

        sym[..., mask] = diagonal

    sym[..., mask] *= sqrt(2)

    return sym





def cov_to_corr(covariance):

    diagonal = np.atleast_2d(1. / np.sqrt(np.diag(covariance)))

    correlation = covariance * diagonal * diagonal.T

    np.fill_diagonal(correlation, 1.)

    return correlation





def prec_to_partial(precision):

    partial_correlation = -cov_to_corr(precision)

    np.fill_diagonal(partial_correlation, 1.)

    return partial_correlation





class ConnectivityMeasure(BaseEstimator, TransformerMixin):

    def __init__(self, cov_estimator=LedoitWolf(store_precision=False),

                 kind='covariance', vectorize=False, discard_diagonal=False):

        self.cov_estimator = cov_estimator

        self.kind = kind

        self.vectorize = vectorize

        self.discard_diagonal = discard_diagonal



    def _check_input(self, X):

        if not hasattr(X, "__iter__"):

            raise ValueError("'subjects' input argument must be an iterable. "

                             "You provided {0}".format(X.__class__))



        subjects_types = [type(s) for s in X]

        if set(subjects_types) != set([np.ndarray]):

            raise ValueError("Each subject must be 2D numpy.ndarray.\n You "

                             "provided {0}".format(str(subjects_types)))



        subjects_dims = [s.ndim for s in X]

        if set(subjects_dims) != set([2]):

            raise ValueError("Each subject must be 2D numpy.ndarray.\n You"

                             "provided arrays of dimensions "

                             "{0}".format(str(subjects_dims)))



        features_dims = [s.shape[1] for s in X]

        if len(set(features_dims)) > 1:

            raise ValueError("All subjects must have the same number of "

                             "features.\nYou provided: "

                             "{0}".format(str(features_dims)))



    def fit(self, X, y=None):

        self._fit_transform(X, do_fit=True)

        return self



    def _fit_transform(self, X, do_transform=False, do_fit=False):

        self._check_input(X)

        if do_fit:

            self.cov_estimator_ = clone(self.cov_estimator)


        if self.kind == 'correlation':

            covariances_std = [self.cov_estimator_.fit(

                signal_0._standardize(x, detrend=False, normalize=True)

                ).covariance_ for x in X]

            connectivities = [cov_to_corr(cov) for cov in covariances_std]

        else:

            covariances = [self.cov_estimator_.fit(x).covariance_ for x in X]

            if self.kind in ('covariance', 'tangent'):

                connectivities = covariances

            elif self.kind == 'precision':

                connectivities = [linalg.inv(cov) for cov in covariances]

            elif self.kind == 'partial correlation':

                connectivities = [prec_to_partial(linalg.inv(cov))

                                  for cov in covariances]

            else:

                raise ValueError('Allowed connectivity kinds are '

                                 '"correlation", '

                                 '"partial correlation", "tangent", '

                                 '"covariance" and "precision", got kind '

                                 '"{}"'.format(self.kind))

        if do_fit:

            if self.kind == 'tangent':

                self.mean_ = _geometric_mean(covariances, max_iter=30, tol=1e-7)

                self.whitening_ = _map_eigenvalues(lambda x: 1. / np.sqrt(x),

                                                self.mean_)

            else:

                self.mean_ = np.mean(connectivities, axis=0)

                # Fight numerical instabilities: make symmetric

                self.mean_ = self.mean_ + self.mean_.T

                self.mean_ *= .5


        if do_transform:

            if self.kind == 'tangent':

                connectivities = [_map_eigenvalues(np.log, self.whitening_.dot(

                                                   cov).dot(self.whitening_))

                                    for cov in connectivities]



            connectivities = np.array(connectivities)

            if self.vectorize:

                connectivities = sym_matrix_to_vec(

                    connectivities, discard_diagonal=self.discard_diagonal)



        return connectivities



    def fit_transform(self, X, y=None):

        if self.kind == 'tangent':

            if not len(X) > 1:

                raise ValueError("Tangent space parametrization can only "

                    "be applied to a group of subjects, as it returns "

                    "deviations to the mean. You provided %r" % X

                    )

        return self._fit_transform(X, do_fit=True, do_transform=True)





    def transform(self, X):

        self._check_fitted()

        return self._fit_transform(X, do_transform=True)



    def _check_fitted(self):

        if not hasattr(self, "cov_estimator_"):

            raise ValueError('It seems that {0} has not been fitted. '

                             'You must call fit() before calling '

                             'transform().'.format(self.__class__.__name__)

                             )



    def inverse_transform(self, connectivities, diagonal=None):

        self._check_fitted()

        connectivities = np.array(connectivities)

        if self.vectorize:

            if self.discard_diagonal:

                if diagonal is None:

                    if self.kind in ['correlation', 'partial correlation']:

                        diagonal = np.ones((connectivities.shape[0],

                                            self.mean_.shape[0])) / sqrt(2.)

                    else:

                        raise ValueError("diagonal values has been discarded "

                                         "and are unknown for {0} kind, can "

                                         "not reconstruct connectivity "

                                         "matrices.".format(self.kind))



            connectivities = vec_to_sym_matrix(connectivities,

                                               diagonal=diagonal)

        if self.kind == 'tangent':

            mean_sqrt = _map_eigenvalues(lambda x: np.sqrt(x), self.mean_)

            connectivities = [mean_sqrt.dot(

                _map_eigenvalues(np.exp, displacement)).dot(mean_sqrt)

                for displacement in connectivities]

            connectivities = np.array(connectivities)



        return connectivities

if __name__ == '__main__':
    atlas = datasets.fetch_atlas_msdl()
    atlas_filename = atlas['maps']
    labels = atlas['labels']
    data = datasets.fetch_adhd(n_subjects=1)

    print('First subject resting-state nifti image (4D) is located at: %s' %
          data.func[0])

    masker = NiftiMapsMasker(maps_img=atlas_filename, standardize=True,
                             memory='nilearn_cache', verbose=5)

    time_series = masker.fit_transform(data.func[0],
                                       confounds=data.confounds)


    correlation_measure = ConnectivityMeasure(kind='correlation')
    correlation_matrix = correlation_measure.fit_transform([time_series])[0]
    np.fill_diagonal(correlation_matrix, 0)
    plotting.plot_matrix(correlation_matrix, labels=labels, colorbar=True,
                         vmax=0.8, vmin=-0.8)
    coords = atlas.region_coords
    plotting.plot_connectome(correlation_matrix, coords,
                             edge_threshold="80%", colorbar=True)

    plotting.show()
