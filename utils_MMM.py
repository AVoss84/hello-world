import numpy as np
from tqdm import tqdm
from scipy.stats import multinomial, dirichlet
from nltk.tokenize import word_tokenize
from sklearn.base import BaseEstimator, TransformerMixin
import re, nltk, string
from copy import deepcopy
#from importlib import reload


class MultinomialExpectationMaximizer:
    def __init__(self, K, rtol=1e-3, max_iter=100, restarts=10):
        self._K = K
        self._rtol = rtol
        self._max_iter = max_iter
        self._restarts = restarts

    def compute_log_likelihood(self, X_test, alpha, beta):
        mn_probs = np.zeros(X_test.shape[0])
        for k in range(beta.shape[0]):
            mn_probs_k = self._get_mixture_weight(alpha, k) * self._multinomial_prob(X_test, beta[k])
            mn_probs += mn_probs_k
        mn_probs[mn_probs == 0] = np.finfo(float).eps
        return np.log(mn_probs).sum()

    def compute_bic(self, X_test, alpha, beta, log_likelihood=None):
        if log_likelihood is None:
            log_likelihood = self.compute_log_likelihood(X_test, alpha, beta)
        N = X_test.shape[0]
        return np.log(N) * (alpha.size + beta.size) - 2 * log_likelihood

    def compute_icl_bic(self, bic, gamma):
        classification_entropy = -(np.log(gamma.max(axis=1))).sum()
        return bic - classification_entropy

    def _multinomial_prob(self, counts, beta, log=False):
        """
        Evaluates the multinomial probability for a given vector of counts
        counts: (N x C), matrix of counts
        beta: (C), vector of multinomial parameters for a specific cluster k
        Returns:
        p: (N), scalar values for the probabilities of observing each count vector given the beta parameters
        """
        n = counts.sum(axis=-1)
        m = multinomial(n, beta)
        if log:
            return m.logpmf(counts)
        return m.pmf(counts)

    def _e_step(self, X, alpha, beta):
        """
        Performs E-step on MNMM model
        Each input is numpy array:
        X: (N x C), matrix of counts
        alpha: (K) or (NxK) in the case of individual weights, mixture component weights
        beta: (K x C), multinomial categories weights
        Returns:
        gamma: (N x K), posterior probabilities for objects clusters assignments
        """
        # Compute gamma
        N = X.shape[0]
        K = beta.shape[0]
        weighted_multi_prob = np.zeros((N, K))
        for k in range(K):
            weighted_multi_prob[:, k] = self._get_mixture_weight(alpha, k) * self._multinomial_prob(X, beta[k])

        # To avoid division by 0
        weighted_multi_prob[weighted_multi_prob == 0] = np.finfo(float).eps

        denum = weighted_multi_prob.sum(axis=1)
        gamma = weighted_multi_prob / denum.reshape(-1, 1)

        return gamma

    def _get_mixture_weight(self, alpha, k):
        return alpha[k]

    def _m_step(self, X, gamma):
        """
        Performs M-step on MNMM model
        Each input is numpy array:
        X: (N x C), matrix of counts
        gamma: (N x K), posterior probabilities for objects clusters assignments
        Returns:
        alpha: (K), mixture component weights
        beta: (K x C), mixture categories weights
        """
        # Compute alpha
        alpha = self._m_step_alpha(gamma)

        # Compute beta
        beta = self._m_step_beta(X, gamma)

        return alpha, beta

    def _m_step_alpha(self, gamma):
        alpha = gamma.sum(axis=0) / gamma.sum()
        return alpha

    def _m_step_beta(self, X, gamma):
        weighted_counts = gamma.T.dot(X)
        beta = weighted_counts / weighted_counts.sum(axis=-1).reshape(-1, 1)
        return beta

    def _compute_vlb(self, X, alpha, beta, gamma, eps = 1e-10):
        """
        Computes the variational lower bound
        X: (N x C), data points
        alpha: (K) or (NxK) with individual weights, mixture component weights
        beta: (K x C), multinomial categories weights
        gamma: (N x K), posterior probabilities for objects clusters assignments
        Returns value of variational lower bound
        """
        loss = 0
        for k in range(beta.shape[0]):
            weights = gamma[:, k]
            loss += np.sum(weights * (np.log(self._get_mixture_weight(alpha, k)+eps) + np.log(eps + self._multinomial_prob(X, beta[k]+eps))))
            #print('loss1', loss)

            #print('term1', weights * (np.log(self._get_mixture_weight(alpha, k)+eps)))
            #print('term2',np.log(self._multinomial_prob(X, beta[k] + eps)))

            loss -= np.sum(weights * np.log(weights+eps))
            #print('loss2', loss)
        return loss

    def _init_params(self, X):
        C = X.shape[1]
        weights = np.random.randint(1, 20, self._K)
        alpha = weights / weights.sum()
        beta = dirichlet.rvs([2 * C] * C, self._K)
        return alpha, beta

    def _train_once(self, X):
        '''
        Runs one full cycle of the EM algorithm
        :param X: (N, C), matrix of counts
        :return: The best parameters found along with the associated loss
        '''
        loss = float('inf')
        alpha, beta = self._init_params(X)
        gamma = None

        for it in range(self._max_iter):
            prev_loss = loss
            gamma = self._e_step(X, alpha, beta)
            alpha, beta = self._m_step(X, gamma)
            loss = self._compute_vlb(X, alpha, beta, gamma)
            print('Loss: %f' % loss)
            if it > 0 and np.abs((prev_loss - loss) / prev_loss) < self._rtol:
                    break
        return alpha, beta, gamma, loss

    def fit(self, X):
        '''
        Starts with random initialization *restarts* times
        Runs optimization until saturation with *rtol* reached
        or *max_iter* iterations were made.
        :param X: (N, C), matrix of counts
        :return: The best parameters found along with the associated loss
        '''
        best_loss = -float('inf')
        best_alpha = None
        best_beta = None
        best_gamma = None

        for it in range(self._restarts):
            print('iteration %i' % it)
            alpha, beta, gamma, loss = self._train_once(X)
            if loss > best_loss:
                print('better loss on iteration %i: %.10f' % (it, loss))
                best_loss = loss
                best_alpha = alpha
                best_beta = beta
                best_gamma = gamma

        return best_loss, best_alpha, best_beta, best_gamma


class IndividualMultinomialExpectationMaximizer(MultinomialExpectationMaximizer):
    def __init__(self, K, alpha_init, beta_init, household_ids, rtol=1e-3, max_iter=100, restarts=10):
        super().__init__(K, rtol, max_iter, restarts)
        self._household_ids = household_ids
        self._alpha_init = alpha_init
        self._beta_init = beta_init
        self._household_freqs = np.unique(household_ids, return_counts=True)[1]

    def _init_params(self, X):
        N = X.shape[0]
        alpha = np.vstack([self._alpha_init] * N)
        return alpha, self._beta_init

    def _get_mixture_weight(self, alpha, k):
        return alpha[:, k]

    def _m_step_alpha(self, gamma):
        """
        Performs M-step on MNMM model
        Each input is numpy array:
        X: (N x C), data points
        gamma: (N x K), probabilities of clusters for objects
        Returns:
        alpha: (K), mixture component weights
        beta: (K x C), mixture categories weights
        """
        # Compute alpha
        gamma_df = pd.DataFrame(gamma, index=self._household_ids)
        grouped_gamma_sum = gamma_df.groupby(gamma_df.index).apply(sum)
        alpha = grouped_gamma_sum.values / grouped_gamma_sum.sum(axis=1).values.reshape(-1, 1)
        alpha = alpha.repeat(self._household_freqs, axis=0)
        return alpha



def run_em(X, K_max=20, criterion='icl_bic'):
    
    if criterion not in {'icl_bic', 'bic'}:
        raise Exception('Unknown value for criterion: %s' % criterion)

    X = np.vstack(X)
    np.random.shuffle(X)

    nb_train = int(X.shape[0] * 0.8)
    X_train = X[:nb_train]
    X_test = X[nb_train:]

    likelihoods = []
    bics = []
    icl_bics = []
    best_k = -1
    best_alpha = None
    best_beta = None
    best_gamma = None
    prev_criterion = float('inf')
    
    for k in tqdm(range(2, K_max + 1)):

        model = MultinomialExpectationMaximizer(k, restarts=1)
        _, alpha, beta, gamma = model.fit(X_train)
        log_likelihood = model.compute_log_likelihood(X_test, alpha, beta)
        bic = model.compute_bic(X_test, alpha, beta, log_likelihood)
        icl_bic = model.compute_icl_bic(bic, gamma)
        likelihoods.append(log_likelihood)
        bics.append(bic)
        icl_bics.append(icl_bic)

        criterion_cur_value = icl_bic if criterion == 'icl_bic' else bic
        if criterion_cur_value < prev_criterion:
            prev_criterion = criterion_cur_value
            best_alpha = alpha
            best_beta = beta
            best_gamma = gamma
            best_k = k
    
    print('best K = %i' % best_k)
    print('best_alpha: %s' % str(best_alpha))
    print('best_beta: %s' % str(best_beta))
    
    return likelihoods, bics, icl_bics, best_alpha, best_beta, best_gamma



class clean_text(TransformerMixin):

    def __init__(self, verbose : bool = True):
        self.verbose = verbose
        nltk.download('punkt')

    def fit(self, X, y=None):
        return self    
    
    def transform(self, X):    
        corpus = deepcopy(X)
        cleaned_text = []
        # Preprocess:
        for z, se in enumerate(corpus.tolist()):
            if (z % 1000 == 0) & self.verbose: print('Processing document {}'.format(z))
            tokens = word_tokenize(se)
            # convert to lower case
            tokens = [w.lower() for w in tokens]
            # remove punctuation from each word
            table = str.maketrans('', '', string.punctuation)
            stripped = [w.translate(table) for w in tokens]
            # remove remaining tokens that are not alphabetic
            words = [word for word in stripped if word.isalpha()]    
            # filter out stop words
            #stop_words = set(stopwords.words('english'))
            #words = [w for w in words if not w in stop_words]
            cleaned_text.append(' '.join(words))
        return cleaned_text  
