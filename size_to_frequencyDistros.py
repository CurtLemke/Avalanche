"""This script will run the avalanche model and save the size-to-frequency as a JSON 
file where the key is the cascade size and value is the number of occurences."""

import json
from tqdm import tqdm
import numpy as np
import cvxpy as cvx
import time
from contagion import binarize_probabilities, distribute_liabilities, make_connections, DeterministicRatioNetwork, TestNetwork

# adjust numberOfRuns to change number of times entire model is run
# default = 1
numberOfRuns = 100
# adjust steps to change the number of steps per run of the model
# default = 1000000
steps = 1000000

# change distribution:
# options:
# beta       - Change distribution variable to 'beta'
# chi square - Change distribution variable to 'chisquare'
# f          - Change distribution variable to 'f'
# gamma      - Change distribution variable to 'gamma'
# lognormal  - Change distribution variable to 'lognormal'
# normal     - Change distribution variable to 'normal'
# poisson    - Change distribution variable to 'poisson'


#### Distribution variables:
# used for all
cashDistribution = 'poisson'
leverageDistribution = 'poisson'
size = 100;  # ## DO NOT CHANGE!

# # beta ##
  # cash #
betaCashAlpha = 3;  # default = 3
betaCashBeta = 8;  # default = 8
  # leverage #
betaLeverageAlpha = 3;  # default = 3
betaLeverageBeta = 8;  # default = 8

# # chi square ##
  # cash #
chiCashDf = 10;  # default = 10
  # leverage #
chiLeverageDf = 10;  # default = 10

# # f ##
  # cash #
fCashDfnum = 12;  # default = 12
fCashDfden = 12;  # default = 12
  # leverage #
fLeverageDfnum = 12;  # default = 12
fLeverageDfden = 12;  # default = 12

# # gamma ##
  # cash #
gammaCashShape = 2;  # default = 2
gammaCashScale = 1;  # default = 1
  # leverage #
gammaLeverageShape = 2;  # default = 2
gammaLeverageScale = 1;  # default = 1

# # lognormal ##
  # cash #
lognormalCashMean = 0;  # default = 0
lognormalCashSigma = 1;  # default = 1
  # leverage #
lognormalLeverageMean = 0;  # default = 0
lognormalLeverageSigma = 1;  # default = 1

# # normal ##
  # cash #
normalCashLocation = 10000;  # default = 10000
normalCashScale = 10000;  # default = 10000
  # leverage #
normalLeverageLocation = 10;  # default = 10
normalLeverageScale = 2;  # default = 2

# # poisson ##
  # cash #
poissonCashLambda = 3;  # default = 3
  # leverage #
poissonLeverageLambda = 3;  # default = 3

# the below function is called to run the model
def runModel(cashDistribution, leverageDistribution):
    cash_vector = generateCashVector(cashDistribution)
    cash_vector[cash_vector <= 0] = 1 * 10 ** -10
    # cash_vector[cash_vector > 5000] = 6500
    cash_to_connectivity = lambda x: safe_ln(x)
    connectivity_vector = cash_to_connectivity(cash_vector)

    # Make the adjacency matrix
    mat = make_connections(connectivity_vector)
    mat = binarize_probabilities(mat)

    # Distribute liabilities
    leverage_ratios = generateLeverageRatios(leverageDistribution)
    leverage_ratios[leverage_ratios < 5] = 5

    liabilities = np.multiply(cash_vector, leverage_ratios)
    mat = distribute_liabilities(mat, liabilities)
    for i, cash in enumerate(cash_vector):
        mat[i, i] = cash

    defaults_to_freq = {}
    
    for z in tqdm(range(steps)):
        model = TestNetwork(size, mat)
        model.reset_net()

        step_result = model.step()
        defaults = step_result['cascade_defaults'] + step_result['ratio_defaults']
        if defaults in defaults_to_freq:
            defaults_to_freq[defaults] += 1
        else:
            defaults_to_freq[defaults] = 1

    """ the below code saves the results and distribution configuration
    in a .json file with the date and time added to
    the name so as to prevent overwriting."""

    with open('result_' + cashString + leverageString + str(time.strftime("%d_%m_%y_%H%M%S")) + '.json', 'w') as fp:
        json.dump(defaults_to_freq, fp)

# The below function generates the chosen cash distribution
def generateCashVector(distribution):
    if distribution == 'beta':
        cash_vector = np.random.beta(betaCashAlpha, betaCashBeta, size)
    elif distribution == 'chisquare':
        cash_vector = np.random.chisquare(chiCashDf, size)
    elif distribution == 'f':
        cash_vector = np.random.gamma(fCashDfnum, fCashDfden, size)
    elif distribution == 'gamma':
        cash_vector = np.random.gamma(gammaCashShape, gammaCashScale, size)
    elif distribution == 'lognormal':
        cash_vector = np.random.lognormal(lognormalCashMean, lognormalCashSigma, size)
    elif distribution == 'normal':
        cash_vector = np.random.normal(normalCashLocation, normalCashScale, size)
    elif distribution == 'poisson':
        cash_vector = np.random.poisson(poissonCashLambda, size)
    else: cash_vector = np.random.normal(normalCashLocation, normalCashScale, size)
    return cash_vector

# The below function generates a string to reflect the chosen cash distribution
def generateCashString(distribution):
    cashString = 'aoeu'
    if distribution == 'beta':
        cashString = 'BetaCash_Alpha' + str(betaCashAlpha) + 'Beta' + str(betaCashBeta) + '_'
    elif distribution == 'chisquare':
        cashString = 'ChisquareCash_Df' + str(chiCashDf) + '_'
    elif distribution == 'f':
        cashString = 'fCash_Dfnum' + str(fCashDfnum) + 'Dfden' + str(fCashDfden) + '_'
    elif distribution == 'gamma':
        cashString = 'GammaCash_Shape' + str(gammaCashShape) + 'Scale' + str(gammaCashScale) + '_'
    elif distribution == 'normal':
        cashString = 'NormalCash_location' + str(normalCashLocation) + 'Scale' + str(normalCashScale) + '_'
    elif distribution == 'poisson':
        cashString = 'PoissonCash_Lambda' + str(poissonCashLambda) + '_'
    elif distribution == 'lognormal':
        cashString = 'LognormalCash_Mean' + str(lognormalCashMean) + 'Sigma' + str(lognormalCashSigma) + '_'
    else: cashString = 'NormalCash_location' + str(normalCashLocation) + 'Scale' + str(normalCashScale) + '_'
    return cashString

# The below function generates the chosen leverage distribution
def generateLeverageRatios(distribution):
    if distribution == 'beta':
        leverage_ratios = np.random.beta(betaLeverageAlpha, betaLeverageBeta, size)
    elif distribution == 'chisquare':
        leverage_ratios = np.random.chisquare(chiLeverageDf, size)
    elif distribution == 'f':
        leverage_ratios = np.random.f(fLeverageDfnum, fLeverageDfden, size)
    elif distribution == 'gamma':
        leverage_ratios = np.random.gamma(gammaLeverageShape, gammaLeverageScale, size)
    elif distribution == 'lognormal':
        leverage_ratios = np.random.lognormal(lognormalLeverageMean, lognormalLeverageSigma, size)
    elif distribution == 'normal':
        leverage_ratios = np.random.normal(normalLeverageLocation, normalLeverageScale, size)
    elif distribution == 'poisson':
        leverage_ratios = np.random.poisson(poissonLeverageLambda, size)
    else: leverage_ratios = np.random.normal(normalLeverageLocation, normalLeverageScale, size)
    return leverage_ratios

# The below function generates a string to reflect the chosen leverage distribution
def generateLeverageString(distribution):
    leverageString = 'aoeu'
    if distribution == 'beta':
        leverageString = 'BetaLeverage_Alpha' + str(betaLeverageAlpha) + 'Beta' + str(betaLeverageBeta) + '_'
    elif distribution == 'chisquare':
        leverageString = 'ChisquareLeverage_Df' + str(chiLeverageDf) + '_'
    elif distribution == 'f':
        leverageString = 'fLeverage_Dfnum' + str(fLeverageDfnum) + 'Dfden' + str(fLeverageDfden) + '_'
    elif distribution == 'gamma':
        leverageString = 'GammaLeverage_Shape' + str(gammaLeverageShape) + 'Scale' + str(gammaLeverageScale) + '_'
    elif distribution == 'lognormal':
        leverageString = 'LognormalLeverage_Mean' + str(lognormalLeverageMean) + 'Sigma' + str(lognormalLeverageSigma) + '_'
    elif distribution == 'normal':
        leverageString = 'NormalLeverage_location' + str(normalLeverageLocation) + 'Scale' + str(normalLeverageScale) + '_'
    elif distribution == 'poisson':
        leverageString = 'PoissonLeverage_Lambda' + str(poissonLeverageLambda) + '_'
    else: leverageString = 'NormalLeverage_location' + str(normalLeverageLocation) + 'Scale' + str(normalLeverageScale) + '_'
    return leverageString

# The below function checks for and corrects potential division by zero
def safe_ln(x, minval=0.000000000001):  # Value chosen simply as it is relatively close to zero
    return np.log(x.clip(min=minval)).astype(int)


''' Create strings to use in saving results so there is record of distributions
    used and their parameters: '''
cashString = generateCashString(cashDistribution)
leverageString = generateLeverageString(leverageDistribution)

# the below loop runs the program for the desired number of iterations
for j in range(numberOfRuns):
    runModel(cashDistribution, leverageDistribution)
