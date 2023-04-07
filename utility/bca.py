import numpy as np
from rpy2.robjects.packages import importr
import rpy2.robjects as robjects
import pandas as pd
from rpy2.robjects import numpy2ri
from rpy2.robjects import default_converter


def prep_data(datetimes, rounded=False):
    """Prepares the data for bayesian change point analysis. Preparation consists of determining how many messages were

    sent in each 5 min interval from start to end of the conversation.
    :param datetimes: A pandas Series object of datetimes of all messages of the conversation.
    :param rounded: Indicates whether the datetimes in the dataframe have been rounded or not.
    :return: A dataframe with a "freq" column indicating how many messages were sent during those 5 min.
    """
    if rounded:
        new_datetimes = datetimes
    else:
        new_datetimes = datetimes.map(lambda x: x.replace(minute=x.minute - x.minute % 5))

    start = min(new_datetimes)
    end = max(new_datetimes)

    df = pd.DataFrame({'datetime': pd.date_range(start=start, end=end, freq='5min', inclusive='both')})
    df["freq"] = df['datetime'].map(lambda x: len(new_datetimes[new_datetimes == x]) if x in new_datetimes.values else 0)

    return df


def prep_total(convo):
    return prep_data(convo["datetime"])


def prep_individual(convo):
    for author in set(convo["author"]):
        yield prep_data(convo[convo['author'] == author]["datetime"])


def bcp(data):
    r = robjects.r #allows access to r object with r.
    bcp = importr('bcp') #import bayesian change point package in python

    np_cv_rules = default_converter + numpy2ri.converter

    with np_cv_rules:
        values = bcp.bcp(data['freq'].values) #use bcp function on vector

    posterior_means = values['posterior.mean'].flatten()
    posterior_probability = values['posterior.prob']

    return posterior_means, posterior_probability


def get_bcp(convo):
    data = prep_total(convo)
    return data, bcp(data)


if __name__ == "__main__":
    # from extraction import convert
    # from visualisation import plot_freq_and_posterior

    # convo = convert(filename)
    # for res in  prep_individual(convo):
    #     print(res)
    # dataframe, (means, probs) = get_bcp(convo)
    # plot_freq_and_posterior(dataframe, means, probs, "All", "")
    pass
