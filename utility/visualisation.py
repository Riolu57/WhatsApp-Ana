import calplot
import pandas as pd
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt


def plot_sentiment(dataframe, msg_author, path, save=True):
    """
    Plots all sentiments over time.
    :param dataframe: A dataframe with sentiment scores.
    :param msg_author: The name of the author of these messages.
    :param path: The path where to save the image.
    :param save: Whether to save the image.
    :return: None.
    """
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex=True, sharey=False)
    fig.suptitle(f'Sentiment over Time of {msg_author}')

    ax1.plot(dataframe['datetime'].values, dataframe["sent_score"].apply(lambda x: x["compound"]), c='k')
    ax1.set_title(f"Compound Score")
    plt.xlabel("Dates")
    fig.supylabel("Sentiment Score")

    ax2.plot(dataframe['datetime'].values, dataframe["sent_score"].apply(lambda x: x["pos"]), c='g')
    ax2.set_title(f"Positive Score")

    ax3.plot(dataframe['datetime'].values, dataframe["sent_score"].apply(lambda x: x["neu"]), c='b')
    ax3.set_title(f"Neutral Score")

    ax4.plot(dataframe['datetime'].values, dataframe["sent_score"].apply(lambda x: x["neg"]), c='r')
    ax4.set_title(f"Negative Score")

    # Tell matplotlib to interpret the x-axis values as dates
    ax1.xaxis_date()

    # Make space for and rotate the x-axis tick labels
    fig.autofmt_xdate()

    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)


    if save:
        plt.savefig(rf"{path}\{msg_author}_idx_convo.pdf", bbox_inches='tight', transparent=True)
    else:
        plt.show()

    plt.clf()


def plot_convo_idx(dataframe, msg_author, path, save=True):
    """
    Plots all conversation time intervals (without 0 freq points) that have more than 10 messages.
    Each neighboring conversation is different in colour.


    :param dataframe: A dataframe with indexed conversations and frequency.
    :param msg_author: The name of the author of these messages.
    :param path: The path where to save the image.
    :param save: Whether to save the image.
    :return: None.
    """
    fig, ax = plt.subplots()

    cleaned_convo = dataframe[dataframe["freq"] > 0]

    corrected_convos_idx = cleaned_convo.groupby("convo_idx").sum()[
        cleaned_convo.groupby("convo_idx").sum()["freq"] > 10]
    all_convo = cleaned_convo[cleaned_convo["convo_idx"].isin(corrected_convos_idx.index)]

    plt.scatter(all_convo['datetime'].values, all_convo['freq'].values, alpha=0.2,
                c=[['r', 'g', 'b'][int(i)] for i in all_convo["convo_idx"].map(lambda x: x % 3).values])
    plt.title(f"Different Conversations of {msg_author}")
    plt.ylabel("# Texts")
    plt.xlabel("Dates")

    # Tell matplotlib to interpret the x-axis values as dates
    ax.xaxis_date()

    # Make space for and rotate the x-axis tick labels
    fig.autofmt_xdate()


    if save:
        plt.savefig(rf"{path}\{msg_author}_idx_convo.pdf", bbox_inches='tight', transparent=True)
    else:
        plt.show()

    plt.clf()


def plot_freq_and_posterior(dataframe, means, probs, msg_author, path, save=True):
    """
    Plots the message frequency over times, as well as the probability of change in frequency and the mean frequency.

    :param dataframe: A dataframe containing the msg frequency of all 5 min time intervals from start to finish of the
        entire conversation.
    :param means: The mean frequency at each point
    :param probs: The probability of change at each point
    :param msg_author: The author of all messages
    :param path: The path where to save the plot
    :param save: Whether to save the plot.
    :return: None.
    """
    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)

    ax1.plot(dataframe['datetime'].values, means, c='tab:orange')
    ax1.scatter(dataframe['datetime'].values, dataframe['freq'].values, alpha=0.2, c='k')
    ax1.set_title(f"Texting frequency of {msg_author}")
    ax2.set_xlabel("Dates")
    ax1.set_ylabel("# Texts")

    ax2.plot(dataframe['datetime'].values, probs, c='tab:cyan')
    ax2.set_title(f"Probability of Change in Texting Frequency")
    ax2.set_xlabel("Dates")
    ax2.set_ylabel("Probability of Change")

    # Tell matplotlib to interpret the x-axis values as dates
    ax2.xaxis_date()

    # Make space for and rotate the x-axis tick labels
    fig.autofmt_xdate()

    if save:
        plt.savefig(rf"{path}\{msg_author}_frequency_posterior.pdf", bbox_inches='tight', transparent=True)
    else:
        plt.show()

    plt.clf()


def plot_frequency(dataframe, msg_author, path, save=True):
    """
    Plots the message frequency over time.

    :param dataframe: A dataframe containing the msg frequency of all 5 min time intervals from start to finish of the
    entire conversation.
    :param msg_author: The author of all messages
    :param path: The path where to save the plot
    :param save: Whether to save the plot.
    :return: None.
    """
    fig, ax = plt.subplots(figsize=((10*len(dataframe)//(365*24*60*12)) + 10, 3))
    plt.scatter(dataframe['datetime'].values, dataframe['freq'].values, alpha=0.2)
    plt.title(f"Texting frequency of {msg_author}")
    plt.xlabel("# Texts")
    plt.ylabel("Dates")

    # Tell matplotlib to interpret the x-axis values as dates
    ax.xaxis_date()

    # Make space for and rotate the x-axis tick labels
    fig.autofmt_xdate()


    if save:
        plt.savefig(rf"{path}\{msg_author}_frequency.pdf", bbox_inches='tight', transparent=True)

    plt.clf()


def plot_all_dates(convo, path, save=True):
    """
    Plots the message activities per day of each author of the conversation.

    :param convo: The entire conversation, only datetime and author are needed.
    :param path: The path where to save the plots.
    :param save: Whether to save the plots.
    :return: None.
    """
    authors = set(convo["author"])
    for author in authors:
        cur_author_times = convo[convo['author'] == author]['datetime']
        plot_dates(cur_author_times, author, path, save=save)
    plot_dates(convo['datetime'], "All Authors", path, save=save)



def plot_dates(dates, msg_author, path, save=True):
    """
    Plots the daily message activity of a single author.

    :param dates: A pandas Series containing the datetimes of all messages.
    :param msg_author: The author of the messages whose's datetimes were passed earlier.
    :param path: The path where to save the plot.
    :param save: Whether to save the image.
    :return: None.
    """
    date_count = Counter(dates)
    num_years = len(set(map(lambda x: x.year, date_count.keys())))
    msgs = pd.Series(date_count.values(), index=date_count.keys())
    plt, ax = calplot.calplot(msgs, cmap='YlGn', figsize=(12, num_years*2), suptitle=msg_author)

    if not save:
        plt.show(transparent=True)
    else:
        plt.savefig(rf"{path}\{msg_author}_dates.pdf", bbox_inches='tight', transparent=True)

    plt.clf()


def plot_all_time(author_list, path, save=True):
    """
    Plots the message activities over time of each author of the conversation.

    :param author_list: A list of Author objects.
    :param path: The path where to save the plots.
    :param save: Whether to save the plots.
    :return: None.
    """
    all_authors = Counter()
    for author in author_list:
        plot_time(author.time_freq_count, author.name, path, save=save)
        all_authors.update(author.time_freq_count)
    plot_time(all_authors, "All Authors", path, save=save)


def plot_time(time_frequency, msg_author, path, save=True):
    """
    Plots the typical texting time of an author.

    :param time_frequency: A counter object of an author.
    :param msg_author: The author of the messages whose's time_frequency was passed.
    :param path: The path where to save the plots.
    :param save: Whether to save the plots.
    :return: None.
    """
    if isinstance(time_frequency, Counter):
        times = time_frequency.most_common()
    elif isinstance(time_frequency, list):
        times = time_frequency

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

    xData = [(time[0][0] + (time[0][1] / 60)) * (np.pi / 12) for time in times]
    plt.scatter(xData, [time[1] for time in times], c=[time[1] for time in times])

    # Set the circumference labels
    ax.set_xticks(np.linspace(0, 2 * np.pi, 24, endpoint=False))
    ax.set_xticklabels(range(24))

    # Make the labels go clockwise
    ax.set_theta_direction(-1)

    # Place 0 at the top
    ax.set_theta_offset(np.pi / 2.0)

    plt.title(f"{msg_author}'s Message Distribution")

    if not save:
        plt.show(transparent=True)
    else:
        plt.savefig(rf"{path}\{msg_author}_times.pdf", bbox_inches='tight', transparent=True)

    plt.clf(); plt.cla(); plt.close()


if __name__ == "__main__":
    # import extraction as ex
    # import content_analysis as cont
    # convo = ex.convert(filename)
    # plot_all_dates(convo)
    # plot_time(cont.analyse_msg(convo)[0].time_freq_count.most_common(), cont.analyse_msg(convo)[0].name, save=False)
    # plot_sentiment(convo, "All", "", save=False)
    # indexed_convo = cont.index_conversations(convo, change_threshold=0.95)
    # plot_convo_idx(indexed_convo, "All", "", save=False)
    pass
