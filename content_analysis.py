from collections import Counter
from functools import reduce
import matplotlib.pyplot as plt
import numpy as np
from utility.author import Author
from utility.stop_words import STOP_WORDS
from utility.stop_chars import STOP_CHARS
from utility.bca import get_bcp
from datetime import timedelta
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import operator
from functools import reduce


def analyse_convos(convo, convo_times):
    """Work in progress. Supposed to find most negative and most positive message in each conversation.

    :param convo:
    :param convo_times:
    :return:
    """
    def get_max_neg(df):
        return max(df["sent_score"], key=lambda x: x["neg"])

    def get_max_pos(df):
        max(df["sent_score"], key=lambda x: x["pos"])

    for (start, end) in convo_times:
        time_range = pd.date_range(start=start, end=end, freq='1min')
        cur_convo = convo[convo["datetime"].isin(time_range)]

        # vectorizer = TfidfVectorizer(input="content", encoding="UTF-8", analyzer="word", )
        # tf_idf_scores = vectorizer.fit_transform(cur_convo['content'])

        viable_msg = cur_convo[cur_convo["content"].map(lambda x: len(x.split(" "))) > 2]
        most_pos = viable_msg[viable_msg["sent_score"] == get_max_pos(viable_msg)]
        most_neg = viable_msg[viable_msg["sent_score"] == get_max_neg(viable_msg)]

    return


def find_convo_times(dataframe, msg_threshold=10):
    """Computes start and end times of all conversations.

    :param dataframe: The conversation as dataframe (with rounded times, summed msg count during that time and
        conversation index)
    :param msg_threshold: A threshold above which a conversation will be considered interesting.
    :return: A list of tuples containing the start and end times of interesting conversations.
    """
    corrected_convos = dataframe.groupby("convo_idx").sum()[
        dataframe.groupby("convo_idx").sum()["freq"] > msg_threshold]

    convo_times = []

    for idx in corrected_convos.index:
        cur_convo = dataframe[dataframe["convo_idx"] == idx]
        cur_convo_times = cur_convo["datetime"]
        start = min(cur_convo_times)
        end = max(cur_convo_times) + timedelta(minutes=4)
        convo_times.append((start, end))

    return convo_times


def index_conversations(dataframe, probs, change_threshold=0.95, clean=False):
    """Add a conversation index to all messages in the conversation based on BCP.

    :param dataframe: A dataframe as the result of the data preperation of the BCP.
    :param probs: The probabilities of the frequency in a conversation changing.
    :param change_threshold: The threshold above which a convo is said to have started.
    :param clean: Whether to remove freq=0 rows from the dataframe.
    :return: The conversation with the additional column "convo_idx", which states the index of the conversation a msg
    belongs to.
    """
    convo_idx = 0
    cut_off_probs = (probs > change_threshold) * 1
    conv_indices = np.zeros(probs.shape)

    for idx, val in enumerate(cut_off_probs):
        convo_idx += val
        conv_indices[idx] = convo_idx

    dataframe["convo_idx"] = conv_indices

    if clean:
        return dataframe[dataframe["freq"] > 0]
    else:
        return dataframe


def time_round(times: list):
    """Rounds the times of a list of hour, min pairs down to the closest 5 min and counts their occurrences.

    :param times: A list of (hour, min) tuples.
    :return: A Counter of the rounded tuples in the list.
    """
    def round_down_to_five(number):
        return number - (number % 5)

    rounded_counter = Counter()

    for hour, minute in times:
        rounded_counter.update([(hour, round_down_to_five(minute))])

    return rounded_counter


def clean_latex_symbols(counter: Counter):
    """Removes symbols latex is sensitive to in a Counter's keys and replaces them with \string{symbol}

    :param counter: A counter containing potentially sensitive symbols in its keys.
    :return: A new counter with "cleaned" keys.
    """
    latex_list = ["^", "_"]
    to_filter_keys = list(filter(lambda x: any([char in x for char in latex_list]), counter.keys()))
    for old_key in to_filter_keys:
        new_key = old_key
        for char in latex_list:
            new_key = new_key.replace(char, f"\string{char}")
        c = counter[old_key]
        del counter[old_key]
        counter[new_key] = c

    return counter


def analyse_msg(convo):
    """Analyse messages and write results to file.

    :param convo: A pandas dataframe consisting of messages.
    :return: List of Author objects containing frequencies.
    """
    results = list()
    authors = set(convo["author"])

    # Do the analysis for each author separately
    for author_idx, author in enumerate(authors):
        cur_author_frame = convo[convo['author'] == author]
        cur_author_messages = cur_author_frame['content'].to_list()

        # Get freq dist of:
        # Message length
        len_count = Counter(map(len, cur_author_messages))

        # Words
        word_freq_count = cur_author_frame["content"].apply(lambda x: x.split())
        word_freq_count = Counter(reduce(operator.add, word_freq_count.to_list()))


        # Characters
        char_freq_count = Counter()
        for msg_char_counter in map(Counter, cur_author_messages):
            char_freq_count.update(msg_char_counter)

        cur_author_datetimes = cur_author_frame['datetime'].to_list()
        # Time
        time_freq_count = time_round(list(map(lambda x: (x.hour, x.minute), cur_author_datetimes)))
        # Date
        date_freq_count = Counter(map(lambda x: (x.year, x.month, x.day), cur_author_datetimes))

        results.append(
            Author(
                name=author,
                len_count=len_count,
                word_freq_count=word_freq_count,
                time_freq_count=time_freq_count,
                date_freq_count=date_freq_count
            )
        )

        # Find most/least/mean/media statistics of things
        sorted_freq = sorted(date_freq_count.most_common(), key=lambda x: x[1], reverse=True)
        median_freq = sorted_freq[len(sorted_freq)//2]

        most_common_date = date_freq_count.most_common(1)[0]
        least_common_date = list(date_freq_count.most_common())[-1]

        most_common_time = time_freq_count.most_common(1)[0]
        lest_common_time = list(time_freq_count.most_common())[-1]

        most_common_words = word_freq_count.most_common(5)
        most_common_words_corrected = Counter(
            {k: c for k, c in word_freq_count.items() if k not in STOP_WORDS and k not in " "})
        most_common_words_corrected = clean_latex_symbols(most_common_words_corrected)
        most_common_words_corrected = most_common_words_corrected.most_common(5)

        most_common_chars = char_freq_count.most_common(5)
        most_common_chars_corrected = Counter(
            {k: c for k, c in char_freq_count.items() if k not in STOP_CHARS and k not in " "})
        most_common_chars_corrected = clean_latex_symbols(most_common_chars_corrected)
        most_common_chars_corrected = most_common_chars_corrected.most_common(5)



        with open(r".\TmpData\variables.tex", "w", encoding="UTF-8") as file:
            file.write(rf"""
\newcommand\Name{{{author.split(" ")[0]}}}
\newcommand\TotalMsgCount{{{len(cur_author_frame)}}}
\newcommand\AvgMsgCount{{{round(len(cur_author_frame) / ((cur_author_frame.iloc[-1]["datetime"] - cur_author_frame.iloc[0]["datetime"]).total_seconds() / (60 * 60 * 24)), 2)}}}
\newcommand\MedianMsgCount{{{median_freq[1]}}}

\newcommand\TotalMsgCountOverall{{{len(convo)}}}
\newcommand\TotalMsgCountRatio{{{round((len(cur_author_frame)/len(convo))*100, 2)}}}

\newcommand\MaxDayMsgCount{{{most_common_date[1]}}}
\newcommand\MaxDayMsgCountDay{{{most_common_date[0][2]}}}
\newcommand\MaxDayMsgCountMonth{{{most_common_date[0][1]}}}
\newcommand\MaxDayMsgCountYear{{{most_common_date[0][0]}}}

\newcommand\MinDayMsgCount{{{least_common_date[1]}}}
\newcommand\MinDayMsgCountDay{{{least_common_date[0][2]}}}
\newcommand\MinDayMsgCountMonth{{{least_common_date[0][1]}}}
\newcommand\MinDayMsgCountYear{{{least_common_date[0][0]}}}

\newcommand\DatePlotName{{{author}_dates.pdf}}
\newcommand\TimePlotName{{{author}_times.pdf}}
\newcommand\FrequencyPlotName{{All_frequency.pdf}}
\newcommand\PosteriorPlotName{{All_frequency_posterior.pdf}}
\newcommand\IdxConvoPlotName{{All_idx_convo.pdf}}


\newcommand\MostMsgTimeHour{{{ most_common_time[0][0] }}}
\newcommand\MostMsgTimeMin{{{ most_common_time[0][1] }}}
\newcommand\MostMsgTime{{{ most_common_time[1] }}}

\newcommand\LeastMsgTimeHour{{{ lest_common_time[0][0] }}}
\newcommand\LeastMsgTimeMin{{{ lest_common_time[0][1] }}}
\newcommand\LeastMsgTime{{{ lest_common_time[1] }}}

\newcommand\MostCommonWordUncorrectedOne{{{ most_common_words[0][0] }}}
\newcommand\MostCommonWordUncorrectedTwo{{{ most_common_words[1][0] }}}
\newcommand\MostCommonWordUncorrectedThree{{{ most_common_words[2][0] }}}
\newcommand\MostCommonWordUncorrectedFour{{{ most_common_words[3][0] }}}
\newcommand\MostCommonWordUncorrectedFive{{{ most_common_words[4][0] }}}

\newcommand\MostCommonWordCorrectedOne{{{ most_common_words_corrected[0][0] }}}
\newcommand\MostCommonWordCorrectedTwo{{{ most_common_words_corrected[1][0] }}}
\newcommand\MostCommonWordCorrectedThree{{{ most_common_words_corrected[2][0] }}}
\newcommand\MostCommonWordCorrectedFour{{{ most_common_words_corrected[3][0] }}}
\newcommand\MostCommonWordCorrectedFive{{{ most_common_words_corrected[4][0] }}}

\newcommand\MostCommonCharUncorrectedOne{{{ most_common_chars[0][0] }}}
\newcommand\MostCommonCharUncorrectedTwo{{{ most_common_chars[1][0] }}}
\newcommand\MostCommonCharUncorrectedThree{{{ most_common_chars[2][0] }}}
\newcommand\MostCommonCharUncorrectedFour{{{ most_common_chars[3][0] }}}
\newcommand\MostCommonCharUncorrectedFive{{{ most_common_chars[4][0] }}}

\newcommand\MostCommonCharCorrectedOne{{{ most_common_chars_corrected[0][0] }}}
\newcommand\MostCommonCharCorrectedTwo{{{ most_common_chars_corrected[1][0] }}}
\newcommand\MostCommonCharCorrectedThree{{{ most_common_chars_corrected[2][0] }}}
\newcommand\MostCommonCharCorrectedFour{{{ most_common_chars_corrected[3][0] }}}
\newcommand\MostCommonCharCorrectedFive{{{ most_common_chars_corrected[4][0] }}}
        """)

    return results


if __name__ == "__main__":
    # import extraction as ex
    # convo = ex.convert(filepath)
    # conversation_count(convo)
    # analyse_msg(convo)
    # indexed_convo = index_conversations(convo, change_threshold=0.95)
    # convo_times = find_convo_times(indexed_convo)
    # analyse_convos(convo, convo_times)
    pass
