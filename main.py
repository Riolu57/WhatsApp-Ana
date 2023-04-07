from utility.visualisation import plot_all_dates, plot_all_time, plot_frequency, plot_convo_idx, plot_freq_and_posterior
from content_analysis import analyse_msg, index_conversations
from extraction import convert
from utility.stop_words import STOP_WORDS
from utility.stop_chars import STOP_CHARS
from utility.bca import get_bcp
from tkinter import Tk
from tkinter.filedialog import askopenfilename

Tk().withdraw()
file = askopenfilename()
plot_path = r"./Plots"
convo = convert(file)
plot_all_dates(convo, plot_path, save=True)
author_list = analyse_msg(convo)
plot_all_time(author_list, plot_path, save=True)
with open(f"{plot_path}/stop_words.txt", "w", encoding="UTF-8") as f:
    f.write(
        "\n".join(str(STOP_WORDS)[1:-1].split(","))
    )

with open(f"{plot_path}/stop_chars.txt", "w", encoding="UTF-8") as f:
    f.write(
        "\n".join(str(STOP_CHARS)[1:-1].split(","))
    )

frequency_df, (means, probs) = get_bcp(convo)
plot_frequency(frequency_df, "All", plot_path, save=True)
plot_freq_and_posterior(frequency_df, means, probs, "All", plot_path, save=True)
indexed_convo = index_conversations(frequency_df, probs, change_threshold=0.95, clean=False)
plot_convo_idx(indexed_convo, "All", plot_path, save=True)

