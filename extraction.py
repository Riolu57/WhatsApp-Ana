import re
import pandas as pd
from datetime import datetime
from nltk.sentiment import SentimentIntensityAnalyzer


def replace_short(content: str, short, full):
    if short in content.lower():
        new_content = content.replace(short, f" {full}")

        return new_content

    return content


def replace_all_shorts(content):
    all_abbr = [("n't", "not"), ("'d", "would"), ("'ll", "will"), ("'m", "am"), ("'re", "are"), ("'s", "is"),
                ("'ve", "have")]
    new_content = content

    for short, full in all_abbr:
        new_content = replace_short(new_content, short, full)

    return new_content


def clean_msg(content):
    cur_content = replace_all_shorts(content)

    for char in ["'"]:
        cur_content = cur_content.replace(char, "")

    return cur_content


def convert(filepath):
    current_msg = None
    msgs = []
    # The regEx to recognize the beginning of a message
    regStr = r"^\d\d.\d\d.\d\d, \d\d:\d\d.+?:."

    # Track msg idx to skip first msg more easily
    idx = 0

    # Get sentiment analyzer
    sia = SentimentIntensityAnalyzer()

    with open(filepath, encoding="utf-8") as f:
        while line := f.readline():
            if not idx:
                # The first line is always "This chat is encoded [...]" bla bla, so we skip it
                idx += 1
                continue

            # Check if we found a beginning, if so, save the beginning (date, time, author) aka. we found a message
            if re.findall(regStr, line) != list():
                if current_msg is not None:
                    current = clean_msg(current)
                    current_msg.append(current[:-1].lower())
                    current_msg.append(sia.polarity_scores(current))
                    msgs.append(current_msg)

                current_msg = []
                current = line

                # Split the regEx returned, at the beginning of the message, at each space
                data = re.findall(regStr, current)[0].split(" ")

                # The message content is just the whole message without the regEx, so we subsitute the regEx with the
                # empty string
                current = re.sub(regStr, "", current)

                media = False
                if "<Medien ausgeschlossen>" in current:
                    media = True
                    current = ""

                # Get datetime
                date_and_time = datetime.strptime(data[0] + " " + data[1], "%d.%m.%y, %H:%M")

                # Clean the author names of any unwanted characters
                author = ""
                for char in " ".join(data[3:-1]):
                    if char.isalnum() or char == " ":
                        author += char

                current_msg.extend([author, date_and_time, media])

            # Otherwise it's a continuation of a message so its appended
            else:
                current += line

        else:
            if current_msg is not None:
                current = clean_msg(current)
                current_msg.append(current[:-1].lower())
                current_msg.append(sia.polarity_scores(current))
                msgs.append(current_msg)

    convo = pd.DataFrame(data=msgs, columns=["author", "datetime",  "media", "content", "sent_score"])
    return convo


if __name__ == "__main__":
    convo = convert(r".\Data\test.txt")
    print(convo)
