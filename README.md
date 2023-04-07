# WhA
Wh(atsApp)A(nalysis) is a software tool written in Python, LaTeX and a little bit of R.
Functionality includes:
  - Anlysis of conversation patterns based on change in message frequency
  - Sentiment analysis of partners
  - Most/Least common texting times
  - Calendar overview of texting frequency over days
  - Automatically generated plots
  - TeX file generated containing variables
  
To run the programm, you need to do three things:
  1. `pip install -r requirements.txt`
  2. Run `main.py`: it will open a file selection window. Select the WhatsApp chat you want to analyse there.
  3. Compile the `template.tex` file into a pdf using any TeX installation. For Unicode support, something like LuaTeX is recommended.
  
