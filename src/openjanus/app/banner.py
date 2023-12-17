from openjanus.utils.text_coloring import GREEN_TEXT, RESET_TEXT


def banner() -> str:
    return GREEN_TEXT + r"""
   ___                       _                       
  / _ \ _ __   ___ _ __     | | __ _ _ __  _   _ ___ 
 | | | | '_ \ / _ \ '_ \ _  | |/ _` | '_ \| | | / __|
 | |_| | |_) |  __/ | | | |_| | (_| | | | | |_| \__ \
  \___/| .__/ \___|_| |_|\___/ \__,_|_| |_|\__,_|___/
       |_|                                                                           
""" + RESET_TEXT + r"""
              (Brought to you by Und3rf10w)
"""