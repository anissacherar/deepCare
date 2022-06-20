import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import streamlit_authenticator as stauth
#multipages
import streamlit_book as stb
from pathlib import Path

st.set_page_config(
     page_title="MALARIS",
     page_icon="🧊",
     layout="wide",
     initial_sidebar_state="expanded",
)

hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
#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

st.session_state["warned_about_save_answers"] = True




current_path = Path(__file__).parent.absolute()
stb.set_book_config(menu_title="Main Menu",
                    menu_icon="",
                    options=[
                              "About",
                              "MALARIS",
                              "Contact",

                            ],
                    paths=[
                         current_path / "apps/about",
                        current_path / "apps/malaris.py",
                        current_path / "apps/Contact",
                          ],
                    icons=[
                          "house",
                          "",
                          "",
                          "trophy"
                          ],
                    save_answers=True,
                    )

# Create an instance of the app 
#app = MultiPage()
col1, col2 = st.columns(2)

# Title of the main page
display = Image.open('logo.JPG')
display = np.array(display)
st.image(display, width = 120)
st.title("DeepCare")

#col1.image(display, width = 120)

#col2.title("")

# Add all your application here
"""app.add_page("THIN SMEAR", thin_smear_analysis.app)
app.add_page("THICK SMEAR", thick_smear_analysis.app)

# The main app
app.run()"""








