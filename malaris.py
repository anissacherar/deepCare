import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import streamlit_authenticator as stauth
#multipages
from multipage import MultiPage
from pages import thin_smear_analysis, thick_smear_analysis
import streamlit_book as stb
from pathlib import Path


#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
st.set_page_config(
     page_title="MALARIS",
     page_icon="🧊",
     layout="wide",
     initial_sidebar_state="expanded",
)

current_path = Path(__file__).parent.absolute()
stb.set_book_config(menu_title="Main Menu",
                    menu_icon="",
                    options=[
                            "Home",
                              "About",
                              "MALARIS",
                              "Contact",

                            ],
                    paths=[
                        current_path / "malaris",
                         current_path / "pages/about",
                        current_path / "pages/thin_smear_analysis",
                        current_path / "pages/Contact",
                          ],
                    icons=[
                          "house",
                          "",
                          "",
                          "",
                          "",
                          "trophy"
                          ],
                    save_answers=True,
                    )

# Create an instance of the app 
app = MultiPage()
col1, col2 = st.columns(2)

# Title of the main page
display = Image.open('logo.JPG')
display = np.array(display)
st.image(display, width = 120)
st.title("DeepCare")

#col1.image(display, width = 120)

#col2.title("")

# Add all your application here
app.add_page("THIN SMEAR", thin_smear_analysis.app)
app.add_page("THICK SMEAR", thick_smear_analysis.app)

# The main app
app.run()








