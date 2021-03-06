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



st.session_state["warned_about_save_answers"] = True

stb.set_book_config(menu_title="Main Menu",
                    menu_icon="lightbulb",
                    options=[
                              "About",
                              "PARASITOLOGY",
                              "HEMATOLOGY",
                              "SEROLOGY",
                              "Contact",
                            ],
                           
                    paths=[
                        "apps/about.py",
                        "apps/malaris.py",
                        "apps/hemato.py",
                        "apps/serology.py",
                        "apps/contact.py",
                          ],
                    
                    icons=[
                          "house",
                          "robot",
                          "robot",
                          "trophy"
                          ],
                    save_answers=True,
                    )

# Create an instance of the app 
#app = MultiPage()

#col1.image(display, width = 120)

#col2.title("")

# Add all your application here
#app.add_page("THIN SMEAR", thin_smear_analysis.app)
#app.add_page("THICK SMEAR", thick_smear_analysis.app)

# The main app
#app.run()"""








