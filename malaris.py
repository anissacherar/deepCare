import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import streamlit_authenticator as stauth
import yaml

with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials']['names'],
    config['credentials']['usernames'],
    config['credentials']['passwords'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)
#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
st.set_page_config(
     page_title="MALARIS",
     page_icon="🧊",
     layout="wide",
     initial_sidebar_state="expanded",
)

#multipages
from multipage import MultiPage
from pages import thin_smear_analysis, thick_smear_analysis

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


