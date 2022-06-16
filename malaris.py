import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import streamlit_authenticator as stauth
#multipages
from multipage import MultiPage
from pages import thin_smear_analysis, thick_smear_analysis

#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
st.set_page_config(
     page_title="MALARIS",
     page_icon="🧊",
     layout="wide",
     initial_sidebar_state="expanded",
)

names = ['Aniss Acherar']
usernames = ['aacherar']
passwords = ['Kaleidoscope64.']
authenticator = stauth.authenticate(names,usernames,['$2y$10$isFct9/05KuS2YGh7PVac.OkM/ole83ufXLOKotIhNUXzNvLYQGaa'],'cookie_name', 'signature_key',cookie_expiry_days=30)
name, authentication_status = authenticator.login('Login','sidebar')

if authentication_status:
     st.write('Welcome *%s*' % (name))
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
 # your application
elif authentication_status == False:
 st.error('Username/password is incorrect')
elif authentication_status == None:
 st.warning('Please enter your username and password')







