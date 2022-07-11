import streamlit as st


with st.form("my_form", clear_on_submit=True):
    name=st.text_input("Enter full name")
    email=st.text_input("Enter email")
    message=st.text_area("Message")

    # Every form must have a submit button.
    submitted = st.form_submit_button("Submit")
    if submitted:
        st.write("Thanks for your message !")