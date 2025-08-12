import streamlit as st

# Import your page modules
import eda_page
import model_design_page
import model_implementation_page

# Sidebar
st.sidebar.title("Navigation")

# Create tabs in the sidebar
tabs = ["Model's Implementation 🎯","Models Design 🧢", "Data Visualization 📊"]
selected_tab = st.sidebar.radio("", tabs, label_visibility="collapsed")

# Routing based on selected tab
if selected_tab == "Data Visualization 📊":
    eda_page.show()
elif selected_tab == "Models Design 🧢":
    model_design_page.show()
elif selected_tab == "Model's Implementation 🎯":
    model_implementation_page.show()