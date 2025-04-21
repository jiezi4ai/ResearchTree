
import datetime
import pandas as pd
import numpy as np

import streamlit as st

st.set_page_config(layout="wide")

st.title('Discover Literature Works and Build A Research Tree!')

if 'num_title_input_boxes' not in st.session_state:
    st.session_state['num_title_input_boxes'] = 3

# --- Function to increment the number of boxes ---
# This function will be called when the button is clicked.
def add_paper_title_box():
    st.session_state['num_title_input_boxes'] += 1

with st.sidebar:
    st.header("Inputs")

    with st.container(border=True):
        st.text("Let's start from some seed papers:")
        
        # for research topics
        with st.container():
            topic_input = st.text_input("Research Topic", "Input research topic you interested in")

        # for seed paper titles
        seed_title_widgets = []
        seed_title_inputs = {}
        with st.container():
            # Loop based on the number stored in session state
            for i in range(st.session_state['num_title_input_boxes']):
                key = f"seed_paper_title_{i}"
                label = f"Input seed paper title #{i + 1}"
                user_input = st.text_input(label, key=key)

                # Store the widget or its value if you need to process them later
                seed_title_widgets.append(user_input) # Note: this stores the *current* value, not the widget object itself
                seed_title_inputs[key] = st.session_state[key] # More reliable way to get current value

        st.button("➕ Add More Input Boxes", on_click=add_paper_title_box, type="primary")

        # for seed paper dois
        with st.container():
            dois_input = st.text_area(
                "Input seed paper dois here:",
                "Example like 'https://doi.org/10.7717/peerj.4375, 10.7717/peerj.4375, 2406.10252' would be accepted."
                "Make sure to intercept each doi with ','. "
            )

    with st.expander("More options: "):
        st.text("Here are optional set up for date range, domains, etc.:")
        with st.container():
            st.text("Specify date range for literature works:")
            start_dt_input = st.date_input("date from ", datetime.date(2020, 1, 1))
            end_dt_input = st.date_input("date to ", datetime.date.today())
        with st.container():
            st.text("Specify domins / filed of works:")
            fileds_input = st.text_area(
                "Example like 'Computer Science, Physics,Mathematics'. Reference from https://api.semanticscholar.org/api-docs/graph#tag/Paper-Data/operation/get_graph_paper_relevance_search."
            )
        with st.container():
            st.text("Specify literature expansion")
            if_citation_expansion = st.checkbox("Citation Expansion")
            if_author_expansion = st.checkbox("Author Expansion")
            if_topic_expansion = st.checkbox("Topic Expansion")
            if_smart_recommendation = st.checkbox("Smart Recomendation")

    with st.container():
        col1, col2, col3 = st.columns(3)
        with col1:
            st.button("Auto Pilot", type="primary")
        with col2:
            st.button("Co-Pilot", type="primary")
        with col3:
            st.button("Reset")

with st.container():
    st.header("Outputs")
    with st.container():
        st.write(topic_input)

    with st.container():
        # Access values directly from session state using the keys we generated
        if st.session_state['num_title_input_boxes'] > 0:
            for i in range(st.session_state['num_title_input_boxes']):
                key = f"seed_paper_title_{i}"
                st.write(f"Box #{i + 1} (Key: `{key}`): **{st.session_state[key]}**")
        else:
            st.write("No input boxes yet.")

    with st.container():
        st.write(dois_input)