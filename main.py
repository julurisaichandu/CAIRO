import json
import os
import streamlit as st

from src.pdl_api import enrich_profiles
from src.hypothesis_generator import generate_hypothesis
from src.openai_api import OpenAIApi
from src.prompts import hypotheis_update_prompt
from src.utils import parse_llm_response
from src.deck_generation import process_multiple_jsons
# from src.process_investor_list import get_aldo_data
# from streamlit_card import card
import streamlit.components.v1 as components

import pandas as pd

st.session_state.df = pd.read_csv('dev_tools_investors_preseed.xlsx - Sheet1.csv')

if os.getenv("OPENAI_API_KEY"):
    print("OPENAI_API_KEY is set")
else:
    print("OPENAI_API_KEY is not set")
openai_api = OpenAIApi(os.getenv("OPENAI_API_KEY"))

st.set_page_config(layout="wide")

st.header("CAIRO: Validate market hypotheses in minutes")


st.markdown(
            (
                '<hr style="background-color: #71eea8; margin-top: 0;'
                ' margin-bottom: 0; height: 3px; border: none; border-radius: 3px;">'
            ),
            unsafe_allow_html=True,
        )


with st.form("Get Company Information"):
    st.write("Please Input Company Information below")

    col1, col2 = st.columns([3,4])

    # Input fields for company information
    with col1:
        company_website = st.text_input("Company Website URL")
        # Color picker for selecting a color
        logo_upload = st.file_uploader("Upload Logo", accept_multiple_files=False)
        selected_color = st.color_picker("Pick a Brand Color")
        # location = st.text_input("Location")
        # Add checkboxes for market type
        # st.write("How type of product are you trying to sell?")
        # market_physical = st.checkbox("Physical Product")
        # market_digital = st.checkbox("Digital Product")
        # market_service = st.checkbox("Service")
        # main_message = st.text_area("Main Message")
        uploaded_file = st.file_uploader("Upload lead list", type=["csv", "xlsx"])
        

            

    with col2:
        company_name = st.text_input("Company Name",placeholder="Company Name")
        company_description = st.text_area("Company Description",height=200)
        product_description = st.text_area("Detailed Product/Service Description",height=200)
        # st.write("What is your Business Model")
        # customer_b2b = st.checkbox("B2B")
        # customer_b2c = st.checkbox("B2C")
        # customer_b2b2c = st.checkbox("B2B2C")
        # target_customer = st.text_area("Target Customer")

        
    # TODO: Add a chat bot

    # Submit button
    submitted = st.form_submit_button("Submit", help="Submit the form to generate hypotheses")

    if submitted:
        # Output the form inputs
        st.write("Company Website:", company_website)
        st.write("Company Name:", company_name)
        # st.write("Location:", location)
        st.write("Company Description:", company_description)
        st.write("Product/Service Description:", product_description)
        # st.write("Main Message:", main_message)
        # st.write("Target Customer:", target_customer)
        # st.write("Customer Type - B2B:", customer_b2b)
        # st.write("Customer Type - B2C:", customer_b2c)
        # st.write("Customer Type - B2B2C:", customer_b2b2c)
        # st.write("Market Type - Physical Product:", market_physical)
        # st.write("Market Type - Digital Product:", market_digital)
        # st.write("Market Type - Service:", market_service)
        st.write("Selected Color:", selected_color)

        form_data = {
            "company_website": company_website,
            "company_name": company_name,
            # "location": location,
            "company_description": company_description,
            "product_description": product_description,
            # "main_message": main_message,
            # "target_customer": target_customer,
            # "customer_b2b": customer_b2b,
            # "customer_b2c": customer_b2c,
            # "customer_b2b2c": customer_b2b2c,
            # "market_physical": market_physical,
            # "market_digital": market_digital,
            # "market_service": market_service,
            "color": selected_color,
            # "logo_upload": [file.name for file in logo_upload] if logo_upload else []
        }

        form_data_json = json.dumps(form_data)
        if 'form_data_json' not in st.session_state:
            st.session_state.form_data = form_data

                # Check if a file is uploaded
        
        
        if uploaded_file:
            # For CSV files
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            # For Excel files
            elif uploaded_file.name.endswith(".xlsx"):
                df = pd.read_excel(uploaded_file, engine='openpyxl')

            # scrape lead list and
            if 'website' in df.columns:
                linkedin_profiles = df['website'].tolist()
            if linkedin_profiles:
                print(linkedin_profiles)
                enriched_data = enrich_profiles(linkedin_profiles)
                with open("my_pdl_enrichment.json", "w") as out:
                    out.write(json.dumps(enriched_data) + "\n")
        
        with st.spinner("Generating Hypotheses..."):
            print("Generating Hypotheses...")
            hypothesis = generate_hypothesis(form_data_json)
            print("Hypothesis Generated")

        if hypothesis is None:
            st.error("Hypothesis generation failed. Please try again.")

        if 'hypothesis' not in st.session_state:
            st.session_state.hypothesis = hypothesis
        
        if 'conversation' not in st.session_state:
            st.session_state.conversation = []

if 'hypothesis' in st.session_state:
    # commenting because its same as the below list of the personas
    # st.write("Hypothesis:", st.session_state.hypothesis)
    # TODO: Better UI Display of Hypothesis
    i = 1
    for person in st.session_state.hypothesis:
        contain_person = st.expander(f"Persona {i}")
        with contain_person:
            st.write("Name:", person["persona_name"])
            st.write("Demographics:", person["demographics"])
            st.write("Psychographics:", person["psychographics"])
            st.write("Pain Points:", person["pain_points"])
            st.write("Needs:", person["needs"])
            st.write("How the Company Addresses These Needs:", person["how_company_addresses_needs"])
            st.write("Preferred Communication Channels:", person["preferred_communication_channels"])
            st.write("Preferred Device Type:", person["preferred_device_type"])
            st.write("Trigger Events:", person["trigger_events"])
            st.write("Purchasing Behavior and Decision-Making Process:", person["purchasing_behavior"])
            st.write("Potential Objections to Overcome:", person["potential_objections"])
            st.write("Influences and Motivators:", person["influences_and_motivators"])
            st.write("Goals and Aspirations:", person["goals_and_aspirations"])
            st.write("Pitch:", person["pitch"])
            i += 1
                
    st.subheader("Confirm the Hypothesis")

    # Display conversation history
    for message in st.session_state.conversation:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if user_input := st.chat_input("Please let me know if you would like to correct any of the hypothesis if you are not satisfied with the result"):
        # Add user message to conversation
        st.session_state.conversation.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Generate LLM response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            input_prompt = hypotheis_update_prompt.format(
                company_details=st.session_state.conversation,
                hypotheses=st.session_state.hypothesis,
                conversation_history=json.dumps(st.session_state.conversation),
                user_input=user_input
            )
            messages = [{"role": "user", "content": input_prompt}]
            full_response = openai_api.get_completion(messages)
            new_hypothesis = parse_llm_response(full_response)
            message_placeholder.code(new_hypothesis, language="json")
            st.session_state.hypothesis = new_hypothesis

        
        # Add assistant response to conversation
        st.session_state.conversation.append({"role": "assistant", "content": full_response})

    # "Done" button to end the conversation
    if st.button("Start Processing Hypothesis"):
        st.write("CAIRO starting...")
        # Here you can add any wrap-up logic or final processing
        hypothesis = st.session_state.hypothesis
        company_data = st.session_state.form_data

        # combine each of the data into a single dictionary
        for item in hypothesis:
            item.update(company_data)

        # for item in hypothesis:
        #     st.write(item)

    # TODO: 2 RANK THE LIST OF LEADS ASYNC
        deck_links = process_multiple_jsons(hypothesis)
        # st.write(deck_links)

        for i, (hypotheses_, deck_link) in enumerate(zip(hypothesis, deck_links)):
            hypo_dict = {k:v for k, v in hypotheses_.items() if k in ["hypothesis", "pain_point", "pitch"]}
            st.markdown(f"### Persona {i+1}")
            with st.expander(f"Expand for details"):
                for key, value in hypo_dict.items():
                    st.markdown(f"**{key}**: {value}")
                st.markdown(f"**Deck Link**: {deck_link[1]}")
                components.iframe(deck_link[1], height=500, scrolling=True)
                st.dataframe(st.session_state.df.sample(5))


