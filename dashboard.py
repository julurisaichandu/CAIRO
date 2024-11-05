import streamlit as st
import pandas as pd
from streamlit_card import card
import plotly.graph_objects as go
from streamlit_echarts import st_echarts
import json

# Set page config with custom theme
import streamlit as st
import pandas as pd
from streamlit_card import card
import plotly.graph_objects as go
from streamlit_echarts import st_echarts

# Set page config
st.set_page_config(layout="wide", page_title="Persona Matcher")

# Custom CSS for compact layout
st.markdown("""
<style>
    .compact-card {
        border: 1px solid #e0e0e0;
        border-radius: 5px;
        padding: 10px;
        margin: 5px 0;
        background-color: white;
    }
    .persona-header {
        font-size: 14px;
        font-weight: bold;
        margin: 0;
        padding: 5px 0;
    }
    .small-text {
        font-size: 12px;
        margin: 2px 0;
    }
    .metric-value {
        font-size: 13px;
        color: #0066cc;
    }
    div[data-testid="stExpander"] {
        border: none;
        box-shadow: none;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 4px 8px;
        font-size: 12px;
    }
    .compact-chart {
        height: 200px !important;
    }
    div[data-testid="stHorizontalBlock"] > div {
        padding: 0 10px;
    }
</style>
""", unsafe_allow_html=True)

def create_compact_radar_chart(similarities):
    categories = list(similarities.keys())
    values = [similarities[cat]['similarity_percentage'] for cat in categories]
    
    option = {
        "tooltip": {"trigger": "axis"},
        "radar": {
            "indicator": [{"name": cat[:10] + "...", "max": 100} for cat in categories],
            "radius": "65%",
        },
        "series": [{
            "type": "radar",
            "data": [{
                "value": values,
                "name": "Match %",
                "areaStyle": {"opacity": 0.3}
            }]
        }]
    }
    return option

def display_compact_person_card(person):
    with st.container():
        st.markdown(f"""
        <div class="compact-card">
            <div class="persona-header">👤 {person['name']}</div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([1.2, 1])
        with col1:
            st.markdown(f"""
            <div class="small-text">📍 {person['current location']}</div>
            <div class="small-text">🏢 {person['current firm name']}</div>
            """, unsafe_allow_html=True)
            
            if isinstance(person['firm investment'], dict):
                with st.expander("Investment Details"):
                    st.markdown(f"""
                    <div class="small-text">🎯 {person['firm investment']['investment_stages']}</div>
                    <div class="small-text">💼 {person['firm investment']['investment_verticals']}</div>
                    """, unsafe_allow_html=True)
        
        with col2:
            if 'persona_matches' in person:
                st_echarts(
                    create_compact_radar_chart(person['persona_matches']),
                    height="150px",
                )

def main():
    col1, col2, col3 = st.columns([1, 1, 1.5])
    
    with col1:
        st.markdown("### 👥 Personas")
        for persona in personas_list:
            with st.expander(f"🎯 {persona['persona_name']}", expanded=False):
                tabs = st.tabs(["Demo", "Psycho", "Needs"])
                
                with tabs[0]:
                    for key, value in persona['demographics'].items():
                        st.markdown(f"""
                        <div class="small-text"><b>{key.title()}:</b> {value}</div>
                        """, unsafe_allow_html=True)
                
                with tabs[1]:
                    for key, value in persona['psychographics'].items():
                        if isinstance(value, list):
                            st.markdown(f"""
                            <div class="small-text"><b>{key.title()}:</b> {', '.join(value)}</div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="small-text"><b>{key.title()}:</b> {value}</div>
                            """, unsafe_allow_html=True)
                
                with tabs[2]:
                    st.markdown(f"""
                    <div class="small-text"><b>Pain Points:</b> {persona['pain_points']}</div>
                    <div class="small-text"><b>Needs:</b> {persona['needs']}</div>
                    """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 🏢 Companies")
        for company in companies_list:
            st.markdown(f"""
            <div class="compact-card">
                <div class="persona-header">{company['name']}</div>
                <div class="small-text">📍 {company['location']} | 💼 {company['industry']}</div>
                <div class="small-text">💰 {company['funding']} | 🏗️ {company['size']}</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("### 👨‍💼 People")
        for person in processed_users:
            display_compact_person_card(person)



import random

def generate_mock_personas(num_personas=5):
    personas = []
    persona_names = [
        "Tech-Savvy Entrepreneur",
        "Corporate IT Manager",
        "Freelance Developer",
        "Small Business Owner",
        "Cybersecurity Student"
    ]
    
    for name in persona_names:
        persona = {
            "persona_name": name,
            "demographics": {
                "age": random.randint(25, 45),
                "gender": random.choice(["Male", "Female", "Non-binary"]),
                "location": random.choice(["San Francisco, CA", "New York, NY", "Austin, TX"]),
                "education": random.choice(["MBA", "Bachelor's", "Self-taught"]),
                "occupation": random.choice(["Startup Founder", "IT Manager", "Developer"])
            },
            "psychographics": {
                "interests": random.sample(["technology", "innovation", "cybersecurity", "AI", "cloud"], k=3),
                "values": random.sample(["creativity", "security", "efficiency", "growth", "learning"], k=3),
                "lifestyle": random.choice(["Fast-paced", "Structured", "Flexible"]),
                "attitudes": random.choice(["Risk-taker", "Analytical", "Innovation-focused"])
            },
            "pain_points": "Struggles with scaling and finding reliable solutions.",
            "needs": "Looking for innovative tools and platforms.",
            "how_company_addresses_needs": "Providing cutting-edge solutions.",
            "preferred_communication_channels": "Email, LinkedIn",
            "preferred_device_type": "Laptop, Smartphone",
            "trigger_events": "Product launches, tech conferences",
            "purchasing_behavior": "Research-driven, peer recommendations",
            "potential_objections": "Concerns about implementation and security",
            "influences_and_motivators": "Industry trends, ROI potential",
            "goals_and_aspirations": "To lead innovation in their field"
        }
        personas.append(persona)
    
    return personas

def generate_mock_companies(num_companies=5):
    companies = []
    company_types = ["Tech", "AI", "Cloud", "Security", "Data"]
    
    for i in range(num_companies):
        company = {
            "name": f"{random.choice(company_types)} Solutions {i+1}",
            "description": f"Leading provider of innovative solutions in {random.choice(company_types)}.",
            "location": random.choice(["San Francisco", "New York", "Austin"]),
            "size": random.choice(["Startup", "SME", "Enterprise"]),
            "industry": random.choice(["Software", "AI/ML", "Cloud Computing"]),
            "funding": random.choice(["Seed", "Series A", "Series B"])
        }
        companies.append(company)
    
    return companies

def generate_mock_users(num_users=5, personas_list=None):
    users = []
    
    for i in range(num_users):
        # Generate random similarity scores that sum to 100
        raw_scores = [random.randint(1, 100) for _ in range(len(personas_list))]
        total = sum(raw_scores)
        normalized_scores = [score/total * 100 for score in raw_scores]
        
        persona_matches = {
            persona['persona_name']: {
                "similarity_percentage": round(normalized_scores[idx], 2),
                "match_details": {
                    "demographics": persona['demographics'],
                    "primary_needs": persona['needs'],
                    "key_pain_points": persona['pain_points']
                }
            }
            for idx, persona in enumerate(personas_list)
        }
        
        user = {
            "name": f"Investor {i+1}",
            "current location": random.choice(["San Francisco, CA", "New York, NY", "Austin, TX"]),
            "description": f"Experienced investor focusing on {random.choice(['AI', 'Cloud', 'Security'])} investments.",
            "website": f"https://linkedin.com/in/investor{i+1}",
            "other links": {
                "twitter_url": f"https://twitter.com/investor{i+1}",
                "crunchbase_url": f"https://crunchbase.com/person/investor{i+1}"
            },
            "current firm name": f"Venture Capital Firm {i+1}",
            "firm url": f"https://vcfirm{i+1}.com",
            "firm description": "Leading venture capital firm focused on early-stage investments.",
            "firm investment": {
                "investment_stages": random.choice(["Seed, Pre-seed", "Series A, Seed", "Pre-seed"]),
                "investment_verticals": "Developer Tools, Security, AI, Enterprise Applications"
            },
            "persona_matches": persona_matches
        }
        users.append(user)
    
    return users

# Generate all mock data
mock_personas = generate_mock_personas()
mock_companies = generate_mock_companies()
mock_users = generate_mock_users(num_users=5, personas_list=mock_personas)


# Run the app
if __name__ == "__main__":
    # Load your data here
    personas_list = mock_personas
    processed_users = mock_users
    companies_list = mock_companies
    
    main()