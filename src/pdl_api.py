import json
from peopledatalabs import PDLPY
import re
from dotenv import load_dotenv
import os
load_dotenv()
pdl_api_key = os.getenv("PDL_API_KEY")

def clean_linkedin_url(url):

    # Remove trailing slash if present
    url = url.rstrip('/')
    
    # Step 1: Handle 'in' vs 'in/'
    if '/in' in url and not '/in/' in url:
        url = url.replace('/in', '/in/')
        
    # Step 2: Remove 'https://www.' prefix
    for prefix in ['https://www.', 'http://www.']:
        if url.startswith(prefix):
            url = url.replace(prefix, '')
            break  # Exit the loop after the first mat
    

    return url

def filter_relevant_fields(api_output):
    filtered_output = {
        "full_name": api_output.get("full_name"),
        "gender": api_output.get("sex"),
        "location": {
            "country": api_output.get("location_country"),
        },
        "education": [
            {
                "school": {
                    "name": edu["school"].get("name") if edu and edu.get("school") else None,
                    "country": edu["school"].get("location").get("country") if edu and edu.get("school") and edu["school"].get("location") else None,
                    "website": edu["school"].get("website") if edu and edu.get("school") else None,
                },
                "start_date": edu.get("start_date") if edu else None,
                "majors": edu.get("majors") if edu else None,
            } for edu in api_output.get("education", []) if edu is not None
        ],
        "interests": api_output.get("interests"),
        "skills": api_output.get("skills"),
    }
    # print("filtered output\n", filtered_output)
    return filtered_output


def enrich_profiles(linkedin_profiles):
    """Enrich LinkedIn profiles using PDL API."""
    client = PDLPY(api_key=pdl_api_key)
    
    # Clean LinkedIn URLs
    normalized_profiles = [clean_linkedin_url(profile) for profile in linkedin_profiles]
    # Filter out any None values resulting from invalid URLs
    normalized_profiles = [profile for profile in normalized_profiles if profile is not None]
    print("normalized profiles\n", normalized_profiles)
    data = {
        "required": "profiles",
        "include_if_matched": "True",
        "requests": [{"params": {"profile": [profile]}} for profile in normalized_profiles]
    }
    
    # Call the PDL API
    json_responses = client.person.bulk(**data).json()

    enriched_profiles = {}
    for response in json_responses:
        if response['status'] == 200:
            enriched_profiles[response['data']['id']] = filter_relevant_fields(response['data'])
        else:
            enriched_profiles[response['data']['id']] = {"error": response}
    
    return enriched_profiles

