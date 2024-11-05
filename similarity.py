from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json



def flatten_persona(persona):
    """Convert nested persona dictionary into a flat text representation"""
    text_parts = []
    
    # Add persona name
    text_parts.append(persona['persona_name'])
    
    # Add demographics
    demo = persona['demographics']
    text_parts.append(f"Age {demo['age']} {demo['gender']} {demo['location']} {demo['education']} {demo['occupation']}")
    
    # Add psychographics
    psycho = persona['psychographics']
    text_parts.append(f"Interests: {' '.join(psycho['interests'])}")
    text_parts.append(f"Values: {' '.join(psycho['values'])}")
    text_parts.append(psycho['lifestyle'])
    text_parts.append(psycho['attitudes'])
    
    # Add other important fields
    important_fields = [
        'pain_points', 'needs', 'how_company_addresses_needs',
        'preferred_communication_channels', 'trigger_events',
        'purchasing_behavior', 'influences_and_motivators',
        'goals_and_aspirations'
    ]
    
    for field in important_fields:
        if field in persona:
            text_parts.append(str(persona[field]))
    
    return ' '.join(text_parts)



def flatten_user(user):
    """Convert user dictionary into a flat text representation"""
    text_parts = []
    
    # Add essential user information with more weight to important fields
    if 'name' in user:
        text_parts.append(str(user['name']) + " " + str(user['name']))  # Double weight for name
    
    # Location and description
    for field in ['current location', 'description']:
        if field in user:
            text_parts.append(str(user[field]))
    
    # Company information with emphasis
    if 'current firm name' in user:
        text_parts.append(str(user['current firm name']) + " " + str(user['current firm name']))  # Double weight
    
    if 'firm description' in user:
        text_parts.append(str(user['firm description']))
    
    # Handle investment information which is a dictionary
    if 'firm investment' in user and isinstance(user['firm investment'], dict):
        investment_info = user['firm investment']
        if 'investment_stages' in investment_info:
            text_parts.append(str(investment_info['investment_stages']))
        if 'investment_verticals' in investment_info:
            text_parts.append(str(investment_info['investment_verticals']))
    elif 'firm investment' in user:  # If it's not a dict, add as is
        text_parts.append(str(user['firm investment']))
    
    # Add score information if available
    if '[Investors who invest in devtools, AI infra, SaaS, tech at pre-seed stage, and located in san franciso]' in user:
        score = user['[Investors who invest in devtools, AI infra, SaaS, tech at pre-seed stage, and located in san franciso]']
        text_parts.append(f"Investor score: {score}")
    
    # Create context from investment focus
    investment_context = "Focuses on investments in technology, "
    if 'firm investment' in user and isinstance(user['firm investment'], dict):
        if 'investment_verticals' in user['firm investment']:
            investment_context += user['firm investment']['investment_verticals']
        if 'investment_stages' in user['firm investment']:
            investment_context += f" at {user['firm investment']['investment_stages']} stages"
    text_parts.append(investment_context)
    
    return ' '.join(text_parts)

def process_users_and_personas(users_list, personas_list, batch_size=32):
    """Process users and personas with batching for better performance"""
    # Initialize the model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Create embeddings for personas
    persona_texts = [flatten_persona(persona) for persona in personas_list]
    persona_embeddings = model.encode(persona_texts, convert_to_numpy=True)
    
    print("Persona embeddings shape:", persona_embeddings.shape)
    # Process users in batches
    for i in range(0, len(users_list), batch_size):
        batch = users_list[i:i + batch_size]
        
        # Create user embeddings for the batch
        user_texts = [flatten_user(user) for user in batch]
        user_embeddings = model.encode(user_texts, convert_to_numpy=True)
        
        # Calculate similarities for each user in the batch
        similarities = cosine_similarity(user_embeddings, persona_embeddings)
        
        # Convert similarities to percentages and store results
        for j, user in enumerate(batch):
            print("User:", i)
            user_similarities = similarities[j]
            # Convert to percentages
            percentages = (user_similarities / np.sum(user_similarities)) * 100
            
            # Store results in user object
            user['persona_matches'] = {
                personas_list[k]['persona_name']: {
                    'similarity_percentage': round(float(percentage), 2),
                    # 'match_details': {
                    #     'demographics': personas_list[k]['demographics'],
                    #     'primary_needs': personas_list[k]['needs'],
                    #     'key_pain_points': personas_list[k]['pain_points']
                    # }
                }
                for k, percentage in enumerate(percentages)
            }
    
    return users_list

def save_results(processed_users, output_file='processed_users.json'):
    """Save processed results to a JSON file"""
    with open(output_file, 'w') as f:
        json.dump(processed_users, f, indent=2)



# Load your data
import csv
import json

def csv_to_json(csv_file_path):
    # Read CSV file and convert each row to a dictionary (object)
    data = []
    with open(csv_file_path, mode='r', encoding='utf-8') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            data.append(dict(row))  # Convert each row to a dictionary and add to list
    
    return data
# Example usage
users_list = csv_to_json('sheet1.csv')[:50]

# Load personas data
personas_list= [{"persona_name":"Tech-Savvy Entrepreneur","demographics":{"age":30,"gender":"Male","location":"San Francisco, CA","education":"MBA","occupation":"Startup Founder"},"psychographics":{"interests":["technology","innovation","business growth"],"values":["creativity","disruption","efficiency"],"lifestyle":"Fast-paced, focused on networking and growth","attitudes":"Open to new ideas, risk-taker"},"pain_points":"Struggles with scaling technology solutions and finding reliable tech partners.","needs":"Looking for innovative tech solutions that can help streamline operations.","how_company_addresses_needs":"Sundai provides cutting-edge hacking solutions that can be tailored to enhance business efficiency.","preferred_communication_channels":"Email, LinkedIn, Tech forums","preferred_device_type":"Smartphone, Laptop","trigger_events":"Funding rounds, product launches, tech conferences","purchasing_behavior":"Researches extensively, seeks peer recommendations, values demos.","potential_objections":"Concerns about the reliability and security of hacking solutions.","influences_and_motivators":"Peer success stories, industry trends, potential ROI.","goals_and_aspirations":"To lead a successful startup that revolutionizes its industry.","pitch":"Unlock your startup's potential with our tailored hacking solutions designed for innovative entrepreneurs."},{"persona_name":"Corporate IT Manager","demographics":{"age":40,"gender":"Female","location":"New York, NY","education":"Bachelor's in Computer Science","occupation":"IT Manager"},"psychographics":{"interests":["cybersecurity","networking","project management"],"values":["security","reliability","teamwork"],"lifestyle":"Structured, focused on team collaboration and project deadlines","attitudes":"Cautious, prefers proven solutions"},"pain_points":"Facing increasing cybersecurity threats and pressure to protect company data.","needs":"Requires robust security solutions that are easy to implement and manage.","how_company_addresses_needs":"Sundai offers advanced hacking solutions that enhance cybersecurity measures.","preferred_communication_channels":"Email, Webinars, Professional networks","preferred_device_type":"Desktop, Laptop","trigger_events":"Cybersecurity incidents, budget approvals, compliance audits","purchasing_behavior":"Follows a formal procurement process, involves multiple stakeholders.","potential_objections":"Worries about integration with existing systems and potential downtime.","influences_and_motivators":"Industry certifications, peer recommendations, case studies.","goals_and_aspirations":"To ensure the security and integrity of the company's IT infrastructure.","pitch":"Secure your company's future with our proven hacking solutions that protect against evolving threats."},{"persona_name":"Freelance Developer","demographics":{"age":28,"gender":"Non-binary","location":"Austin, TX","education":"Self-taught","occupation":"Freelance Software Developer"},"psychographics":{"interests":["coding","open-source projects","tech meetups"],"values":["independence","innovation","community"],"lifestyle":"Flexible, often working remotely or in co-working spaces","attitudes":"Curious, enjoys experimenting with new technologies"},"pain_points":"Limited access to advanced tools and resources for personal projects.","needs":"Wants affordable, high-quality tools that enhance development skills.","how_company_addresses_needs":"Sundai provides accessible hacking tools that empower developers to enhance their projects.","preferred_communication_channels":"Social media, GitHub, Developer forums","preferred_device_type":"Laptop, Tablet","trigger_events":"Project deadlines, new technology releases, community events","purchasing_behavior":"Tends to buy based on reviews and community feedback.","potential_objections":"Concerns about the learning curve and support availability.","influences_and_motivators":"Community endorsements, online tutorials, peer feedback.","goals_and_aspirations":"To build innovative projects and establish a strong portfolio.","pitch":"Empower your development with our innovative hacking tools designed for freelancers."},{"persona_name":"Small Business Owner","demographics":{"age":35,"gender":"Female","location":"Chicago, IL","education":"Bachelor's in Business Administration","occupation":"Owner of a local retail store"},"psychographics":{"interests":["local business development","community engagement","sustainability"],"values":["community","customer service","sustainability"],"lifestyle":"Busy, juggling multiple roles within the business","attitudes":"Community-focused, values personal relationships"},"pain_points":"Struggles with online presence and protecting customer data.","needs":"Looking for solutions that enhance online security and improve customer engagement.","how_company_addresses_needs":"Sundai provides user-friendly hacking solutions that help secure customer data and enhance online presence.","preferred_communication_channels":"Email, Facebook, In-person meetings","preferred_device_type":"Smartphone, Desktop","trigger_events":"Customer data breaches, local business events, seasonal sales","purchasing_behavior":"Relies on recommendations from other small business owners.","potential_objections":"Worries about cost and complexity of implementation.","influences_and_motivators":"Local business networks, success stories from similar businesses.","goals_and_aspirations":"To grow her business and create a loyal customer base.","pitch":"Protect your customers and grow your business with our tailored hacking solutions for small businesses."},{"persona_name":"Cybersecurity Student","demographics":{"age":22,"gender":"Male","location":"Los Angeles, CA","education":"Bachelor's in Cybersecurity","occupation":"Student"},"psychographics":{"interests":["hacking","ethical hacking competitions","technology trends"],"values":["knowledge","ethics","innovation"],"lifestyle":"Active, involved in campus activities and tech clubs","attitudes":"Eager to learn, passionate about cybersecurity"},"pain_points":"Limited access to real-world hacking tools and resources.","needs":"Wants hands-on experience with industry-standard tools.","how_company_addresses_needs":"Sundai offers educational hacking tools that provide practical experience for students.","preferred_communication_channels":"Social media, university forums, email","preferred_device_type":"Laptop, Desktop","trigger_events":"Internship opportunities, hackathons, cybersecurity workshops","purchasing_behavior":"Influenced by academic recommendations and peer reviews.","potential_objections":"Concerns about affordability and relevance of tools.","influences_and_motivators":"Mentorship from professors, participation in competitions.","goals_and_aspirations":"To become a skilled cybersecurity professional and secure a job in the industry.","pitch":"Gain hands-on experience with our educational hacking tools designed for aspiring cybersecurity professionals."}]

# Process the data
processed_users = process_users_and_personas(users_list, personas_list)

# Save results
with open('users_with_similarities.json', 'w') as f:
    json.dump(processed_users, f, indent=2)