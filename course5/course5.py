import streamlit as st
import os
import json
import uuid
import numpy as np
from typing import List, Dict, Any, TypedDict, Annotated
from openai import OpenAI

try:
    import chromadb
except ImportError:
    st.error("ChromaDB not available. Please install chromadb.")
    st.stop()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "voc-334616391126677392082867d3c9ccd42eb5.77541555")

client = OpenAI(
    base_url="https://openai.vocareum.com/v1",
    api_key=OPENAI_API_KEY
)


@st.cache_resource
def init_chromadb():
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    return chroma_client


class AppState(TypedDict):
    buyer_preferences: str
    matched_listings: List[Dict]
    personalized_descriptions: List[str]
    current_step: str


class ListingGenerator:
    def __init__(self):
        self.client = client

    def generate_listings(self, num_listings: int = 15) -> List[Dict]:
        listings = []
        property_types = ["Modern House", "Luxury Apartment", "Cozy Condo", "Spacious Townhouse",
                          "Victorian Home", "Contemporary Loft", "Garden Apartment", "Penthouse"]

        for i in range(num_listings):
            try:
                listing = {
                    "property_id": f"PROP_{uuid.uuid4().hex[:8].upper()}",
                    "address": f"{100 + i * 5} {['Oak', 'Pine', 'Maple', 'Cedar', 'Elm'][i % 5]} Street, {['Downtown', 'Westside', 'Eastgate', 'Northpark', 'Southville'][i % 5]}",
                    "property_type": property_types[i % len(property_types)],
                    "price": f"${250000 + i * 50000:,}",
                    "bedrooms": 2 + (i % 4),
                    "bathrooms": 1 + (i % 3),
                    "sqft": 1200 + i * 200,
                    "features": f"Updated kitchen, hardwood floors, {'garage' if i % 2 == 0 else 'parking'}, {'garden' if i % 3 == 0 else 'balcony'}",
                    "neighborhood": f"Quiet residential area with {'schools' if i % 2 == 0 else 'parks'} nearby",
                    "description": f"Beautiful {property_types[i % len(property_types)].lower()} with modern amenities and great location. Perfect for {'families' if i % 2 == 0 else 'professionals'}."
                }
                listings.append(listing)
            except Exception as e:
                st.error(f"Error generating listing {i}: {e}")

        return listings


class VectorDBManager:
    def __init__(self, chroma_client):
        self.client = chroma_client
        self.collection_name = "real_estate_listings"

    def create_collection(self):
        try:
            self.collection = self.client.get_collection(self.collection_name)
        except:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Real estate listings with embeddings"}
            )
        return self.collection

    def get_embedding(self, text: str):
        try:
            response = client.embeddings.create(
                model="text-embedding-ada-002",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            st.error(f"Error getting embedding: {e}")
            return None

    def store_listings(self, listings: List[Dict]):
        collection = self.create_collection()

        documents = []
        metadatas = []
        ids = []
        embeddings = []

        for listing in listings:
            text = f"{listing['property_type']} {listing['address']} {listing['features']} {listing['neighborhood']} {listing['description']}"
            embedding = self.get_embedding(text)

            if embedding:
                documents.append(text)
                metadatas.append(listing)
                ids.append(listing['property_id'])
                embeddings.append(embedding)

        if embeddings:
            collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids,
                embeddings=embeddings
            )

        return len(embeddings)

    def search_listings(self, query: str, n_results: int = 5) -> List[Dict]:
        collection = self.create_collection()

        query_embedding = self.get_embedding(query)
        if not query_embedding:
            return []

        try:
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )

            matched_listings = []
            if results['metadatas']:
                matched_listings = results['metadatas'][0]

            return matched_listings
        except Exception as e:
            st.error(f"Error searching listings: {e}")
            return []


def analyze_preferences(preferences: str) -> str:
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are a real estate expert. Analyze buyer preferences and extract key search terms for property matching."
                },
                {
                    "role": "user",
                    "content": f"Analyze these buyer preferences and create a comprehensive search query: {preferences}"
                }
            ],
            max_tokens=200
        )
        return response.choices[0].message.content or preferences
    except Exception as e:
        st.error(f"Error analyzing preferences: {e}")
        return preferences


def personalize_description(preferences: str, listing: Dict) -> str:
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are a real estate agent. Rewrite property descriptions to appeal to specific buyer preferences while keeping all facts accurate."
                },
                {
                    "role": "user",
                    "content": f"""
                    Buyer Preferences: {preferences}

                    Property Details:
                    Type: {listing.get('property_type', '')}
                    Address: {listing.get('address', '')}
                    Price: {listing.get('price', '')}
                    Bedrooms: {listing.get('bedrooms', '')}
                    Bathrooms: {listing.get('bathrooms', '')}
                    Square Feet: {listing.get('sqft', '')}
                    Features: {listing.get('features', '')}
                    Neighborhood: {listing.get('neighborhood', '')}
                    Description: {listing.get('description', '')}

                    Create a personalized description that highlights aspects most relevant to the buyer's needs.
                    """
                }
            ],
            max_tokens=300
        )
        return response.choices[0].message.content or listing.get('description', 'No description available')
    except Exception as e:
        st.error(f"Error personalizing description: {e}")
        return listing.get('description', 'No description available')


def main():
    st.set_page_config(
        page_title="HomeMatch - Personalized Real Estate",
        page_icon="🏠",
        layout="wide"
    )

    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

        * {
            font-family: 'Poppins', sans-serif;
        }

        .main-header {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 10px;
            color: white;
            text-align: center;
            margin-bottom: 2rem;
        }

        .property-card {
            background: white;
            border-radius: 15px;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            border-left: 4px solid #667eea;
        }

        .preference-box {
            background: #f8f9fa;
            border-radius: 10px;
            padding: 1.5rem;
            margin: 1rem 0;
        }

        .stats-container {
            background: #e3f2fd;
            border-radius: 10px;
            padding: 1rem;
            margin: 1rem 0;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class="main-header">
            <h1>🏠 HomeMatch</h1>
            <p>Personalized Real Estate Matching with AI</p>
        </div>
    """, unsafe_allow_html=True)

    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.client = init_chromadb()
        st.session_state.db_manager = VectorDBManager(st.session_state.client)

        with st.spinner("Setting up property database..."):
            generator = ListingGenerator()
            listings = generator.generate_listings(15)
            stored_count = st.session_state.db_manager.store_listings(listings)
            st.session_state.total_listings = stored_count

    with st.sidebar:
        st.markdown(f"""
            <div class="stats-container">
                <h3>📊 Database Stats</h3>
                <p>Total Properties: {st.session_state.get('total_listings', 0)}</p>
                <p>Powered by: ChromaDB + OpenAI</p>
                <p>Features: Vector Search, AI Personalization</p>
            </div>
        """, unsafe_allow_html=True)

        if st.button("🔄 Regenerate Property Database"):
            with st.spinner("Regenerating properties..."):
                try:
                    st.session_state.client.delete_collection("real_estate_listings")
                except:
                    pass

                generator = ListingGenerator()
                listings = generator.generate_listings(15)
                stored_count = st.session_state.db_manager.store_listings(listings)
                st.session_state.total_listings = stored_count
                st.success(f"Generated {stored_count} new properties!")

    st.markdown('<div class="preference-box">', unsafe_allow_html=True)
    st.subheader("🎯 Tell us your preferences")

    preferences = st.text_area(
        "Describe your ideal home:",
        placeholder="e.g., I'm looking for a modern 3-bedroom house with a garden, good schools nearby, budget around $400k, quiet neighborhood...",
        height=100
    )

    col1, col2 = st.columns([1, 4])
    with col1:
        search_button = st.button("🔍 Find My Home", type="primary")

    st.markdown('</div>', unsafe_allow_html=True)

    if search_button and preferences:
        with st.spinner("Finding your perfect match..."):
            analyzed_prefs = analyze_preferences(preferences)

            matched_listings = st.session_state.db_manager.search_listings(analyzed_prefs, n_results=3)

            if matched_listings:
                st.subheader("🎯 Perfect Matches for You")

                for i, listing in enumerate(matched_listings):
                    with st.expander(
                            f"🏠 {listing.get('property_type', 'Property')} - {listing.get('address', 'Address not available')}",
                            expanded=True):
                        col1, col2 = st.columns([2, 1])

                        with col1:
                            st.markdown(f"**Price:** {listing.get('price', 'N/A')}")
                            st.markdown(
                                f"**Bedrooms:** {listing.get('bedrooms', 'N/A')} | **Bathrooms:** {listing.get('bathrooms', 'N/A')}")
                            st.markdown(f"**Square Footage:** {listing.get('sqft', 'N/A')} sq ft")
                            st.markdown(f"**Features:** {listing.get('features', 'N/A')}")
                            st.markdown(f"**Neighborhood:** {listing.get('neighborhood', 'N/A')}")

                        with col2:
                            st.markdown("**Property ID:**")
                            st.code(listing.get('property_id', 'N/A'))

                        with st.spinner("Personalizing description..."):
                            personalized_desc = personalize_description(preferences, listing)

                        st.markdown("**🎯 Personalized for You:**")
                        st.markdown(f'<div class="property-card">{personalized_desc}</div>', unsafe_allow_html=True)

                        st.markdown("---")
            else:
                st.warning("No matching properties found. Try adjusting your preferences.")

    elif search_button and not preferences:
        st.warning("Please enter your preferences to find matching properties.")


if __name__ == "__main__":
    main()
