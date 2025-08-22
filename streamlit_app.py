import streamlit as st
import pandas as pd
import joblib
import numpy as np
from PIL import Image
import os

# Set page configuration
st.set_page_config(
    page_title="Schistosoma haematobium Risk Prediction Dashboard",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Styled Cards ---
st.markdown("""
<style>
.metric-card {
    background-color: #FFFFFF;
    border: 1px solid #E0E0E0;
    border-radius: 10px;
    padding: 25px;
    text-align: center;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    transition: transform 0.2s;
}
.metric-card:hover {
    transform: scale(1.05);
}
.metric-card h3 {
    margin: 0;
    color: #4F8BC9;
}
.metric-card p {
    margin: 10px 0 0 0;
    color: #2E4053;
}
</style>
""", unsafe_allow_html=True)

# Load the trained model
@st.cache_resource
def load_model():
    # Make sure the model file is in the correct path
    return joblib.load('schisto_model.pkl')

# Load the dataset
@st.cache_data
def load_data():
    # Ensure the CSV file path is correct
    df = pd.read_csv('schist.csv')
    # Clean data
    df['Age'] = df['Age'].replace({'10 -14 Years': '10 - 14 Years', '15- 19 Years': '15 - 19 Years'})
    # This line seems to replace 'Haematuria' with 'No'. Be sure this is the intended logic.
    df['Haematuria'] = df['Haematuria'].replace({'Haematuria': 'No'})
    return df

# Main title and header
st.title("🩺 Schistosoma haematobium Risk Prediction Dashboard")
st.markdown("---")

# Load model and data
model = load_model()
df = load_data()

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page", ["Dashboard Overview", "Risk Assessment", "About Schistosomiasis"])

# --- Page Content ---

if page == "Dashboard Overview":
    st.header("Schistosoma haematobium")
    # --- Styled Metric Cards ---
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🧬 Species</h3>
            <p><i>Schistosoma haematobium</i> is the main cause of urinary schistosomiasis in humans.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>🚻 Main Symptom</h3>
            <p>Blood in urine (haematuria) is the classic sign of infection, especially in children.</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>🏞️ Habitat</h3>
            <p>Larvae are released by freshwater snails and infect people during water contact.</p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class="metric-card">
            <h3>⚠️ Complications</h3>
            <p>Chronic infection can cause bladder damage, kidney failure.</p>
        </div>
        """, unsafe_allow_html=True)
 
    st.markdown("<br>", unsafe_allow_html=True) # Add some space
    st.markdown("---")
    

elif page == "Risk Assessment":
    st.header("🎯 Personal Risk Assessment")
    st.write("Please fill out the form below to assess your risk of Schistosoma haematobium infection:")
    
    # Create input form
    with st.form("risk_assessment_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            sex = st.selectbox("Sex", ["Female", "Male"])
            age = st.selectbox("Age", ["10 - 14 Years", "15 - 19 Years"])
            haematuria = st.selectbox("Haematuria (Blood in urine)", ["No", "Yes"])
            female_gs = st.selectbox("Female GS (Knowledge)", ["No", "Yes"])
            water_school = st.selectbox("Water facility in the school", ["No", "Yes"])
            water_home = st.selectbox("Water facility in the home", ["Available", "Fairly", "Poor", "Not available"])
            drinking_water = st.selectbox("Source of drinking water",
                                          ["Borehole", "Borehole/Freshwater body", "Satchet/bottled water",
                                           "All water sources", "Borehole/Satchet/bottled water", "Freshwater body"])
        
        with col2:
            visit_waterbodies = st.selectbox("Visit to Waterbodies", ["No", "Yes"])
            direct_contact = st.selectbox("Direct contact with freshwater body", ["No", "Yes"])
            abdominal_pain = st.selectbox("Lower abdominal pain", ["No", "Mild", "Moderate", "Severe"])
            urination_pain = st.selectbox("Urination pain", ["No", "Mild", "Moderate", "Severe"])
            family_history = st.selectbox("Family history", ["No", "Yes", "Not sure"])
            genital_itching = st.selectbox("Have you experienced itching or burning in your genitals?",
                                           ["No", "Mild", "Moderate", "Severe"])
            genital_discharge = st.selectbox("Do you experience genital discharge?", ["No", "Yes", "Sometimes"])
        
        submitted = st.form_submit_button("Assess Risk")
        
        if submitted:
            # Create input dataframe
            input_data = pd.DataFrame({
                'Sex': [sex], 'Age': [age], 'Haematuria': [haematuria],
                'Female GS(Knowledge)': [female_gs], 'Water facility in the school': [water_school],
                'Water facility in the home': [water_home], 'source of drinking water': [drinking_water],
                'Visit to Waterbodies': [visit_waterbodies], 'Direct contact with freshwater body': [direct_contact],
                'lower abdominal pain': [abdominal_pain], 'urination pain': [urination_pain],
                'family history': [family_history],
                'Have you experienced itching or burning in your genitals?': [genital_itching],
                'Do you experience genital discharge?': [genital_discharge]
            })
            
            # Make prediction
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0]
            
            st.markdown("---")
            st.subheader("🎯 Risk Assessment Results")
            
            col1, col2 = st.columns([1, 2]) # Adjust column widths
            
            with col1:
                if prediction == 1:
                    st.error(f"### ⚠️ HIGH RISK")
                    risk_level = "HIGH"
                else:
                    st.success(f"### ✅ LOW RISK")
                    risk_level = "LOW"
                
                st.metric("Probability of Infection", f"{probability[1]*100:.1f}%")
            
            with col2:
                if risk_level == "HIGH":
                    st.warning("**Important Recommendations:**")
                    st.markdown("""
                    - **Consult a healthcare provider immediately.** Professional medical advice is crucial.
                    - **Get tested for schistosomiasis** to confirm the diagnosis.
                    - **Avoid all contact with freshwater sources** like rivers, lakes, and ponds.
                    - **Follow prescribed treatment** diligently if you are diagnosed.
                    """)
                else:
                    st.info("**Preventative Measures:**")
                    st.markdown("""
                    - **Continue avoiding contact with potentially contaminated freshwater.**
                    - **Maintain good hygiene practices,** especially after being near water sources.
                    - **Stay informed** about schistosomiasis risks in your area.
                    - **Consider regular health check-ups** if you live in an endemic region.
                    """)


elif page == "About Schistosomiasis":
    st.header("🔬 About Schistosomiasis (Bilharzia)")
    
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("How is Schistosomiasis Transmitted?")
        st.write("""
        Schistosomiasis is transmitted when people come into contact with freshwater contaminated 
        by the parasite. The life cycle involves freshwater snails that release tiny larvae 
        (cercariae) into the water. These larvae can penetrate human skin during activities 
        like swimming, bathing, fishing, or farming. Once inside the body, they develop into 
        adult worms that live in blood vessels and produce eggs, which cause disease.
        """)

    
    with col2:
        st.subheader("What is Schistosomiasis?")
        st.write("""
        Schistosomiasis, also known as bilharzia, is a disease caused by parasitic flatworms called schistosomes. 
        *Schistosoma haematobium* is one of the main species that causes urogenital schistosomiasis, 
        primarily affecting the bladder and urinary tract, which can lead to serious long-term health problems if left untreated.
        """)
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)

    with col1:
        st.info("#### 💧 Transmission")
        st.write("""
        Infection occurs when skin comes into contact with contaminated freshwater where certain types of snails that carry the parasites live. The parasite larvae penetrate the skin and develop into adult worms inside the body.
        """)
    
    with col2:
        st.warning("#### 🩺 Symptoms")
        st.write("""
        - **Blood in urine (haematuria)** - The most common sign.
        - Painful or frequent urination.
        - Lower abdominal pain.
        - In women: Genital sores, vaginal bleeding, and pain during intercourse.
        - In men: Pathology of the seminal vesicles and prostate.
        """)
        
    with col3:
        st.success("#### 🛡️ Prevention & Treatment")
        st.write("""
        - **Avoid wading, swimming, or bathing** in freshwater in endemic areas.
        - Use safe water from boreholes or taps.
        - Improve sanitation to prevent human waste from contaminating water.
        """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; font-size:16px; padding:10px;">
        <b>Disclaimer:</b> This tool is for educational purposes only and should not replace professional medical advice.
    </div>
    """,
    unsafe_allow_html=True
)
