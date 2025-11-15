import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

# Set page config - Force Light Theme
st.set_page_config(
    page_title="Sleep Disorder Prediction App",
    page_icon="😴",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Premium Light Theme with Enhanced Styling
st.markdown("""
<style>
    /* Ultimate light theme enforcement with premium aesthetics */
    *, *::before, *::after {
        background-color: inherit !important;
        color: inherit !important;
    }
    
    html {
        background: linear-gradient(135deg, #ffffff 0%, #fafafa 50%, #f5f5f5 100%) !important;
        color: #1a202c !important;
        font-family: 'Segoe UI', 'Roboto', 'Helvetica', sans-serif !important;
    }
    
    body {
        background: linear-gradient(135deg, #ffffff 0%, #fafafa 50%, #f5f5f5 100%) !important;
        color: #1a202c !important;
        font-smooth: always !important;
        -webkit-font-smoothing: antialiased !important;
        -moz-osx-font-smoothing: grayscale !important;
    }
    
    /* Complete dark mode override with premium styling */
    @media (prefers-color-scheme: dark) {
        html, body {
            background: linear-gradient(135deg, #ffffff 0%, #fafafa 50%, #f5f5f5 100%) !important;
            color: #1a202c !important;
        }
        
        .stApp {
            background: linear-gradient(135deg, #ffffff 0%, #fafafa 50%, #f5f5f5 100%) !important;
            color: #1a202c !important;
        }
        
        [data-testid="stAppViewContainer"] {
            background: linear-gradient(135deg, #ffffff 0%, #fafafa 50%, #f5f5f5 100%) !important;
            color: #1a202c !important;
        }
        
        .main {
            background: linear-gradient(135deg, #ffffff 0%, #f8fafc 50%, #e2e8f0 100%) !important;
            color: #1a202c !important;
        }
        
        /* Force all text elements to dark with enhanced readability */
        h1, h2, h3, h4, h5, h6, p, div, span, label, a, li, td, th {
            color: #1a202c !important;
            text-shadow: 0 1px 2px rgba(255, 255, 255, 0.8) !important;
        }
        
        /* Override any remaining dark elements */
        .stMarkdown, .stMarkdown *, .stText, .stText * {
            color: #2d3748 !important;
        }
    }
    
    /* Enhanced plotly chart styling */
    .js-plotly-plot {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%) !important;
        border-radius: 15px !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.08) !important;
    }
    
    /* Premium container styling */
    .block-container {
        background: rgba(255, 255, 255, 0.95) !important;
        backdrop-filter: blur(10px) !important;
        border-radius: 20px !important;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.08) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
    }
</style>
""", unsafe_allow_html=True)

# Enhanced Light Theme with Premium Components
st.markdown("""
<style>
    /* Premium light theme with enhanced components */
    .stApp {
        background: linear-gradient(135deg, #ffffff 0%, #fafafa 50%, #f0f2f6 100%) !important;
        color: #1a202c !important;
    }
    
    /* Enhanced app container styling */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #ffffff 0%, #fafafa 50%, #f0f2f6 100%) !important;
        color: #1a202c !important;
    }
    
    [data-testid="stHeader"] {
        background: rgba(255, 255, 255, 0.95) !important;
        backdrop-filter: blur(20px) !important;
        border-bottom: 1px solid rgba(226, 232, 240, 0.6) !important;
    }
    
    .main {
        padding-top: 2rem;
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 50%, #e2e8f0 100%) !important;
        color: #1a202c !important;
        min-height: 100vh !important;
    }
    
    /* Enhanced text styling with better contrast */
    .stApp > div {
        background: transparent !important;
        color: #1a202c !important;
    }
    
    .block-container {
        background: rgba(255, 255, 255, 0.9) !important;
        color: #1a202c !important;
        backdrop-filter: blur(10px) !important;
        border-radius: 25px !important;
        box-shadow: 0 25px 50px rgba(0, 0, 0, 0.1) !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        margin-top: 2rem !important;
        padding: 2rem !important;
    }
    
    /* Premium input styling with black borders */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > div {
        background: #ffffff !important;
        color: #2d3748 !important;
        border: 4px solid #000000 !important;
        border-radius: 15px !important;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.06), 0 0 0 2px #333333 !important;
        backdrop-filter: none !important;
        transition: all 0.3s ease !important;
    }
    
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus {
        border: 4px solid #000000 !important;
        box-shadow: 0 0 0 4px #4299e1, 0 8px 25px rgba(0, 0, 0, 0.2) !important;
        transform: translateY(-1px) !important;
        outline: none !important;
    }
    
    /* Enhanced text colors with perfect contrast */
    h1, h2, h3, h4, h5, h6, p, div, span {
        color: #1a202c !important;
        text-shadow: 0 1px 3px rgba(255, 255, 255, 0.6) !important;
    }
    
    .stMarkdown {
        color: #2d3748 !important;
    }
    
    .stMarkdown p {
        color: #4a5568 !important;
        line-height: 1.7 !important;
    }
    
    /* Enhanced heading styles */
    h1, h2, h3, h4, h5, h6 {
        color: #1a202c !important;
        font-weight: 700 !important;
        letter-spacing: -0.025em !important;
        text-shadow: 0 2px 4px rgba(255, 255, 255, 0.8) !important;
    }
    
    h1 {
        background: linear-gradient(135deg, #1a202c 0%, #2d3748 100%) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        background-clip: text !important;
    }
    
    /* Enhanced metric styling */
    .stMetric > div > div > div > div {
        font-size: 2.5rem !important;
        color: #1a202c !important;
        font-weight: 800 !important;
        text-shadow: 0 2px 4px rgba(255, 255, 255, 0.8) !important;
    }
    .stMetric > div > div > div {
        color: #4a5568 !important;
        font-weight: 500 !important;
    }
    
    /* Premium card styling */
    .prediction-box {
        padding: 2rem;
        border-radius: 20px;
        border: 2px solid #48bb78;
        background: linear-gradient(135deg, #f0fff4 0%, #e6fffa 50%, #dcfce7 100%);
        color: #1a202c;
        box-shadow: 0 20px 40px rgba(72, 187, 120, 0.2);
        backdrop-filter: blur(10px);
        position: relative;
        overflow: hidden;
    }
    
    .prediction-box::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #48bb78, #38a169, #48bb78);
        animation: shimmer 2s infinite;
    }
    
    @keyframes shimmer {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }
    
    .info-box {
        padding: 2rem;
        border-radius: 20px;
        border-left: 6px solid #4299e1;
        background: linear-gradient(135deg, #ebf8ff 0%, #bee3f8 50%, #90cdf4 100%);
        margin: 1.5rem 0;
        color: #1a202c;
        box-shadow: 0 20px 40px rgba(66, 153, 225, 0.2);
        backdrop-filter: blur(10px);
    }
    
    .academic-box {
        padding: 2rem;
        border-radius: 20px;
        border-left: 6px solid #9f7aea;
        background: linear-gradient(135deg, #faf5ff 0%, #e9d8fd 50%, #d6bcfa 100%);
        margin: 1.5rem 0;
        color: #1a202c;
        box-shadow: 0 20px 40px rgba(159, 122, 234, 0.2);
        backdrop-filter: blur(10px);
    }
    
    /* Enhanced form styling */
    .stSelectbox > div > div > div {
        background: #ffffff !important;
        color: #2d3748 !important;
        border: 2px solid rgba(226, 232, 240, 0.8) !important;
        border-radius: 15px !important;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.06) !important;
        backdrop-filter: none !important;
    }
    
    .stSelectbox > div > div > div:hover {
        border-color: #4299e1 !important;
        box-shadow: 0 8px 25px rgba(66, 153, 225, 0.15) !important;
        transform: translateY(-1px) !important;
    }
    
    .stSelectbox > div > div > div > div {
        color: #2d3748 !important;
        font-weight: 500 !important;
    }
    
    /* Enhanced input styling with black borders */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input {
        background: #ffffff !important;
        color: #2d3748 !important;
        border: 4px solid #000000 !important;
        border-radius: 15px !important;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.06), 0 0 0 2px #333333 !important;
        padding: 1rem !important;
        font-weight: 500 !important;
        transition: all 0.3s ease !important;
    }
    
    .stTextInput > div > div > input:hover,
    .stNumberInput > div > div > input:hover {
        border: 4px solid #000000 !important;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15), 0 0 0 3px #666666 !important;
        transform: translateY(-1px) !important;
    }
    
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus {
        border: 4px solid #000000 !important;
        box-shadow: 0 0 0 4px #4299e1, 0 8px 25px rgba(0, 0, 0, 0.2) !important;
        transform: translateY(-1px) !important;
        outline: none !important;
    }
    
    /* Enhanced slider styling */
    .stSlider > div > div > div {
        color: #2d3748 !important;
        font-weight: 600 !important;
    }
    
    .stSlider > div > div > div > div {
        background: linear-gradient(135deg, #4299e1 0%, #3182ce 100%) !important;
        border-radius: 10px !important;
        box-shadow: 0 4px 15px rgba(66, 153, 225, 0.3) !important;
    }
    
    /* Enhanced data and sidebar styling */
    .stDataFrame {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%) !important;
        border-radius: 20px !important;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.08) !important;
        border: 2px solid rgba(226, 232, 240, 0.6) !important;
        overflow: hidden !important;
    }
    
    .stDataFrame table {
        background: transparent !important;
        color: #2d3748 !important;
        border-collapse: separate !important;
        border-spacing: 0 !important;
    }
    
    .stDataFrame th {
        background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%) !important;
        color: #1a202c !important;
        font-weight: 700 !important;
        padding: 1rem !important;
        border-bottom: 2px solid #e2e8f0 !important;
    }
    
    .stDataFrame td {
        padding: 0.75rem 1rem !important;
        border-bottom: 1px solid rgba(226, 232, 240, 0.5) !important;
        transition: background-color 0.3s ease !important;
    }
    
    .stDataFrame tbody tr:hover {
        background: rgba(66, 153, 225, 0.05) !important;
    }
    
    .stSidebar {
        background: linear-gradient(135deg, #ffffff 0%, #f7fafc 50%, #edf2f7 100%) !important;
        border-right: 2px solid rgba(226, 232, 240, 0.8) !important;
        backdrop-filter: blur(20px) !important;
    }
    
    .stSidebar > div {
        background: transparent !important;
        color: #1a202c !important;
    }
    
    /* ULTIMATE Premium button styling with beautiful black border */
    .stButton > button {
        background: linear-gradient(135deg, #ff4757 0%, #ff3838 50%, #c44569 100%) !important;
        color: #ffffff !important;
        border: 8px solid #000000 !important;
        border-radius: 30px !important;
        font-weight: 1000 !important;
        font-size: 1.8rem !important;
        box-shadow: 0 25px 60px rgba(255, 71, 87, 0.6), inset 0 1px 0 rgba(255,255,255,0.2), 0 0 0 4px #333333 !important;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
        padding: 2rem 4rem !important;
        text-transform: uppercase !important;
        letter-spacing: 2px !important;
        width: 100% !important;
        height: 90px !important;
        outline: 6px solid #000000 !important;
        outline-offset: 8px !important;
        position: relative !important;
        overflow: hidden !important;
        font-family: 'Arial Black', 'Impact', sans-serif !important;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.8) !important;
        transform: scale(1.02) !important;
    }
    
    .stButton > button::before {
        content: '' !important;
        position: absolute !important;
        top: 0 !important;
        left: -100% !important;
        width: 100% !important;
        height: 100% !important;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.4), transparent) !important;
        transition: left 0.6s ease !important;
    }
    
    .stButton > button::after {
        content: '🚀' !important;
        position: absolute !important;
        right: 2rem !important;
        top: 50% !important;
        transform: translateY(-50%) !important;
        font-size: 2rem !important;
        opacity: 0.9 !important;
    }
    
    .stButton > button:hover::before {
        left: 100% !important;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #ff3838 0%, #c44569 50%, #2f1b69 100%) !important;
        border: 8px solid #000000 !important;
        transform: translateY(-8px) scale(1.05) !important;
        box-shadow: 0 35px 80px rgba(255, 71, 87, 0.8), inset 0 1px 0 rgba(255,255,255,0.3), 0 0 0 6px #1a1a1a !important;
        color: #ffffff !important;
        outline: 6px solid #000000 !important;
        outline-offset: 10px !important;
        text-shadow: 3px 3px 6px rgba(0, 0, 0, 1) !important;
    }
    
    /* Enhanced form container for button visibility */
    .stForm {
        background: linear-gradient(135deg, #ffffff 0%, #f7fafc 50%, #edf2f7 100%) !important;
        border-radius: 30px !important;
        border: 3px solid rgba(255, 71, 87, 0.3) !important;
        padding: 3rem !important;
        box-shadow: 0 30px 60px rgba(0, 0, 0, 0.15) !important;
        backdrop-filter: blur(15px) !important;
        position: relative !important;
        overflow: hidden !important;
    }
    
    .stForm::before {
        content: '' !important;
        position: absolute !important;
        top: 0 !important;
        left: 0 !important;
        right: 0 !important;
        height: 4px !important;
        background: linear-gradient(90deg, #ff4757, #ff3838, #c44569, #2f1b69) !important;
        opacity: 0.9 !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        background-color: #f7fafc !important;
        border-radius: 12px !important;
        padding: 0.5rem !important;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent !important;
        color: #718096 !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
    }
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        background-color: #4299e1 !important;
        color: #ffffff !important;
        box-shadow: 0 4px 15px rgba(66, 153, 225, 0.2) !important;
    }
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f7fafc 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.05);
        text-align: center;
        color: #2d3748;
    }
    .level-indicator {
        display: inline-block;
        padding: 0.6rem 1.2rem;
        border-radius: 25px;
        font-weight: 600;
        color: #ffffff;
        margin: 0.3rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    .level-1 { 
        background: linear-gradient(135deg, #48bb78 0%, #38a169 100%); 
        box-shadow: 0 4px 15px rgba(72, 187, 120, 0.3);
    }
    .level-2 { 
        background: linear-gradient(135deg, #ed8936 0%, #dd6b20 100%); 
        box-shadow: 0 4px 15px rgba(237, 137, 54, 0.3);
    }
    .level-3 { 
        background: linear-gradient(135deg, #e53e3e 0%, #c53030 100%); 
        box-shadow: 0 4px 15px rgba(229, 62, 62, 0.3);
    }
    .level-4 { 
        background: linear-gradient(135deg, #9f7aea 0%, #805ad5 100%); 
        box-shadow: 0 4px 15px rgba(159, 122, 234, 0.3);
    }
    .warning-box {
        background: linear-gradient(135deg, #fed7d7 0%, #feb2b2 100%);
        border: 1px solid #fc8181;
        border-radius: 12px;
        padding: 1rem;
        color: #742a2a;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(252, 129, 129, 0.15);
    }
    .success-box {
        background: linear-gradient(135deg, #c6f6d5 0%, #9ae6b4 100%);
        border: 1px solid #68d391;
        border-radius: 12px;
        padding: 1rem;
        color: #22543d;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(104, 211, 145, 0.15);
    }
    /* Custom scrollbar for webkit browsers */
    ::-webkit-scrollbar {
        width: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #cbd5e0 0%, #a0aec0 100%);
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #a0aec0 0%, #718096 100%);
    }
</style>
""", unsafe_allow_html=True)

# Enhanced Title with Better Visibility
st.markdown("""
<div style='text-align: center; background: linear-gradient(135deg, #ffffff 0%, #f7fafc 50%, #e2e8f0 100%); 
           padding: 3rem 2rem; border-radius: 25px; margin: 2rem 0; 
           box-shadow: 0 20px 50px rgba(0, 0, 0, 0.15); border: 3px solid #4299e1;'>
    <h1 style='font-size: 4rem; color: #1a202c; margin: 0; font-weight: 900; 
               text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3); letter-spacing: 2px;
               font-family: "Arial Black", Arial, sans-serif;'>
        <span style='font-size: 5rem; margin-right: 1rem; text-shadow: 2px 2px 8px rgba(0, 0, 0, 0.4);'></span>
        SLEEP DISORDER PREDICTION
    </h1>
    <p style='font-size: 1.5rem; color: #4a5568; margin: 1rem 0 0 0; font-weight: 600;
             text-shadow: 1px 1px 2px rgba(255, 255, 255, 0.8);'>
        Advanced AI-Powered Health Assessment System
    </p>
</div>
""", unsafe_allow_html=True)
st.markdown("---")

# Load data function with all 13 features
@st.cache_data
def load_data():
    try:
        # Try to load the data file
        df = pd.read_excel('scoring_sleep.xlsx')
        
        # Encode categorical variables only if they exist
        le_gender = LabelEncoder()
        
        # Check which columns exist and encode accordingly
        if 'Gender' in df.columns:
            df['Gender'] = le_gender.fit_transform(df['Gender'])
        
        if 'Academic Level' in df.columns:
            le_academic = LabelEncoder()
            df['Academic Level'] = le_academic.fit_transform(df['Academic Level'])
        
        if 'BMI Category' in df.columns:
            le_bmi = LabelEncoder()
            df['BMI Category'] = le_bmi.fit_transform(df['BMI Category'])
        
        return df
    except FileNotFoundError:
        # If file not found, create comprehensive sample data with all 13 features
        np.random.seed(42)
        n_samples = 500
        
        data = {
            'Person ID': range(1, n_samples + 1),
            'Gender': np.random.choice([0, 1], n_samples),  # 0: Female, 1: Male
            'Age': np.random.randint(18, 65, n_samples),
            'Academic Level': np.random.choice([0, 1, 2, 3], n_samples),  # 0-3 for Level 1-4
            'Sleep Duration': np.random.normal(7, 1.5, n_samples).clip(3, 12),
            'Quality of Sleep': np.random.randint(1, 11, n_samples),
            'Physical Activity Level': np.random.randint(15, 120, n_samples),
            'Stress Level': np.random.randint(1, 11, n_samples),
            'BMI Category': np.random.choice([0, 1, 2], n_samples),  # 0: Normal, 1: Overweight, 2: Obese
            'Heart Rate (bpm)': np.random.randint(55, 110, n_samples),
            'Daily Steps': np.random.randint(2000, 15000, n_samples),
            'Systolic BP': np.random.randint(100, 160, n_samples),
            'Diastolic BP': np.random.randint(60, 100, n_samples)
        }
        
        # Create sleep disorders with clear academic level risk progression
        sleep_disorders = []
        for i in range(n_samples):
            stress_score = data['Stress Level'][i] * 5.0  # Highest weight - 50%
            sleep_quality_score = (11 - data['Quality of Sleep'][i]) * 3.0  # Second highest - 30% 
            activity_score = max(0, 60 - data['Physical Activity Level'][i]) * 2.0  # Third - 2
            
            # Academic level risk increases progressively: Level 0=minimal, Level 3=maximum
            academic_level_risk = data['Academic Level'][i] * 3.0  # 0, 3, 6, 9, 12 progression
            duration_score = abs(8 - data['Sleep Duration'][i]) * 1.0  # Fifth - 10%
            
            # Other factors have lower weights
            bp_score = max(0, data['Systolic BP'][i] - 120) * 0.5  # 5%
            hr_score = max(0, data['Heart Rate (bpm)'][i] - 80) * 0.3  # 3%
            bmi_score = data['BMI Category'][i] * 0.5  # 5%
            steps_score = max(0, 8000 - data['Daily Steps'][i]) * 0.0002  # 2%
            age_score = max(0, data['Age'][i] - 40) * 0.2  # 2%
            gender_score = data['Gender'][i] * 0.1  # 1%
            
            base_risk_score = (stress_score + sleep_quality_score + activity_score + 
                              academic_level_risk + duration_score + bp_score + hr_score + 
                              bmi_score + steps_score + age_score + gender_score)
            
            # Academic level multiplier: Level 0=0.5x, Level 1=0.8x, Level 2=1.2x, Level 3=1.8x
            academic_multipliers = [0.5, 0.8, 1.2, 1.8]  # Progressive risk increase
            academic_multiplier = academic_multipliers[data['Academic Level'][i]]
            total_risk_score = base_risk_score * academic_multiplier
            
            # Risk thresholds adjusted for academic levels
            high_risk_threshold = 25 - (data['Academic Level'][i] * 3)  # Lower threshold for higher levels
            medium_risk_threshold = 15 - (data['Academic Level'][i] * 2)
            
            if total_risk_score > high_risk_threshold and data['Stress Level'][i] >= (7 - data['Academic Level'][i]):
                # Higher academic levels develop disorders with lower stress
                if data['Academic Level'][i] >= 3:  # Level 4 (Expert)
                    sleep_disorders.append('Insomnia' if np.random.random() < 0.9 else 'Sleep Apnea')
                elif data['Academic Level'][i] >= 2:  # Level 3 (Advanced)
                    sleep_disorders.append('Insomnia' if np.random.random() < 0.8 else 'Sleep Apnea')
                elif data['Academic Level'][i] >= 1:  # Level 2 (Intermediate)
                    if data['Heart Rate (bpm)'][i] > 85 or data['Systolic BP'][i] > 140:
                        sleep_disorders.append('Sleep Apnea' if np.random.random() < 0.6 else 'Insomnia')
                    else:
                        sleep_disorders.append('Insomnia')
                else:  # Level 1 (Basic) - lowest risk
                    if data['Stress Level'][i] >= 8 and data['Heart Rate (bpm)'][i] > 90:
                        sleep_disorders.append('Sleep Apnea')
                    else:
                        sleep_disorders.append('None')  # Often no disorder for basic level
            elif total_risk_score > medium_risk_threshold:
                # Medium risk - academic level influences disorder probability
                disorder_probabilities = [0.1, 0.3, 0.6, 0.8]  # Level 0-3 disorder chances
                disorder_chance = disorder_probabilities[data['Academic Level'][i]]
                
                if np.random.random() < disorder_chance:
                    if data['Academic Level'][i] >= 2:  # Advanced levels prefer insomnia
                        sleep_disorders.append('Insomnia')
                    elif data['Academic Level'][i] == 1:  # Intermediate - mixed
                        sleep_disorders.append('Insomnia' if np.random.random() < 0.7 else 'Sleep Apnea')
                    else:  # Basic level - mostly none
                        sleep_disorders.append('None' if np.random.random() < 0.7 else 'Sleep Apnea')
                else:
                    sleep_disorders.append('None')
            else:
                # Low risk - but higher academic levels can still develop issues
                low_risk_probabilities = [0.02, 0.08, 0.15, 0.25]  # Even low risk varies by level
                if np.random.random() < low_risk_probabilities[data['Academic Level'][i]]:
                    if data['Academic Level'][i] >= 3 and data['Stress Level'][i] >= 5:
                        sleep_disorders.append('Insomnia')
                    elif data['Academic Level'][i] >= 2 and data['Stress Level'][i] >= 6:
                        sleep_disorders.append('Insomnia' if np.random.random() < 0.8 else 'None')
                    else:
                        sleep_disorders.append('None')
                else:
                    sleep_disorders.append('None')
        
        data['Sleep Disorder'] = sleep_disorders
        
        return pd.DataFrame(data)

# Train model function with weighted features
@st.cache_data
def train_model(df):
    # Feature importance weights based on requirements
    feature_weights = {
        'Stress Level': 4.5,          # Highest importance (50%)
        'Quality of Sleep': 3.5,      # Second highest (30%)
        'Physical Activity Level': 2.5, # Third (20%)
        'Academic Level': 2.0,        # Progressive risk: Level 0=low, Level 3=high (25%)
        'Sleep Duration': 1.0,        # Fifth (10%)
        'Systolic BP': 0.5,          # Lower importance (5%)
        'BMI Category': 0.5,         # Lower importance (5%)
        'Heart Rate (bpm)': 0.3,     # Lower importance (3%)
        'Diastolic BP': 0.2,         # Lower importance (2%)
        'Daily Steps': 0.2,          # Lower importance (2%)
        'Age': 0.2,                  # Lower importance (2%)
        'Gender': 0.1                # Lowest importance (1%)
    }
    
    # Prepare features and target - use only available columns
    potential_features = ['Gender', 'Age', 'Academic Level', 'Sleep Duration',
                         'Quality of Sleep', 'Physical Activity Level', 'Stress Level',
                         'BMI Category', 'Heart Rate (bpm)', 'Daily Steps', 'Systolic BP',
                         'Diastolic BP']
    
    # Use only available columns from the dataset
    available_features = [col for col in potential_features if col in df.columns]
    X = df[available_features]
    y = df['Sleep Disorder']
    
    # Apply feature weights to scale the importance
    X_weighted = X.copy()
    for col in X_weighted.columns:
        if col in feature_weights:
            # Apply square root to moderate the effect while preserving importance
            weight = np.sqrt(feature_weights[col])
            X_weighted[col] = X_weighted[col] * weight
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_weighted, y, test_size=0.2, 
                                                        random_state=42, stratify=y)
    
    # Apply SMOTE to handle class imbalance
    smote = SMOTE(random_state=42, k_neighbors=min(3, len(X_train)-1))
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_smote)
    X_test_scaled = scaler.transform(X_test)
    
    # Train enhanced Random Forest model with optimized parameters
    rf_model = RandomForestClassifier(
        n_estimators=200,           # More trees for better performance
        max_depth=15,               # Control overfitting
        min_samples_split=5,        # Prevent overfitting
        min_samples_leaf=2,         # Prevent overfitting
        random_state=42,
        class_weight='balanced'     # Handle class imbalance
    )
    rf_model.fit(X_train_scaled, y_train_smote)
    
    # Make predictions
    y_pred = rf_model.predict(X_test_scaled)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Get feature importance
    feature_importance = dict(zip(X_weighted.columns, rf_model.feature_importances_))
    
    return (rf_model, scaler, X_weighted.columns.tolist(), accuracy, y_test, y_pred, 
            X_train, X_test, feature_importance, feature_weights)

# Load data and train model
df = load_data()
rf_model, scaler, feature_cols, model_accuracy, y_test, y_pred, X_train, X_test, feature_importance, feature_weights = train_model(df)

# Create comprehensive input form
with st.form("prediction_form"):
    st.markdown("### Enter Your Health Information")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("#### 👤 Personal Info")
        age = st.number_input("Age", min_value=16, max_value=80, value=25, 
                             help="Your current age")
        gender = st.selectbox("Gender", options=[0, 1], 
                            format_func=lambda x: "Female" if x == 0 else "Male")
        
        st.markdown("#### 🏢 Academic Info")
        academic_level = st.selectbox("Academic Level 🎓", 
                                    options=[0, 1, 2, 3],
                                    format_func=lambda x: f"Level {x+1}",
                                    help="Level 1: Basic | Level 2: Intermediate | Level 3: Advanced | Level 4: Expert")
    
    with col2:
        st.markdown("#### 😴 Sleep Patterns")
        sleep_duration = st.slider("Sleep Duration (hours)", 3.0, 12.0, 7.5, 0.5,
                                  help="Average hours of sleep per night")
        quality_of_sleep = st.slider("Sleep Quality (1-10)", 1, 10, 7,
                                   help="1=Very Poor, 10=Excellent quality sleep")
        
        st.markdown("#### 😰 **Stress Level**")
        stress_level = st.slider("Stress Level (1-10)", 1, 10, 5,
                                help="1=No Stress, 10=Extreme Stress. Consider academic workload, deadlines, life pressures")
        st.markdown(f"**Current Stress:** {'🔴 Very High' if stress_level >= 8 else '🟡 High' if stress_level >= 6 else '🟢 Moderate' if stress_level >= 4 else '✅ Low'}")
    
    with col3:
        st.markdown("#### 🏃‍♂️ Physical Activity")
        physical_activity = st.slider("Physical Activity (min/day)", 0, 150, 45,
                                    help="Minutes of physical activity per day")
        daily_steps = st.number_input("Daily Steps", 1000, 20000, 7000,
                                    help="Average steps per day")
        
        st.markdown("#### ⚖️ Body Metrics")
        bmi_category = st.selectbox("BMI Category", 
                                  options=[0, 1, 2],
                                  format_func=lambda x: ["Normal", "Overweight", "Obese"][x])
    
    with col4:
        st.markdown("#### ❤️ Health Vitals")
        heart_rate = st.number_input("Heart Rate (bpm)", 40, 130, 72,
                                   help="Resting heart rate in beats per minute")
        systolic_bp = st.number_input("Systolic BP", 80, 200, 120,
                                    help="Upper blood pressure reading")
        diastolic_bp = st.number_input("Diastolic BP", 50, 130, 80,
                                     help="Lower blood pressure reading")
    
    # Submit button
    submitted = st.form_submit_button("Predict Sleep Disorder", use_container_width=True)
        
    if submitted:
        # Prepare input data with all 13 features and apply weights
        input_data = pd.DataFrame({
            'Gender': [gender],
            'Age': [age],
            'Academic Level': [academic_level],
            'Sleep Duration': [sleep_duration],
            'Quality of Sleep': [quality_of_sleep],
            'Physical Activity Level': [physical_activity],
            'Stress Level': [stress_level],
            'BMI Category': [bmi_category],
            'Heart Rate (bpm)': [heart_rate],
            'Daily Steps': [daily_steps],
            'Systolic BP': [systolic_bp],
            'Diastolic BP': [diastolic_bp]
        })
        
        # Apply the same feature weights as in training
        input_data_weighted = input_data.copy()
        for col in input_data_weighted.columns:
            if col in feature_weights:
                weight = np.sqrt(feature_weights[col])
                input_data_weighted[col] = input_data_weighted[col] * weight
        
        # Make sure input data has the same features as training data (only available columns)
        available_input_cols = [col for col in input_data_weighted.columns if col in feature_cols]
        input_data_final = input_data_weighted[available_input_cols]
        
        # Scale the input
        input_scaled = scaler.transform(input_data_final)
        
        # Make prediction
        prediction = rf_model.predict(input_scaled)[0]
        prediction_proba = rf_model.predict_proba(input_scaled)[0]
        
        # Calculate risk score with clear academic level progression
        stress_risk = stress_level * 10
        sleep_quality_risk = (11 - quality_of_sleep) * 8
        activity_risk = max(0, 60 - physical_activity) * 5
        
        # Academic risk progression: Level 1=5, Level 2=10, Level 3=15, Level 4=20
        academic_base_risk = (academic_level + 1) * 5
        
        duration_risk = abs(8 - sleep_duration) * 4
        
        # Academic multiplier shows clear risk increase: 0.7x, 1.0x, 1.4x, 1.9x
        academic_multipliers = [0.7, 1.0, 1.4, 1.9]
        academic_multiplier = academic_multipliers[academic_level]
        
        base_total = stress_risk + sleep_quality_risk + activity_risk + academic_base_risk + duration_risk
        total_risk = base_total * academic_multiplier
        
        # Display results
        st.markdown("---")
        st.markdown("### 🎯 Comprehensive Prediction Results")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            disorder_color = '#e53e3e' if prediction != 'None' else '#48bb78'
            st.markdown(f"""
            <div class="prediction-box">
            <h3 style='font-size: 1.8rem; color: #1a202c; font-weight: 800; 
                      text-shadow: 1px 1px 3px rgba(255, 255, 255, 0.8); margin-bottom: 1rem;
                      font-family: "Arial Black", Arial, sans-serif;'>
                <span style='font-size: 2rem; margin-right: 0.5rem;'>🎯</span>
                Sleep Disorder Prediction:
            </h3>
            <h2 style="color: {disorder_color}; font-size: 2.5rem; font-weight: 900;
                      text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3); margin: 0;
                      font-family: 'Arial Black', Arial, sans-serif;">{prediction}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Risk level indicator with academic level context
            risk_labels = ["🟢 Low Risk", "🟡 Moderate Risk", "🟠 High Risk", "🔴 Very High Risk"]
            risk_colors = ["#48bb78", "#d69e2e", "#ed8936", "#e53e3e"]
            
            # Academic level affects risk thresholds
            thresholds = [40 - (academic_level * 5), 70 - (academic_level * 5), 100 - (academic_level * 5)]
            
            if total_risk >= thresholds[2]:
                risk_level = risk_labels[3]
                risk_color = risk_colors[3]
            elif total_risk >= thresholds[1]:
                risk_level = risk_labels[2] 
                risk_color = risk_colors[2]
            elif total_risk >= thresholds[0]:
                risk_level = risk_labels[1]
                risk_color = risk_colors[1]
            else:
                risk_level = risk_labels[0]
                risk_color = risk_colors[0]
                
            # Risk assessment section removed as per user request
        
        with col3:
            # Prediction probabilities chart
            classes = rf_model.classes_
            prob_df = pd.DataFrame({
                'Sleep Disorder': classes,
                'Probability': prediction_proba
            }).sort_values('Probability', ascending=False)
            
            fig_prob = px.bar(
                prob_df,
                x='Sleep Disorder',
                y='Probability',
                title="📊 Prediction Probabilities",
                color='Probability',
                color_continuous_scale='viridis',
                text='Probability'
            )
            fig_prob.update_traces(
                texttemplate='%{text:.1%}', 
                textposition='outside',
                textfont=dict(size=16, color='#1a202c', family='Arial Black')
            )
            fig_prob.update_layout(
                height=600,
                showlegend=False,
                yaxis_title="Probability",
                xaxis_title="Sleep Disorder Type",
                margin=dict(t=100, b=100, l=80, r=80),
                font=dict(size=16, color='#1a202c', family='Arial', weight='bold'),
                title_font=dict(size=20, color='#1a202c', family='Arial Black'),
                xaxis=dict(tickfont=dict(size=14, color='#1a202c', family='Arial Bold')),
                yaxis=dict(tickfont=dict(size=14, color='#1a202c', family='Arial Bold')),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_prob, use_container_width=True)        # Beautiful Health Recommendations System
        st.markdown("### 💡 Personalized Health Recommendations")
        
        # Create recommendation categories
        urgent_recommendations = []
        high_priority_recommendations = []
        moderate_recommendations = []
        general_tips = []
        academic_stress = False
        
        # 🚨 URGENT RECOMMENDATIONS
        if stress_level >= 8:
            urgent_recommendations.append({
                "icon": "🚨",
                "title": "Critical Stress Level",
                "description": "Your stress level is extremely high and requires immediate attention.",
                "actions": ["Seek professional counseling", "Practice daily meditation (10+ minutes)", "Consider temporary workload reduction"]
            })
        
        if quality_of_sleep <= 3:
            urgent_recommendations.append({
                "icon": "😴",
                "title": "Severe Sleep Quality Issues",
                "description": "Your sleep quality is critically poor and affecting your health.",
                "actions": ["Create a strict sleep routine", "Eliminate screens 2 hours before bed", "Consult a sleep specialist"]
            })
        
        # 🔥 HIGH PRIORITY RECOMMENDATIONS
        if stress_level >= 6:
            high_priority_recommendations.append({
                "icon": "😰",
                "title": "Elevated Stress Management",
                "description": "Your stress levels are high and need active management.",
                "actions": ["Practice deep breathing exercises", "Take regular 15-minute breaks", "Try progressive muscle relaxation"]
            })
            if academic_level >= 2:
                academic_stress = True
        
        if quality_of_sleep <= 6:
            high_priority_recommendations.append({
                "icon": "🌙",
                "title": "Sleep Quality Improvement",
                "description": "Your sleep quality needs significant improvement for better health.",
                "actions": ["Maintain consistent sleep schedule", "Create comfortable sleep environment", "Avoid caffeine after 2 PM"]
            })
        
        if physical_activity < 30:
            high_priority_recommendations.append({
                "icon": "🏃‍♂️",
                "title": "Increase Physical Activity",
                "description": "Your activity level is too low and affecting your sleep quality.",
                "actions": ["Start with 20-minute daily walks", "Take stairs instead of elevator", "Do simple stretching exercises"]
            })
        
        if sleep_duration < 6:
            high_priority_recommendations.append({
                "icon": "⏰",
                "title": "Insufficient Sleep Duration",
                "description": "You're not getting enough sleep for optimal health and recovery.",
                "actions": ["Aim for 7-9 hours of sleep", "Set a fixed bedtime", "Avoid late-night activities"]
            })
        
        # 🟡 MODERATE RECOMMENDATIONS
        if stress_level >= 4 and stress_level < 6:
            moderate_recommendations.append({
                "icon": "😌",
                "title": "Stress Management",
                "description": "Maintain good stress levels with healthy coping strategies.",
                "actions": ["Continue current stress management", "Try yoga or meditation", "Maintain work-life balance"]
            })
        
        if quality_of_sleep == 7:
            moderate_recommendations.append({
                "icon": "💤",
                "title": "Sleep Quality Enhancement",
                "description": "Your sleep is good, but can be optimized further.",
                "actions": ["Fine-tune your sleep routine", "Consider blackout curtains", "Keep room temperature cool (65-68°F)"]
            })
        
        if physical_activity >= 30 and physical_activity < 60:
            moderate_recommendations.append({
                "icon": "🚶‍♀️",
                "title": "Activity Level Optimization",
                "description": "Good activity level! Consider increasing for maximum benefits.",
                "actions": ["Aim for 45-60 minutes daily", "Add strength training twice a week", "Try different activities for variety"]
            })
        
        # Academic Level Specific Recommendations
        academic_risk_levels = ["Minimal", "Low", "Moderate", "High"]
        academic_colors = ["#48bb78", "#d69e2e", "#ed8936", "#e53e3e"]
        
        if academic_level >= 2 and stress_level >= 5:
            academic_stress = True
            high_priority_recommendations.append({
                "icon": "🎓",
                "title": f"Academic Stress - Level {academic_level + 1}",
                "description": f"Advanced academic level creating {academic_risk_levels[academic_level]} sleep disorder risk.",
                "actions": ["Schedule regular study breaks", "Use time-blocking techniques", "Seek academic support when needed"]
            })
        
        # Health Metrics Recommendations
        if systolic_bp > 130 or diastolic_bp > 80:
            moderate_recommendations.append({
                "icon": "🩺",
                "title": "Blood Pressure Management",
                "description": "Your blood pressure is elevated and needs attention.",
                "actions": ["Reduce sodium intake", "Increase potassium-rich foods", "Monitor BP regularly"]
            })
        
        if bmi_category >= 1:  # Overweight or Obese
            moderate_recommendations.append({
                "icon": "⚖️",
                "title": "Weight Management",
                "description": "Consider healthy weight management for better sleep.",
                "actions": ["Focus on balanced nutrition", "Increase physical activity", "Consider portion control"]
            })
        
        if heart_rate > 85:
            moderate_recommendations.append({
                "icon": "❤️",
                "title": "Heart Health",
                "description": "Your resting heart rate could benefit from improvement.",
                "actions": ["Increase cardiovascular exercise", "Practice stress reduction", "Ensure adequate hydration"]
            })
        
        # 💚 GENERAL WELLNESS TIPS
        general_tips = [
            {"icon": "🥗", "tip": "Eat a balanced diet rich in fruits and vegetables"},
            {"icon": "💧", "tip": "Drink 8-10 glasses of water daily"},
            {"icon": "🌱", "tip": "Spend time in nature for mental wellness"},
            {"icon": "📚", "tip": "Read before bed instead of using screens"},
            {"icon": "🎵", "tip": "Listen to calming music or sounds before sleep"}
        ]
        
        # Display Beautiful Recommendations
        
        # 🚨 URGENT RECOMMENDATIONS
        if urgent_recommendations:
            st.markdown("#### 🚨 Urgent Action Required")
            for rec in urgent_recommendations:
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, #fed7d7 0%, #feb2b2 100%); 
                           border-left: 5px solid #e53e3e; border-radius: 15px; 
                           padding: 1.5rem; margin: 1rem 0; box-shadow: 0 4px 15px rgba(229, 62, 62, 0.2);'>
                    <h5 style='color: #742a2a; margin: 0 0 0.5rem 0; display: flex; align-items: center;'>
                        <span style='font-size: 1.5rem; margin-right: 0.5rem;'>{rec['icon']}</span>
                        {rec['title']}
                    </h5>
                    <p style='color: #742a2a; margin: 0.5rem 0; font-weight: 500;'>{rec['description']}</p>
                    <ul style='color: #742a2a; margin: 0.5rem 0; padding-left: 1.5rem;'>
                        {"".join([f"<li>{action}</li>" for action in rec['actions']])}
                    </ul>
                </div>
                """, unsafe_allow_html=True)
        
        # 🔥 HIGH PRIORITY RECOMMENDATIONS  
        if high_priority_recommendations:
            st.markdown("#### 🔥 High Priority Actions")
            for rec in high_priority_recommendations:
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%); 
                           border-left: 5px solid #f59e0b; border-radius: 15px; 
                           padding: 1.5rem; margin: 1rem 0; box-shadow: 0 4px 15px rgba(245, 158, 11, 0.2);'>
                    <h5 style='color: #92400e; margin: 0 0 0.5rem 0; display: flex; align-items: center;'>
                        <span style='font-size: 1.5rem; margin-right: 0.5rem;'>{rec['icon']}</span>
                        {rec['title']}
                    </h5>
                    <p style='color: #92400e; margin: 0.5rem 0; font-weight: 500;'>{rec['description']}</p>
                    <ul style='color: #92400e; margin: 0.5rem 0; padding-left: 1.5rem;'>
                        {"".join([f"<li>{action}</li>" for action in rec['actions']])}
                    </ul>
                </div>
                """, unsafe_allow_html=True)
        
        # 🟡 MODERATE RECOMMENDATIONS
        if moderate_recommendations:
            st.markdown("#### 🟡 Moderate Improvements")
            for rec in moderate_recommendations:
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, #f0fff4 0%, #dcfce7 100%); 
                           border-left: 5px solid #22c55e; border-radius: 15px; 
                           padding: 1.5rem; margin: 1rem 0; box-shadow: 0 4px 15px rgba(34, 197, 94, 0.2);'>
                    <h5 style='color: #15803d; margin: 0 0 0.5rem 0; display: flex; align-items: center;'>
                        <span style='font-size: 1.5rem; margin-right: 0.5rem;'>{rec['icon']}</span>
                        {rec['title']}
                    </h5>
                    <p style='color: #15803d; margin: 0.5rem 0; font-weight: 500;'>{rec['description']}</p>
                    <ul style='color: #15803d; margin: 0.5rem 0; padding-left: 1.5rem;'>
                        {"".join([f"<li>{action}</li>" for action in rec['actions']])}
                    </ul>
                </div>
                """, unsafe_allow_html=True)
        
        # Academic-specific recommendations
        if academic_stress:
            level_names = ["Basic", "Intermediate", "Advanced", "Expert"]
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #faf5ff 0%, #e9d5ff 100%); 
                       border-left: 5px solid #a855f7; border-radius: 15px; 
                       padding: 1.5rem; margin: 1rem 0; box-shadow: 0 4px 15px rgba(168, 85, 247, 0.2);'>
                <h4 style='color: #7c2d92; margin: 0 0 1rem 0; display: flex; align-items: center;'>
                    <span style='font-size: 1.5rem; margin-right: 0.5rem;'>🎓</span>
                    Academic Level {academic_level + 1} - {level_names[academic_level]} Special Care
                </h4>
                <ul style='color: #7c2d92; margin: 0; padding-left: 1.5rem;'>
                    <li>🕒 Take 15-minute breaks every 45 minutes of study</li>
                    <li>🧘‍♀️ Practice mindfulness between study sessions</li>
                    <li>📱 Set digital boundaries: no study devices in bedroom</li>
                    <li>👥 Connect with classmates for emotional support</li>
                    <li>⚖️ Maintain strict sleep schedule during exam periods</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # 💚 GENERAL WELLNESS TIPS
        st.markdown("#### 💚 Daily Wellness Tips")
        col1, col2 = st.columns(2)
        
        with col1:
            for i, tip in enumerate(general_tips[:3]):
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, #f0fdfa 0%, #ccfbf1 100%); 
                           border-radius: 10px; padding: 1rem; margin: 0.5rem 0; 
                           border: 1px solid #5eead4; box-shadow: 0 2px 8px rgba(94, 234, 212, 0.3);'>
                    <p style='color: #0f766e; margin: 0; font-weight: 500; display: flex; align-items: center;'>
                        <span style='font-size: 1.2rem; margin-right: 0.5rem;'>{tip['icon']}</span>
                        {tip['tip']}
                    </p>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            for tip in general_tips[3:]:
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, #f0fdfa 0%, #ccfbf1 100%); 
                           border-radius: 10px; padding: 1rem; margin: 0.5rem 0; 
                           border: 1px solid #5eead4; box-shadow: 0 2px 8px rgba(94, 234, 212, 0.3);'>
                    <p style='color: #0f766e; margin: 0; font-weight: 500; display: flex; align-items: center;'>
                        <span style='font-size: 1.2rem; margin-right: 0.5rem;'>{tip['icon']}</span>
                        {tip['tip']}
                    </p>
                </div>
                """, unsafe_allow_html=True)
        
        # Success message if no issues
        if not urgent_recommendations and not high_priority_recommendations and not moderate_recommendations:
            st.markdown("""
            <div style='background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%); 
                       border-left: 5px solid #10b981; border-radius: 15px; 
                       padding: 2rem; margin: 1rem 0; text-align: center;
                       box-shadow: 0 4px 15px rgba(16, 185, 129, 0.2);'>
                <h3 style='color: #065f46; margin: 0 0 1rem 0; display: flex; align-items: center; justify-content: center;'>
                    <span style='font-size: 2rem; margin-right: 0.5rem;'>🌟</span>
                    Excellent Health Status!
                </h3>
                <p style='color: #065f46; margin: 0; font-size: 1.1rem; font-weight: 500;'>
                    Your health metrics are in great shape! Keep maintaining these wonderful habits for optimal sleep health.
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        # Risk Summary section removed as per user request

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; background: linear-gradient(135deg, #ffffff 0%, #f7fafc 100%); padding: 1.5rem; border-radius: 15px; margin-top: 2rem; border: 1px solid #e2e8f0;'>
    <h4 style='color: #2d3748; margin-bottom: 1rem;'>Sleep Disorder Prediction</h4>
    <p style='color: #48bb78; font-size: 1.1rem; margin: 0;'>Take care of your sleep health! 😴💤</p>
</div>
""", unsafe_allow_html=True)