import streamlit as st
import pandas as pd
import pickle
import dill
import numpy as np
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(page_title="Insurance Fraud Detection", layout="wide")

# Load model and explainer
@st.cache_resource
def load_model_and_explainer():
    with open("Final_modelpipe_tuned_20251105_1035.pkl", "rb") as f:
        model = pickle.load(f)
    with open("lime_explainer.dill", "rb") as f:
        explainer = dill.load(f)
    return model, explainer

try:
    model, explainer = load_model_and_explainer()
except Exception as e:
    st.error(f"Error loading model or explainer: {str(e)}")
    st.stop()

# Title
st.title("🚗 Insurance Fraud Detection System")
st.markdown("---")

# Create input form in the center
col_left, col_center, col_right = st.columns([1, 3, 1])

with col_center:
    st.header("Enter Claim Information")
    
    # Group 1: Personal Information
    st.subheader("👤 Personal Information")
    col1, col2, col3 = st.columns(3)
    with col1:
        sex = st.selectbox("Sex", ["Female", "Male"])
    with col2:
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Widow", "Divorced"])
    with col3:
        age_of_policy_holder = st.selectbox("Age of Policy Holder", 
            ["16 to 17", "18 to 20", "21 to 25", "26 to 30", "31 to 35", "36 to 40", "41 to 50", "51 to 65", "over 65"])
    
    st.markdown("---")
    
    # Group 2: Vehicle Information
    st.subheader("🚙 Vehicle Information")
    col1, col2, col3 = st.columns(3)
    with col1:
        make = st.selectbox("Make", 
            ["Honda", "Toyota", "Ford", "Mazda", "Chevrolet", "Pontiac", "Accura", "Dodge", 
             "Mercury", "Jaguar", "Nisson", "VW", "Saab", "Saturn", "Porche", "BMW", "Mecedes", "Ferrari", "Lexus"])
    with col2:
        vehicle_category = st.selectbox("Vehicle Category", ["Sport", "Utility", "Sedan"])
    with col3:
        vehicle_price = st.selectbox("Vehicle Price", 
            ["less than 20000", "20000 to 29000", "30000 to 39000", "40000 to 59000", "60000 to 69000", "more than 69000"])
    
    col1, col2 = st.columns(2)
    with col1:
        age_of_vehicle = st.selectbox("Age of Vehicle", 
            ["new", "2 years", "3 years", "4 years", "5 years", "6 years", "7 years", "more than 7"])
    with col2:
        number_of_cars = st.selectbox("Number of Cars", 
            ["1 vehicle", "2 vehicles", "3 to 4", "5 to 8", "more than 8"])
    
    st.markdown("---")
    
    # Group 3: Policy Information
    st.subheader("📋 Policy Information")
    col1, col2, col3 = st.columns(3)
    with col1:
        policy_type = st.selectbox("Policy Type", 
            ["Sport - Liability", "Sport - Collision", "Sport - All Perils",
             "Sedan - Liability", "Sedan - Collision", "Sedan - All Perils",
             "Utility - Liability", "Utility - Collision", "Utility - All Perils"])
    with col2:
        base_policy = st.selectbox("Base Policy", ["Liability", "Collision", "All Perils"])
    with col3:
        agent_type = st.selectbox("Agent Type", ["External", "Internal"])
    
    col1, col2, col3 = st.columns(3)
    with col1:
        days_policy_accident = st.selectbox("Days: Policy to Accident", 
            ["none", "1 to 7", "8 to 15", "15 to 30", "more than 30"])
    with col2:
        days_policy_claim = st.selectbox("Days: Policy to Claim", 
            ["none", "8 to 15", "15 to 30", "more than 30"])
    with col3:
        deductible = st.number_input("Deductible", min_value=300, max_value=700, value=400, step=100)
    
    col1, col2 = st.columns(2)
    with col1:
        policy_number = st.number_input("Policy Number", min_value=1, max_value=15420, value=7710, step=1)
    with col2:
        address_change_claim = st.selectbox("Address Change Before Claim", 
            ["no change", "under 6 months", "1 year", "2 to 3 years", "4 to 8 years"])
    
    st.markdown("---")
    
    # Group 4: Accident Details
    st.subheader("🚨 Accident Details")
    col1, col2, col3 = st.columns(3)
    with col1:
        month = st.selectbox("Month of Accident", 
            ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    with col2:
        day_of_week = st.selectbox("Day of Week (Accident)", 
            ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
    with col3:
        week_of_month = st.number_input("Week of Month (Accident)", min_value=1, max_value=5, value=3, step=1)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        accident_area = st.selectbox("Accident Area", ["Urban", "Rural"])
    with col2:
        fault = st.selectbox("Fault", ["Policy Holder", "Third Party"])
    with col3:
        police_report_filed = st.selectbox("Police Report Filed", ["No", "Yes"])
    with col4:
        witness_present = st.selectbox("Witness Present", ["No", "Yes"])
    
    st.markdown("---")
    
    # Group 5: Claims History
    st.subheader("📊 Claims History")
    col1, col2, col3 = st.columns(3)
    with col1:
        month_claimed = st.selectbox("Month Claimed", 
            ["0", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    with col2:
        day_of_week_claimed = st.selectbox("Day of Week Claimed", 
            ["0", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
    with col3:
        week_of_month_claimed = st.number_input("Week of Month Claimed", min_value=1, max_value=5, value=3, step=1)
    
    col1, col2 = st.columns(2)
    with col1:
        past_number_of_claims = st.selectbox("Past Number of Claims", 
            ["none", "1", "2 to 4", "more than 4"])
    with col2:
        number_of_suppliments = st.selectbox("Number of Supplements", 
            ["none", "1 to 2", "3 to 5", "more than 5"])
    
    st.markdown("---")
    
    # Group 6: Other Information
    st.subheader("ℹ️ Other Information")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        age = st.number_input("Age", min_value=0, max_value=80, value=38, step=1)
    with col2:
        rep_number = st.number_input("Rep Number", min_value=1, max_value=16, value=8, step=1)
    with col3:
        driver_rating = st.number_input("Driver Rating", min_value=1, max_value=4, value=2, step=1)
    with col4:
        year = st.number_input("Year", min_value=1994, max_value=1996, value=1995, step=1)
    
    st.markdown("---")
    
    # Predict button
    predict_button = st.button("🔍 Predict Fraud", type="primary", use_container_width=True)

# Prediction section
if predict_button:
    # Create input dataframe
    input_data = pd.DataFrame({
        'Month': [month],
        'WeekOfMonth': [week_of_month],
        'DayOfWeek': [day_of_week],
        'Make': [make],
        'AccidentArea': [accident_area],
        'DayOfWeekClaimed': [day_of_week_claimed],
        'MonthClaimed': [month_claimed],
        'WeekOfMonthClaimed': [week_of_month_claimed],
        'Sex': [sex],
        'MaritalStatus': [marital_status],
        'Age': [age],
        'Fault': [fault],
        'PolicyType': [policy_type],
        'VehicleCategory': [vehicle_category],
        'VehiclePrice': [vehicle_price],
        'Days_Policy_Accident': [days_policy_accident],
        'Days_Policy_Claim': [days_policy_claim],
        'PastNumberOfClaims': [past_number_of_claims],
        'AgeOfVehicle': [age_of_vehicle],
        'AgeOfPolicyHolder': [age_of_policy_holder],
        'PoliceReportFiled': [police_report_filed],
        'WitnessPresent': [witness_present],
        'AgentType': [agent_type],
        'NumberOfSuppliments': [number_of_suppliments],
        'AddressChange_Claim': [address_change_claim],
        'NumberOfCars': [number_of_cars],
        'Year': [year],
        'BasePolicy': [base_policy],
        'PolicyNumber': [policy_number],
        'RepNumber': [rep_number],
        'Deductible': [deductible],
        'DriverRating': [driver_rating]
    })
    
    # Make prediction
    try:
        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[0]
        
        # Display prediction
        col_left, col_center, col_right = st.columns([1, 3, 1])
        with col_center:
            st.markdown("---")
            st.header("Prediction Result")
            
            if prediction == 1:
                st.error("⚠️ **FRAUDULENT CLAIM DETECTED**")
                confidence = prediction_proba[1] * 100
            else:
                st.success("✅ **LEGITIMATE CLAIM**")
                confidence = prediction_proba[0] * 100
            
            st.metric("Confidence", f"{confidence:.2f}%")
            
            # Display probabilities
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Legitimate Probability", f"{prediction_proba[0]*100:.2f}%")
            with col2:
                st.metric("Fraud Probability", f"{prediction_proba[1]*100:.2f}%")
            
            st.markdown("---")
            
            # LIME Explanation
            st.header("LIME Explanation")
            st.markdown("*Understanding the prediction: Features contributing to the decision*")
            
            # Transform input data using preprocessing step
            preprocessed_data = model['prepocessing'].transform(input_data)
            
            # Generate LIME explanation
            with st.spinner("Generating explanation..."):
                # Convert to numpy array for LIME
                if isinstance(preprocessed_data, pd.DataFrame):
                    preprocessed_array = preprocessed_data.values
                else:
                    preprocessed_array = preprocessed_data
                
                # Get the model's predict_proba function
                exp = explainer.explain_instance(
                    preprocessed_array[0],
                    model['model'].predict_proba,
                    num_features=10
                )
                
                # Create figure
                fig = exp.as_pyplot_figure()
                fig.set_size_inches(10, 6)
                fig.tight_layout()
                
                st.pyplot(fig)
                plt.close()
            
            st.markdown("---")
            st.info("💡 **Note**: Positive values (orange) indicate features pushing towards fraud detection, "
                   "while negative values (blue) indicate features suggesting legitimacy.")
    
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        st.exception(e)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Insurance Fraud Detection System | Powered by XGBoost and LIME</p>
    </div>
    """,
    unsafe_allow_html=True
)