import streamlit as st
import joblib
import numpy as np

# Cache the model loading function with resource caching
@st.cache_resource
def load_model():
    model_path = "random_forest_model.pkl"
    try:
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        st.error("Model file not found. Please train and save the model first.")
        return None

@st.cache_resource
def load_scaler():
    scaler_path = "scaler.pkl"
    try:
        scaler = joblib.load(scaler_path)
        return scaler
    except FileNotFoundError:
        st.error("Scaler file not found. Please train and save the scaler first.")
        return None

@st.cache_resource
def load_target_scaler():
    target_scaler_path = "target_scaler_df1.pkl"
    try:
        target_scaler = joblib.load(target_scaler_path)
        return target_scaler
    except FileNotFoundError:
        st.error("Target scaler file not found. Please train and save the target scaler first.")
        return None

# Load model, scaler, and target scaler
model = load_model()
scaler = load_scaler()
target_scaler = load_target_scaler()

# Page Header
st.markdown('<h1 class="title">🏎 Motor Speed Predictor</h1>', unsafe_allow_html=True)
st.markdown("**Predict motor speed based on various operational parameters.**")

# Min-Max values for each parameter
parameter_ranges = {
    "ambient": (-25.29, 30.71),
    "coolant": (13.76, 101.60),
    "u_d": (-131.53, 131.47),
    "u_q": (-278.00, 133.03),
    "torque": (-246.47, 261.01),
    "i_d": (-278.00, 0.05),
    "i_q": (-293.43, 301.71),
    "pm": (20.86, 113.61),
    "stator_yoke": (18.08, 99.86),
    "stator_tooth": (18.13, 111.95),
    "stator_winding": (18.59, 141.36),
}

# Input Fields with Sliders
with st.form("input_form"):
    st.subheader("📊 Input Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        ambient = st.slider(
            "❄️ Ambient Temperature (°C)", 
            min_value=parameter_ranges["ambient"][0], 
            max_value=parameter_ranges["ambient"][1], 
            value=25.0,
            help="Temperature of the surrounding environment."
        )
        coolant = st.slider(
            "💦 Coolant Temperature (°C)", 
            min_value=parameter_ranges["coolant"][0], 
            max_value=parameter_ranges["coolant"][1], 
            value=30.0,
            help="Temperature of the cooling liquid in the system."
        )
        u_d = st.slider(
            "⚡ Voltage D-component", 
            min_value=parameter_ranges["u_d"][0], 
            max_value=parameter_ranges["u_d"][1], 
            value=0.0,
            help="Direct-axis voltage component of the motor."
        )
        u_q = st.slider(
            "⚡ Voltage Q-component", 
            min_value=parameter_ranges["u_q"][0], 
            max_value=parameter_ranges["u_q"][1], 
            value=0.0,
            help="Quadrature-axis voltage component of the motor."
        )
        torque = st.slider(
            "🔩 Torque (Nm)", 
            min_value=parameter_ranges["torque"][0], 
            max_value=parameter_ranges["torque"][1], 
            value=0.0,
            help="Rotational force applied to the motor shaft in Newton-meters."
        )
        i_d = st.slider(
            "🔌 Current D-component", 
            min_value=parameter_ranges["i_d"][0], 
            max_value=parameter_ranges["i_d"][1], 
            value=0.0,
            help="Direct-axis current component related to motor efficiency."
        )
    
    with col2:
        i_q = st.slider(
            "🔌 Current Q-component", 
            min_value=parameter_ranges["i_q"][0], 
            max_value=parameter_ranges["i_q"][1], 
            value=0.0,
            help="Quadrature-axis current component affecting torque production."
        )
        pm = st.slider(
            "🧲 Permanent Magnet Temp (°C)", 
            min_value=parameter_ranges["pm"][0], 
            max_value=parameter_ranges["pm"][1], 
            value=50.0,
            help="Temperature of the permanent magnets inside the motor."
        )
        stator_yoke = st.slider(
            "🌀 Stator Yoke Temp (°C)", 
            min_value=parameter_ranges["stator_yoke"][0], 
            max_value=parameter_ranges["stator_yoke"][1], 
            value=60.0,
            help="Heat level in the motor's stator yoke component."
        )
        stator_tooth = st.slider(
            "🦷 Stator Tooth Temp (°C)", 
            min_value=parameter_ranges["stator_tooth"][0], 
            max_value=parameter_ranges["stator_tooth"][1], 
            value=65.0,
            help="Temperature of the stator teeth, affecting efficiency."
        )
        stator_winding = st.slider(
            "🎛 Stator Winding Temp (°C)", 
            min_value=parameter_ranges["stator_winding"][0], 
            max_value=parameter_ranges["stator_winding"][1], 
            value=70.0,
            help="Heat level in the stator windings, impacting performance."
        )
    
    predict_button = st.form_submit_button("🚀 Predict Motor Speed")

# Prediction Logic
if predict_button:
    input_values = [ambient, coolant, u_d, u_q, torque, i_d, i_q, pm, stator_yoke, stator_tooth, stator_winding]
    
    @st.cache_data
    def predict_motor_speed(input_values, _model, _scaler, _target_scaler):
        features = np.array(input_values).reshape(1, -1)
        scaled_features = _scaler.transform(features)
        prediction = _model.predict(scaled_features)[0]
        
        if _target_scaler:
            prediction = _target_scaler.inverse_transform([[prediction]])[0][0]
        
        return prediction
    
    prediction = predict_motor_speed(input_values, model, scaler, target_scaler)
    st.success(f'🚀 Predicted Motor Speed: **{prediction:.2f} RPM**')

# Additional Information
with st.expander("ℹ How to Use"):
    st.markdown("""
    1. Adjust the slider values for each parameter.
    2. Click the 'Predict Motor Speed' button.
    3. View the predicted motor speed.
    4. Modify inputs and try again for different scenarios.
    """)

