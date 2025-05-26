import streamlit as st

# Page configuration - MUST be the first Streamlit command
st.set_page_config(
    page_title="Insurance Cost Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)
# streamlit run .\insurance_streamlit_app.py

import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os

def load_model_components():
    """Load model components with error handling"""
    try:
        if not os.path.exists('insurance_model_components.joblib'):
            return None, "Model file 'insurance_model_components.joblib' not found!"
        
        components = joblib.load('insurance_model_components.joblib')
        return components, None
    except Exception as e:
        return None, f"Error loading model: {str(e)}"

def prepare_features(data, categorical_cols, encoding_maps, global_mean):
    """Prepare features including engineered features"""
    if isinstance(data, dict):
        data = pd.DataFrame([data])
        
    df = data.copy()
    
    # 1. Health Risk Score
    df['age_risk'] = (df['age'] - 18) / 10
    
    # BMI risk
    df['bmi_risk'] = 0
    df.loc[df['bmi'] < 18.5, 'bmi_risk'] = 1
    df.loc[(df['bmi'] >= 18.5) & (df['bmi'] < 25), 'bmi_risk'] = 0
    df.loc[(df['bmi'] >= 25) & (df['bmi'] < 30), 'bmi_risk'] = 1
    df.loc[df['bmi'] >= 30, 'bmi_risk'] = 2
    
    # Smoker risk
    df['smoker_risk'] = np.where(df['smoker'] == 'yes', 3, 0)
    
    # Total health risk score
    df['health_risk_score'] = df['age_risk'] + df['bmi_risk'] + df['smoker_risk']
    
    # 2. Feature Interactions
    df['smoker_binary'] = np.where(df['smoker'] == 'yes', 1, 0)
    df['age_smoker'] = df['age'] * df['smoker_binary']
    df['bmi_smoker'] = df['bmi'] * df['smoker_binary']
    df['bmi_age'] = df['bmi'] * df['age'] / 100
    
    # 3. BMI Categories
    df['bmi_category'] = 'normal'
    df.loc[df['bmi'] < 18.5, 'bmi_category'] = 'underweight'
    df.loc[(df['bmi'] >= 25) & (df['bmi'] < 30), 'bmi_category'] = 'overweight'
    df.loc[df['bmi'] >= 30, 'bmi_category'] = 'obese'
    
    # 4. Age Groups
    df['age_group'] = 'middle'
    df.loc[df['age'] < 35, 'age_group'] = 'young'
    df.loc[df['age'] >= 50, 'age_group'] = 'senior'
    
    # 5. Apply target encoding
    for col in categorical_cols:
        encoding_map = encoding_maps.get(col, {})
        df[f'{col}_encoded'] = df[col].map(encoding_map)
        
        # Handle unseen categories
        if df[f'{col}_encoded'].isna().any():
            df[f'{col}_encoded'].fillna(global_mean, inplace=True)
            
    return df

def predict_insurance_charges(data, components):
    """Make prediction using the loaded model components"""
    model = components['model']
    preprocessor = components['preprocessor']
    categorical_cols = components['categorical_cols']
    encoding_maps = components['encoding_maps']
    log_transform = components['log_transform']
    global_mean = components['global_mean']
    
    # Prepare features
    df_prepared = prepare_features(data, categorical_cols, encoding_maps, global_mean)
    
    # Transform using preprocessing pipeline
    X_transformed = preprocessor.transform(df_prepared)
    
    # Make prediction
    pred_log = model.predict(X_transformed)
    
    # Transform back if needed
    if log_transform:
        pred = np.expm1(pred_log)
    else:
        pred = pred_log
        
    return pred[0] if len(pred) == 1 else pred, df_prepared

def get_bmi_category(bmi):
    """Get BMI category description"""
    if bmi < 18.5:
        return "Underweight", "🔵"
    elif bmi < 25:
        return "Normal", "🟢"
    elif bmi < 30:
        return "Overweight", "🟡"
    else:
        return "Obese", "🔴"

def create_gauge_chart(value, title, max_value, color_ranges):
    """Create a gauge chart for health metrics"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title, 'font': {'size': 16}},
        gauge = {
            'axis': {'range': [None, max_value], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': color_ranges,
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': value
            }
        }
    ))
    
    fig.update_layout(height=250)
    return fig

def create_feature_importance_chart(predicted_df, components):
    """Create feature importance visualization"""
    model = components['model']
    preprocessor = components['preprocessor']
    
    # Get feature names
    numeric_features = ['age', 'bmi', 'children', 'health_risk_score', 
                       'age_smoker', 'bmi_smoker', 'bmi_age']
    categorical_features = ['sex', 'smoker', 'region', 'bmi_category', 'age_group']
    encoded_features = [f'{col}_encoded' for col in categorical_features]
    
    feature_names = []
    
    # Access ColumnTransformer from Pipeline
    column_transformer = preprocessor.named_steps['preprocessor']
    
    for name, _, cols in column_transformer.transformers_:
        if name == 'num':
            feature_names.extend(numeric_features)
        elif name == 'cat':
            cat_encoder = column_transformer.named_transformers_['cat'].named_steps['onehot']
            feature_names.extend(cat_encoder.get_feature_names_out(categorical_features))
        elif name == 'encoded':
            feature_names.extend(encoded_features)
    
    # Get coefficients
    coefficients = model.coef_[:len(feature_names)]
    
    # Create DataFrame and sort by absolute coefficient
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients,
        'Abs_Coefficient': np.abs(coefficients)
    }).sort_values('Abs_Coefficient', ascending=True).tail(10)
    
    # Create horizontal bar chart
    colors = ['red' if x < 0 else 'blue' for x in coef_df['Coefficient']]
    
    fig = go.Figure(go.Bar(
        y=coef_df['Feature'],
        x=coef_df['Coefficient'],
        orientation='h',
        marker_color=colors,
        text=[f'{val:.3f}' for val in coef_df['Coefficient']],
        textposition='auto',
    ))
    
    fig.update_layout(
        title="Top 10 Feature Importance (Model Coefficients)",
        xaxis_title="Coefficient Value",
        yaxis_title="Features",
        height=400,
        showlegend=False
    )
    
    return fig

def get_risk_interpretation(health_score, age, bmi, smoker):
    """Provide risk interpretation"""
    risk_factors = []
    
    if smoker == 'yes':
        risk_factors.append("🚬 Smoking significantly increases insurance costs")
    
    if bmi >= 30:
        risk_factors.append("⚖️ Obesity (BMI ≥ 30) is a major cost factor")
    elif bmi >= 25:
        risk_factors.append("⚖️ Overweight status may increase costs")
    
    if age >= 50:
        risk_factors.append("👴 Senior age group typically has higher costs")
    elif age < 35:
        risk_factors.append("👶 Young age group typically has lower costs")
    
    return risk_factors

def main():
    # Load model components
    components, error_msg = load_model_components()
    
    if components is None:
        st.error(f"❌ {error_msg}")
        st.info("Please ensure the model file is in the same directory as this application.")
        st.stop()
    
    # App header
    st.title("🏥 Insurance Cost Predictor - Dr. Eng. Farrikh Alzami,M.kom")
    st.markdown("---")
    
    # Create input form
    st.subheader("📝 Customer Information")
    
    # Create columns for better layout
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Personal Information**")
        age = st.number_input(
            "Age (years)", 
            min_value=18, 
            max_value=100, 
            value=30,
            help="Age of the insurance applicant"
        )
        
        sex = st.radio(
            "Sex",
            options=['male', 'female'],
            horizontal=True
        )
        
        children = st.number_input(
            "Number of Children", 
            min_value=0, 
            max_value=10, 
            value=0,
            help="Number of dependent children"
        )
    
    with col2:
        st.markdown("**Health Information**")
        bmi = st.number_input(
            "BMI (Body Mass Index)", 
            min_value=10.0, 
            max_value=60.0, 
            value=25.0,
            step=0.1,
            help="Body Mass Index (kg/m²)"
        )
        
        smoker = st.radio(
            "Smoking Status",
            options=['no', 'yes'],
            horizontal=True
        )
    
    with col3:
        st.markdown("**Location Information**")
        region = st.radio(
            "Region",
            options=['northeast', 'northwest', 'southeast', 'southwest'],
            help="Geographic region"
        )
    
    # Input validation
    if st.button("🔮 Predict Insurance Cost", type="primary"):
        # Validate inputs
        if bmi <= 0:
            st.error("❌ BMI must be greater than 0")
            return
        
        if age <= 0:
            st.error("❌ Age must be greater than 0")
            return
        
        if children < 0:
            st.error("❌ Number of children cannot be negative")
            return
        
        # Prepare input data
        input_data = {
            'age': age,
            'sex': sex,
            'bmi': bmi,
            'children': children,
            'smoker': smoker,
            'region': region
        }
        
        # Make prediction
        try:
            predicted_cost, prepared_df = predict_insurance_charges(input_data, components)
            
            # Display results
            st.markdown("---")
            st.subheader("📊 Prediction Results")
            
            # Main prediction result
            st.success(f"**Predicted Annual Insurance Cost: ${predicted_cost:,.2f}**")
            
            # Create metrics display
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # BMI Category
                bmi_cat, bmi_emoji = get_bmi_category(bmi)
                st.metric(
                    label="BMI Category",
                    value=f"{bmi_cat} {bmi_emoji}",
                    help=f"BMI: {bmi:.1f}"
                )
            
            with col2:
                # Health Risk Score
                health_score = prepared_df['health_risk_score'].iloc[0]
                st.metric(
                    label="Health Risk Score",
                    value=f"{health_score:.1f}",
                    help="Calculated based on age, BMI, and smoking status"
                )
            
            with col3:
                # Age Group
                age_group = prepared_df['age_group'].iloc[0].title()
                st.metric(
                    label="Age Group",
                    value=age_group,
                    help=f"Age: {age} years"
                )
            
            # Visualizations
            st.markdown("---")
            st.subheader("📈 Analysis & Insights")
            
            # Create two columns for visualizations
            viz_col1, viz_col2 = st.columns(2)
            
            with viz_col1:
                # Health Risk Gauge
                gauge_fig = create_gauge_chart(
                    value=health_score,
                    title="Health Risk Score",
                    max_value=10,
                    color_ranges=[
                        {'range': [0, 3], 'color': "lightgreen"},
                        {'range': [3, 6], 'color': "yellow"},
                        {'range': [6, 10], 'color': "red"}
                    ]
                )
                st.plotly_chart(gauge_fig, use_container_width=True)
            
            with viz_col2:
                # Feature importance chart
                try:
                    importance_fig = create_feature_importance_chart(prepared_df, components)
                    st.plotly_chart(importance_fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"⚠️ Could not generate feature importance chart: {str(e)}")
                    # Alternative: Show simple coefficient table
                    st.markdown("**Top Important Features:**")
                    try:
                        model = components['model']
                        if hasattr(model, 'coef_'):
                            # Show top 5 absolute coefficients
                            coefs = model.coef_
                            top_coefs = sorted(enumerate(coefs), key=lambda x: abs(x[1]), reverse=True)[:5]
                            for i, (idx, coef) in enumerate(top_coefs, 1):
                                direction = "📈" if coef > 0 else "📉"
                                st.write(f"{i}. Feature {idx}: {direction} {abs(coef):.3f}")
                    except:
                        st.info("Feature importance data not available")
            
            # Risk interpretation
            st.markdown("---")
            st.subheader("🔍 Risk Analysis")
            
            risk_factors = get_risk_interpretation(health_score, age, bmi, smoker)
            
            if risk_factors:
                st.markdown("**Key factors affecting your insurance cost:**")
                for factor in risk_factors:
                    st.markdown(f"• {factor}")
            else:
                st.success("✅ You have relatively low risk factors!")
            
            # Cost breakdown explanation
            st.markdown("---")
            st.subheader("💡 Understanding Your Prediction")
            
            # Cost comparison
            if smoker == 'yes':
                st.warning("🚬 **Smoking Impact**: Smoking is the strongest predictor of high insurance costs. Quitting smoking could significantly reduce your premiums.")
            
            if bmi >= 30:
                st.warning("⚖️ **BMI Impact**: Obesity (BMI ≥ 30) significantly increases insurance costs due to associated health risks.")
            elif bmi < 18.5:
                st.info("⚖️ **BMI Impact**: Being underweight may also affect insurance costs due to potential health concerns.")
            
            if age >= 50:
                st.info("👴 **Age Impact**: Insurance costs typically increase with age due to higher healthcare utilization.")
            
            # Additional insights
            st.markdown("---")
            st.subheader("📋 Additional Information")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Your Profile Summary:**")
                st.markdown(f"• **Age**: {age} years ({age_group.lower()} adult)")
                st.markdown(f"• **BMI**: {bmi:.1f} ({bmi_cat.lower()})")
                st.markdown(f"• **Smoking**: {'Yes' if smoker == 'yes' else 'No'}")
                st.markdown(f"• **Children**: {children}")
                st.markdown(f"• **Region**: {region.title()}")
            
            with col2:
                st.markdown("**Cost Factors:**")
                st.markdown(f"• **Health Risk Score**: {health_score:.1f}/10")
                st.markdown(f"• **Primary Risk**: {'Smoking' if smoker == 'yes' else 'Age/BMI related'}")
                
                # Cost range estimate
                if predicted_cost < 5000:
                    cost_range = "Low"
                elif predicted_cost < 15000:
                    cost_range = "Moderate"
                else:
                    cost_range = "High"
                st.markdown(f"• **Cost Category**: {cost_range}")
            
        except Exception as e:
            st.error(f"❌ Error making prediction: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
        <small>
        📋 This prediction is based on historical data and machine learning models. 
        Actual insurance costs may vary based on additional factors not included in this model.
        </small>
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()