# STEP 1- Import necessary libraries

import streamlit as st  # Web app framework
import pandas as pd  # Data manipulation and analysis
import pickle  # Serializing and deserializing Python objects
import joblib  # Alternative to pickle, better for large numpy arrays
import numpy as np  # Numerical operations
import skfuzzy as fuzz  # Fuzzy logic for decision-making
import skfuzzy.control as ctrl  # Control system for fuzzy logic
from sklearn.svm import SVC  # Support Vector Classifier (SVM)
import shap  # Explainability for machine learning models
import matplotlib.pyplot as plt  # Plotting and visualization
from alibi.explainers import AnchorTabular  # Model explainability tool
import os  # OS-related functions
import random  # Random number generation
from pulp import (  # Linear programming for optimization
    LpProblem, LpVariable, LpMaximize, lpSum, PULP_CBC_CMD, LpBinary
)
from PIL import Image  # Image processing
import google.generativeai as genai  # Google's Generative AI integration
from api_key import api_key  # Import API key from a separate config file
from io import BytesIO  # In-memory binary streams
import base64  # Encoding and decoding of binary data



# STEP 2 - Load Saved Models, Encoders, and Features

def load_label_encoders(): # Function to load label encoder form pickle files
    with open("label_encoders.pkl", "rb") as f:
        return pickle.load(f)

def load_one_hot_encoder(): # Function to load one_hot encoder form pickle files
    with open("one_hot_encoder.pkl", "rb") as f:
        ohe = pickle.load(f)
    with open("one_hot_encoded_column_names.pkl", "rb") as f:
        encoded_columns = pickle.load(f)
    return ohe, encoded_columns

def load_scaler(): # Function to load min-max scalar form pickle files
    with open("scaler_minmax.pkl", "rb") as f:
        return pickle.load(f)

def load_top_features(): # Function to load top features form pickle files
    with open("top_features.pkl", "rb") as f:
        top_features = pickle.load(f)
    return list(top_features['Feature'])

def load_svm_model(): # Function to load optimal classifier SVM form pickle files
    return joblib.load("best_svm_model.pkl")

def load_shap_values(): # Function to load SHAP values and explainer from pickle files
    with open('shap_values_class_1.pkl', 'rb') as f:
        shap_values_class_1 = pickle.load(f)
    with open('shap_values.pkl', 'rb') as f:
        shap_values = pickle.load(f)
    with open('svc_explainer.pkl', 'rb') as f:
        svc_explainer = pickle.load(f)
    return shap_values_class_1, shap_values, svc_explainer


# STEP 3 - Apply transformation and prediction

def apply_label_encoding(df, label_encoders, label_encode_cols): # ---------------- Apply Label Encoding ----------------
    for col in label_encode_cols:
        if col in df.columns and col in label_encoders:
            le = label_encoders[col]
            df[col] = le.transform(df[col])
    return df

def apply_one_hot_encoding(df, one_hot_encode_cols, ohe, encoded_columns): # ---------------- Apply One-Hot Encoding ----------------
    if not set(one_hot_encode_cols).issubset(df.columns):
        st.error("Some categorical columns are missing from the dataset.")
        return df
    df_ohe = ohe.transform(df[one_hot_encode_cols])
    df_ohe = pd.DataFrame(df_ohe, columns=encoded_columns, index=df.index)
    df.drop(columns=one_hot_encode_cols, inplace=True)
    return pd.concat([df, df_ohe], axis=1)

def apply_scaling(df, num_features, scaler): # ---------------- Apply Scaling ----------------
    existing_features = [col for col in num_features if col in df.columns]
    df[existing_features] = scaler.transform(df[existing_features])
    return df

def reverse_scaling(df, num_features, scaler): # ---------------- Reverse Scaling ----------------
    existing_features = [col for col in num_features if col in df.columns]
    df[existing_features] = scaler.inverse_transform(df[existing_features])
    return df

def predict_churn(df_selected, model): # ---------------- Predict Churn Probability ----------------
    if 'customer_id' in df_selected.columns:
        X = df_selected.drop(columns=['customer_id'])
    else:
        X = df_selected
    churn_probabilities = model.predict_proba(X)[:, 1]
    
    df_selected['Churn_Probability'] = churn_probabilities
    return df_selected

def classify_churn(df, threshold): # ---------------- Classify Churn Based on Threshold ----------------
    df['Churn_Status'] = df['Churn_Probability'].apply(lambda x: "Likely to Churn 🚩" if x >= threshold else "No Churn ✅")
    return df

def get_sample_data(df, customer_id): # Utility function to get sample data for a specific customer
    sample_data = df[df['customer_id'] == customer_id]
    if sample_data.empty:
        st.error(f"Customer ID {customer_id} not found in the DataFrame.")
        return None
    return sample_data.drop(columns=['customer_id'])




## STEP 4 - Integrate XAI functions 

def visualize_shap_waterfall(df, customer_id):
    shap_values_class_1, _, svc_explainer = load_shap_values()
    sample_data = get_sample_data(df, customer_id)
    if sample_data is None:
        return
    
    output_dir = f"Output_{customer_id}"
    os.makedirs(output_dir, exist_ok=True)
    
    customer_index = df.index[df['customer_id'] == customer_id].tolist()[0]
    
    if 'customer_id' in df.columns:
        df = df.drop(columns=['customer_id'], errors='ignore')

    st.subheader(f"SHAP Waterfall Plot for Customer {customer_id}")
    
    fig, ax = plt.subplots()
    shap.plots.waterfall(shap.Explanation(
        values=shap_values_class_1[customer_index],
        base_values=svc_explainer.expected_value[1],
        data=sample_data.iloc[0],
        feature_names=df.columns))
    
    st.pyplot(fig)

    image_path = os.path.join(output_dir, f"{customer_id}_waterfall.png")
    fig.savefig(image_path, bbox_inches='tight')
    st.success(f"SHAP waterfall plot saved successfully as {image_path}")
    
    if st.button("Key Insights", key=f"explain_waterfall_{customer_id}"):
        generate_analysis_from_image(image_path)
    
    if st.button("Clear Explanation", key=f"clear_waterfall_{customer_id}"):
        st.empty()

def anchor_analysis_for_customer(customer_id, model, df):
    # Ensure df is a DataFrame with feature names
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a DataFrame with customer_id column")

    # Define the predict function for the model
    def predict_fn(X):
        if isinstance(X, np.ndarray):  # Convert to DataFrame if it's a NumPy array
            X = pd.DataFrame(X, columns=df.columns.drop('customer_id'))
        return model.predict_proba(X)  # This returns the class probabilities

    # Method to get the class predictions from your model (for Anchor)
    def anchor_predict_fn(X):
        return np.argmax(predict_fn(X), axis=1)  # Get the class with the highest probability

    # Initialize the AnchorTabular explainer
    anchor_explainer = AnchorTabular(
        predictor=anchor_predict_fn,  # Set the function to get model predictions
        feature_names=df.columns.drop('customer_id').tolist()  # Feature names
    )

    # Fit the explainer on the data (excluding the customer_id column)
    anchor_explainer.fit(df.drop(columns=['customer_id']).values)

    # Select data for the specific customer
    sample_data = df[df['customer_id'] == customer_id]
    if sample_data.empty:
        print(f"Customer ID {customer_id} not found in the DataFrame.")
        return None
    
    sample_data = sample_data.drop(columns=['customer_id'])
    
    # Get the predicted probabilities for the sample
    original_probabilities = predict_fn(sample_data.values)
    
    # Get the anchor explanation for the sample
    anchor_exp = anchor_explainer.explain(sample_data.values.reshape(1, -1))
    
    # Extract features involved in the anchor rule and their conditions (if-then)
    anchor_conditions = anchor_exp.anchor
    
    # Prepare the output table
    print(f"\nAnchor Explanation for Customer ID {customer_id}:")
    print("Feature -> Condition (If-Then Rule)")

    anchor_table = []

    # Add rows for each condition
    for condition in anchor_conditions:
        feature, threshold = condition.split(" > ") if ">" in condition else condition.split(" <= ")
        feature = feature.strip()
        threshold = float(threshold.strip())
        anchor_table.append([feature, f"{condition}"])

    anchor_df = pd.DataFrame(anchor_table, columns=["Feature", "Condition"])
    st.dataframe(anchor_df)

    # Create the output directory named "Output_<customer_id>"
    output_dir = f"Output_{customer_id}"
    os.makedirs(output_dir, exist_ok=True)

    # Save the rules as a CSV file
    csv_file_path = os.path.join(output_dir, f"{customer_id}_anchor_rules.csv")
    anchor_df.to_csv(csv_file_path, index=False)
    st.success(f"Anchor rules saved to {csv_file_path}")
    
    # Save the rules as an image
    # Assuming `anchor_df` is your DataFrame and `output_dir` is the directory to save the image
    fig, ax = plt.subplots(figsize=(8, 4))  # Set figure size
    ax.xaxis.set_visible(False)  # Hide the x axis
    ax.yaxis.set_visible(False)  # Hide the y axis
    ax.set_frame_on(False)  # No visible frame

    # Create table with proper formatting
    tab = ax.table(cellText=anchor_df.values, colLabels=anchor_df.columns, loc='center', cellLoc='center', colColours=['lightgray'] * len(anchor_df.columns))

    # Adjust table font size and alignment
    tab.auto_set_font_size(False)  # Turn off auto font size
    tab.set_fontsize(12)  # Set a reasonable font size
    tab.scale(1.2, 1.2)  # Scale the table to make it more readable

    # Adjust column width to avoid truncating text
    for i, col in enumerate(anchor_df.columns):
        max_len = max(anchor_df[col].astype(str).apply(len).max(), len(col))  # Calculate max column width based on text length
        tab.auto_set_column_width([i])

    # Save the image with the table
    image_path = os.path.join(output_dir, f"{customer_id}_anchor.png")
    plt.savefig(image_path, bbox_inches='tight', dpi=300)  # Increase DPI for better quality
    st.success(f"Anchor rules table saved successfully as {image_path}")
    
    if st.button("Rule Interpretation and Potential Actions", key=f"explain_anchor_{customer_id}"):
        generate_recommendation_from_image(image_path)
    
    if st.button("Clear Explanation", key=f"clear_anchor_{customer_id}"):
        st.empty()




## STEP 5 - Integrate prescriptive analytics functions 

def normalize_column(column): # ---------------- Normalize Values for Clustering ----------------
    return (column - column.min()) / (column.max() - column.min())

def classify_customers(df): # ---------------- Apply Fuzzy Clustering ----------------
    # Normalize clustering features
    df['total_charges'] = normalize_column(df['total_charges'])
    df['monthly_charges'] = normalize_column(df['monthly_charges'])
    df['tenure'] = normalize_column(df['tenure'])

    # Define fuzzy logic system
    total_charges = ctrl.Antecedent(np.arange(0, 1.1, 0.01), 'total_charges')
    monthly_charges = ctrl.Antecedent(np.arange(0, 1.1, 0.01), 'monthly_charges')
    tenure = ctrl.Antecedent(np.arange(0, 1.1, 0.01), 'tenure')
    customer_type = ctrl.Consequent(np.arange(0, 1.1, 0.01), 'customer_type', defuzzify_method='centroid')

    # Define membership functions
    total_charges.automf(3)
    monthly_charges.automf(3)
    tenure.automf(3)
    customer_type.automf(3)

    # Define fuzzy rules
    rule1 = ctrl.Rule(total_charges['good'] & monthly_charges['good'] & tenure['good'], customer_type['good'])
    rule2 = ctrl.Rule(total_charges['average'] & monthly_charges['average'] & tenure['average'], customer_type['average'])
    rule3 = ctrl.Rule(total_charges['poor'] & monthly_charges['poor'] & tenure['poor'], customer_type['poor'])
    default_rule = ctrl.Rule(~(total_charges['good'] | total_charges['average'] | total_charges['poor']), customer_type['average'])

    # Create control system
    customer_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, default_rule])
    customer_simulation = ctrl.ControlSystemSimulation(customer_ctrl)

    # Apply fuzzy classification
    customer_types = []
    for _, row in df.iterrows():
        customer_simulation.input['total_charges'] = row['total_charges']
        customer_simulation.input['monthly_charges'] = row['monthly_charges']
        customer_simulation.input['tenure'] = row['tenure']
        customer_simulation.compute()

        output_value = customer_simulation.output['customer_type']
        if output_value < 0.3:
            customer_types.append('Low Engagement')
        elif output_value < 0.6:
            customer_types.append('Standard')
        else:
            customer_types.append('Premium')

    df['Customer Type'] = customer_types
    return df

def optimisation(filtered_df, budget, tot_dis, mon_dis, target_churn): # ---------------- Apply Optimisation with Liner Integer programming ----------------
    # Set fixed seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    N = len(filtered_df)  # Number of customers

    # Extract initial churn probabilities from the DataFrame
    initial_churn_prob = filtered_df['Churn_Probability'].values

    # **Sort Customers by Priority**
    priority_order = {'Premium': 1, 'Standard': 2, 'Low Engagement': 3}
    filtered_df['priority'] = filtered_df['Customer Type'].map(priority_order)
    filtered_df = filtered_df.sort_values(['priority', 'Churn_Probability'], ascending=[True, False])

    # **Intervention Cost Calculation and Adjustment**
    # Calculate the intervention cost as a weighted sum of total and monthly charges
    intervention_cost = (tot_dis * filtered_df["total_charges"] + 
                         mon_dis * filtered_df["monthly_charges"]).values

    # Reduce intervention costs by 30% if the minimum cost is greater than the budget
    if intervention_cost.min() > budget:
        intervention_cost *= 0.7

    # **Define Decision Variables**
    # x[i] = 1 if customer i is selected for intervention, 0 otherwise
    x = [LpVariable(f"x_{i}", cat="Integer", lowBound=0, upBound=1) for i in range(N)]
    
    # y[i] = 1 if customer i's churn is reduced below the target churn, 0 otherwise
    y = [LpVariable(f"y_{i}", cat="Binary") for i in range(N)]
    
    # churn_reduction_var[i] = amount by which customer i's churn probability is reduced
    churn_reduction_var = [LpVariable(f"churn_reduction_{i}", lowBound=0, cat="Continuous") for i in range(N)]

    # **Define the Optimization Problem**
    prob = LpProblem("Churn_Reduction_Optimization", LpMaximize)

    # **Objective Function**
    # Maximize the total number of customers with churn probability below the target churn
    prob += lpSum(y[i] for i in range(N))

    # **Constraints**
    prob += lpSum(intervention_cost[i] * x[i] for i in range(N)) <= budget

    for i in range(N):
        prob += churn_reduction_var[i] <= initial_churn_prob[i]

    # Ensure that a customer's churn probability is below the target churn if selected
    for i in range(N):
        prob += initial_churn_prob[i] - churn_reduction_var[i] <= target_churn + (1 - y[i]) * 1000

    for i in range(N):
        prob += x[i] <= y[i]

    for i in range(N):
        prob += y[i] <= x[i]

    # **Prioritize Premium Customers**
    for i in range(N):
        for j in range(i + 1, N):
            if filtered_df['priority'].iloc[i] < filtered_df['priority'].iloc[j]:
                prob += x[i] >= x[j]

    # **Solve the Optimization Problem**
    prob.solve(PULP_CBC_CMD(msg=False, warmStart=True))

    # **Extract Results**
    selected_customers = [i for i in range(N) if x[i].value() == 1]

    total_cost = sum(intervention_cost[i] for i in selected_customers)
    total_churn_after = sum((initial_churn_prob[i] - churn_reduction_var[i].value()) for i in selected_customers)

    new_churn_prob = [initial_churn_prob[i] - churn_reduction_var[i].value() for i in range(N)]

    # **Create Result DataFrame for Visualization**
    result_df = pd.DataFrame({
        "customer_id": filtered_df['customer_id'],
        "Initial_Churn_Prob": initial_churn_prob,
        "Intervention_Cost": intervention_cost,
        "Churn_Reduction": [churn_reduction_var[i].value() for i in range(N)],
        "Selected": [int(x[i].value()) for i in range(N)],
        "Churn_Below_Target": [int(y[i].value()) for i in range(N)],
        "New_Churn_Prob": new_churn_prob  # Add new churn probability
    })
    

    # Assuming result_df is already defined from your code
    # Create a new DataFrame where 'Selected' == 1
    selected_df = result_df[result_df['Selected'] == 1]

    # Display the total churn customers (count of unique customer_ids where Selected == 1)
    total_churn_customers = result_df['customer_id'].nunique()

    # Count total customers eligible for discount
    total_eligible_for_discount = selected_df['customer_id'].nunique()

    return result_df, selected_df, total_cost, total_churn_customers, total_eligible_for_discount

def calculate_intervention_for_customer(selected_df, filter_df, customer_id, tot_dis, mon_dis):
    # Check if customer_id is in selected_df
    if customer_id not in selected_df['customer_id'].values:
        return "This customer_id is not eligible for discount."

    # Filter filter_df for the specific customer_id
    customer_data = filter_df[filter_df['customer_id'] == customer_id]

    # Check if customer_data is empty
    if customer_data.empty:
        return "This customer_id is not eligible for discount."

    # Ensure the required columns exist in the filtered DataFrame
    if not {'total_charges', 'monthly_charges'}.issubset(customer_data.columns):
        raise ValueError("filter_df must contain 'total_charges' and 'monthly_charges' columns.")

    # Calculate intervention cost components
    total_charges_cost = tot_dis * customer_data["total_charges"].values[0]
    monthly_charges_cost = mon_dis * customer_data["monthly_charges"].values[0]

    # Calculate the total discount
    total_discount = total_charges_cost + monthly_charges_cost

    # Calculate adjusted charges
    total_charges_after_intervention = customer_data["total_charges"].values[0] - total_charges_cost
    monthly_charges_after_intervention = customer_data["monthly_charges"].values[0] - monthly_charges_cost

    # Create a results DataFrame
    result = pd.DataFrame({
        "Metric": [
            "Customer ID",
            "Total Discount",
            "Total Charge Before Discount",
            "Total Charge After Discount",
            "Monthly Charge Before Discount",
            "Monthly Charge After Discount"
        ],
        "Value": [
            customer_id,
            round(total_discount, 2),
            round(customer_data["total_charges"].values[0], 2),
            round(total_charges_after_intervention, 2),
            round(customer_data["monthly_charges"].values[0], 2),
            round(monthly_charges_after_intervention, 2)
        ]
    })

    return result, total_discount, customer_data["total_charges"].values[0], total_charges_after_intervention, customer_data["monthly_charges"].values[0], monthly_charges_after_intervention


## STEP 6 - Integrate Google Gemini for Image to text interpretation and recommendation

genai.configure(api_key=api_key)  # ------------------------------ Configure Gen AI---------------------

generation_config = {   #------------------ Adjusted generation configuration ----------------------
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50,
    "max_output_tokens": 2048,
    "response_mime_type": "text/plain",
}

safety_settings = [ #-------------------------- Apply safety settings---------------------------
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
]

model = genai.GenerativeModel(model_name="gemini-1.5-flash", #------------------------- Load Google Gemini Pro Vision API and get response
                              generation_config=generation_config,
                              safety_settings=safety_settings)

def generate_analysis_from_image(image_path): #------------Function for interpret image--------------------
    
    system_prompt = """
    As a highly skilled telecommunications expert, you specialize in Explainable AI (XAI) techniques such as SHAP waterfall plots to understand feature importance for individual customer predictions. Your expertise is essential in deriving key insights that can guide strategic decision-making.

    Primary Objectives
    Carefully analyze and interpret SHAP waterfall plots to understand the contribution of each feature in a particular customer's prediction.

    Identify key drivers influencing the prediction and provide insightful observations based on the SHAP values.
    
    Use below details to interpret: 
    A waterfall plot provides a detailed breakdown of how each input variable contributes towards the classified as likely to churn for a single instance of the data.
    SHAP Values Explanation:
        The SHAP values quantify the amount and direction in which each variable impacts the predicted likely to churn.
        
        SHAP values inside red arrows and blue arrows.
        Red arrows indicate features that contribute negatively to the prediction. This means that these features push the prediction towards the negative class (e.g., class 0).
        Blue arrows indicate features that contribute positively to the prediction. This means that these features push the prediction towards the positive class (e.g., class 1).
                
        
        f(X) is the log-odds output by the model, E(f(x))-The base value is the average log-odds of the target class across the dataset.
        The final log-odds output, f(x), is equal to the base log-odds value plus the sum of all the SHAP values.
        The final log-odds output is converted into a probability using the Gaussian function (or a radial basis function).
        Input variables, ranked from top to bottom by how much impact they have on the model’s prediction for this example from the data.
        The grey numbers denote the values of the variables for this particular instance. Input values for tenure, total-charges,and monthly-charges are normalized, not a real value
        Input features dependents, tenure are non actionable features
        
        Below are real values: 
        tenure - minimum real value is 0 month,  maximum real value is 0 month 72 months
        total_charges - minimum real value is $18.8, maximum real value is  $8684.8
        monthly_charges - minimum real value is $18.25, maximum real value is  $118.75
        internet_service_No <= 0.00 mean it has DSL connection


    Important Considerations
    ✅ Ensure interpretations are data-driven, objective, and transparent.
    ✅ Focus on how features increase or decrease the predicted outcome.
    ✅ Avoid assumptions—base insights strictly on the given SHAP plot.

    Disclaimer
    "This analysis is intended as a preliminary review. Please consult with your data science and business operations teams before making strategic or business decisions. Further analysis may be required to validate these findings."
    """
    
    with open(image_path, "rb") as file:
        uploaded_image = file.read()
    
    # Process image files
    img = Image.open(image_path)
    
    # Convert the image to RGB mode if it has an alpha channel (RGBA)
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    
    # Resize to 512x512
    # img = img.resize((512, 512))
    
    # Save the image into a buffer
    buffer = BytesIO()
    img.save(buffer, format="png")
    image_data = buffer.getvalue()

    # Encode image data in base64
    image_base64 = base64.b64encode(image_data).decode('utf-8')

    # Display the resized image in the app
    # st.image(img, caption="Uploaded Image (Resized)", use_column_width=True)

    image_parts = [
        {
            "mime_type": "image/png",  # Assuming JPEG format
            "data": image_base64
        }
    ]

    prompt_parts = [
        image_parts[0],
        system_prompt,
    ]

    # Generate a response based on the prompt image
    try:
        response = model.generate_content(prompt_parts)
        st.write(response.text)
    except Exception as e:
        st.error(f"An error occurred: {e}")

def generate_recommendation_from_image(image_path): # Function for generate recommendation from image ---------
    
    system_prompt = """
    As a highly skilled telecommunications expert specializing in Explainable AI (XAI), your role is to interpret Anchors (If-Then) rules for improving customer retention. Your expertise is critical in analyzing customer-specific anchors, plots, tables, or images to derive meaningful insights.

    Primary Objectives
    Carefully analyze the given customer's if-then anchor rules from plots, tables, or images.

    Sample interpretation:
    Feature -> Condition (If-Then Rule) : The explanation is given in the form of if-then rules based on the features and their conditions.
        monthly_charges > 0.71 → The customer has relatively high monthly charges.
        tenure ≤ 0.39 → The customer has been with the service provider for a short period of time.
        internet_service_Fiber optic > 0.00 → The customer is subscribed to a Fiber Optic internet plan.
        payment_method_Electronic check > 0.00 → The customer is using Electronic Check as their payment method.
        tech_support ≤ 0.00 → The customer does not have technical support services included in their plan.
    
        Input values for tenure, total-charges,and monthly-charges are normalized numerical value between 0 to 1 , not a real value
        Other input values 1 mean Yes , 0 mean No
        
        Below are real values: 
        tenure - minimum real value is 0 month,  maximum real value is 0 month 72 months
        total_charges - minimum real value is $18.8, maximum real value is  $8684.8
        monthly_charges - minimum real value is $18.25, maximum real value is  $118.75
        internet_service_No <= 0.00 mean it has DSL connection

    
    Provide only practical, data-driven, and actionable recommendations that can effectively improve customer retention.

    Important Considerations
    ✅ First interpret each rules like above example, Secondly recommend 
    ✅ Recommendations must be ethical, legal, and aligned with business best practices.
    ✅ Focus on interpretable and transparent AI-driven insights.
    ✅ Ensure strategies are feasible and relevant to the customer's situation.
    ✅ Input features dependents, tenure are non actionable features, so values can not change consider during recommendation 


    Disclaimer
    "This analysis is intended as a preliminary review. Please consult with your data science and business operations teams before making strategic or business decisions. Further analysis may be required to validate these findings."
    """
    
    with open(image_path, "rb") as file:
        uploaded_image = file.read()
    
    # Process image files
    img = Image.open(image_path)
    
    # Convert the image to RGB mode if it has an alpha channel (RGBA)
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    
    # Resize to 512x512
    # img = img.resize((512, 512))
    
    # Save the image into a buffer
    buffer = BytesIO()
    img.save(buffer, format="png")
    image_data = buffer.getvalue()

    # Encode image data in base64
    image_base64 = base64.b64encode(image_data).decode('utf-8')

    # Display the resized image in the app
    # st.image(img, caption="Uploaded Image (Resized)", use_column_width=True)

    image_parts = [
        {
            "mime_type": "image/png",  # Assuming JPEG format
            "data": image_base64
        }
    ]

    prompt_parts = [
        image_parts[0],
        system_prompt,
    ]

    # Generate a response based on the prompt image
    try:
        response = model.generate_content(prompt_parts)
        st.write(response.text)
    except Exception as e:
        st.error(f"An error occurred: {e}")




## STEP 7 - Develop Streamlit App

def main(): # ---------------- Main Streamlit App ----------------
    st.markdown("<h1 style='margin-bottom: 0;'> RetenNet</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='margin-top: 0;'>Stronger Through Retention.</h3>", unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: center;'>Welcome to the RetenNet, This app helps you analyze and interpret customer churn predictions and provide actionable retention recommendations.</h5>", unsafe_allow_html=True)
    
    
    uploaded_file = st.file_uploader("Upload your data (CSV or Excel)", type=["csv", "xlsx"])

    # Ensure threshold persists across tabs
    if "threshold" not in st.session_state:
        st.session_state.threshold = 0.5  # Default threshold value

    st.session_state.threshold = st.slider("Select Churn Probability Threshold", 0.0, 1.0, st.session_state.threshold)

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('csv') else pd.read_excel(uploaded_file)
        if 'customer_id' not in df.columns:
            st.error("Missing 'customer_id' column in dataset. Please upload a valid file.")
            return

        # Load encoders, scalers, features & model
        label_encoders = load_label_encoders()
        ohe, encoded_columns = load_one_hot_encoder()
        scaler = load_scaler()
        top_features = load_top_features()
        model = load_svm_model()

        if st.button("Submit"):
            customer_id_col = df[['customer_id']].copy()
            df = df.drop(columns=['customer_id'])

            # Apply Label Encoding
            df = apply_label_encoding(df, label_encoders, [
                'senior_citizen', 'partner', 'dependents',
                'phone_service', 'paperless_billing', 'multiple_lines', 'online_security',
                'online_backup', 'device_protection', 'tech_support',
                'streaming_tv', 'streaming_movies'
            ])

            # Apply One-Hot Encoding
            df = apply_one_hot_encoding(df, ['contract', 'payment_method', 'internet_service'], ohe, encoded_columns)

            # Apply Scaling
            df = apply_scaling(df, ['tenure', 'total_charges', 'monthly_charges'], scaler)

            # Ensure top features are selected AFTER transformations
            df = df[top_features]  # Keep only top 15 features

            # Reattach customer_id
            df = pd.concat([customer_id_col, df], axis=1)

            # **Predict Churn Before Classifying**
            df = predict_churn(df, model)

            # Apply Churn Classification
            df = classify_churn(df, st.session_state.threshold)

            # Classify Customers based on Fuzzy Clustering
            df = classify_customers(df)

            st.session_state['processed_df'] = df  # Save processed dataframe for other tabs


    if 'processed_df' in st.session_state:
        df = st.session_state['processed_df']

        # 🟢 **Download Button for Processed Data**
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Predicted Data",
            data=csv_data,
            file_name="predicted_churn_data.csv",
            mime="text/csv"
        )

        # Default: No customer selected
        selected_customer_id = st.selectbox("Select Customer ID", ["Select Customer"] + df['customer_id'].tolist(), index=0)

        if selected_customer_id != "Select Customer":
            # Fetch churn probability and status for the selected customer
            customer_data = df[df['customer_id'] == selected_customer_id]
            customer_status = customer_data['Churn_Status'].values[0]
            Customer_Type = customer_data['Customer Type'].values[0]
            churn_probability = customer_data['Churn_Probability'].values[0]

            # Display customer information
            st.markdown(f"**Customer {selected_customer_id}**")
            st.markdown(f"**Customer is {Customer_Type} Customer**")
            st.markdown(f"**Churn Probability: {churn_probability:.2f}**")

            if customer_status == "Likely to Churn 🚩":
                st.markdown(f"**Status: {customer_status}**")
                
                tabs = st.tabs(["Discount Eligibility", "Feature Importance analysis with SHAP", "Anchor Explanation and Recommendation"])
                
                with tabs[0]:
                    discount_df = df[df['Churn_Status'] == "Likely to Churn 🚩"]
                    discount_df = reverse_scaling(discount_df, ['tenure', 'total_charges', 'monthly_charges'], load_scaler())
                    #st.write("Customers Eligible for Discount (Reversed Scaling Applied):")
                    #st.dataframe(discount_df[['customer_id', 'Churn_Probability', 'Customer Type'] + [col for col in discount_df.columns if col not in ['customer_id', 'Churn_Probability', 'Customer Type']]].style.hide(axis="index"))
                
                    # User inputs
                    st.write('Enter the values to determine customers eligibility for a Discount')
                    budget = st.number_input('Enter the budget:', min_value=0, value=20000)
                    tot_dis = st.number_input('Enter the total discount (as a decimal):', min_value=0.0, max_value=1.0, value=0.10)
                    mon_dis = st.number_input('Enter the monthly discount (as a decimal):', min_value=0.0, max_value=1.0, value=0.10)
                    target_churn = st.number_input('Enter the target churn rate (as a decimal):', min_value=0.0, max_value=1.0, value=0.30)

                    # Apply Optimization
                    if st.button('Check Discount Eligibility'):
                        result_df, selected_df, total_cost, total_churn_customers, total_eligible_for_discount = optimisation(discount_df, budget=budget, tot_dis=tot_dis, mon_dis=mon_dis, target_churn=target_churn)
                        #st.dataframe(result_df)
                        #st.dataframe(selected_df)
                        st.write("### Total Discount Eligibility Summary")
                        st.write(f"Total churn customers:{total_churn_customers}")
                        st.write(f"Total eligible customers for discount: {total_eligible_for_discount}")
                        st.write(f"Total cost: ${total_cost:.2f}")
                        
                        result = calculate_intervention_for_customer(filter_df=discount_df, selected_df=selected_df, customer_id=selected_customer_id, tot_dis=tot_dis, mon_dis=mon_dis)

                        if isinstance(result, str):
                            st.markdown(f"<span style='color:red'>{result}</span>", unsafe_allow_html=True)
                        else:
                            result, total_discount, total_charges_before, total_charges_after, monthly_charges_before, monthly_charges_after = result
                            st.write("### Customer Discount Summary")
                            st.markdown("<span style='color:green'>Eligible for discount</span>", unsafe_allow_html=True)
                            st.write(f"Total discount: {total_discount:.2f}")
                            st.write(f"Total charges before discount: {total_charges_before:.2f}")
                            st.write(f"Total charges after discount applied: {total_charges_after:.2f}")
                            st.write(f"Monthly charges before discount: {monthly_charges_before:.2f}")
                            st.write(f"Monthly charges after discount applied: {monthly_charges_after:.2f}")
                        
                # Modify the Feature Analysis Tab in the Streamlit App
                with tabs[1]:
                    feature_df = df.drop(columns=['Churn_Probability', 'Churn_Status', 'Customer Type'], errors='ignore')

                    # visualize_shap_waterfall(df, customer_id)
                    visualize_shap_waterfall(feature_df, selected_customer_id)
                    
                with tabs[2]:
                    feature_df = df.drop(columns=['Churn_Probability', 'Churn_Status', 'Customer Type'], errors='ignore')
                    
                    # Load model again for SHAP analysis
                    svm_model = load_svm_model()
                    
                    # Anchor analysis
                    anchor_analysis_for_customer(selected_customer_id, svm_model, feature_df)
            else:
                st.markdown(f"**Status: {customer_status}**")
                st.write("No further actions required.")

if __name__ == "__main__":
    main()