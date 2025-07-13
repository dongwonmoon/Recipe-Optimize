import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import pickle

# Set matplotlib to use a default font that should be available
plt.rcParams['font.family'] = 'sans-serif'


# --- Load Data and Models ---
@st.cache_data
def load_data():
    try:
        model = joblib.load('et_regressor_model.joblib')
        with open('X_test.pkl', 'rb') as f:
            x_test = pickle.load(f)
        explainer = shap.TreeExplainer(model)
        best_recipes = pd.read_csv('best_recipes.csv')
        best_recipes_fitness = pd.read_csv('best_recipes_fitness.csv')
        return model, x_test, explainer, best_recipes, best_recipes_fitness
    except FileNotFoundError as e:
        st.error(f"Error loading necessary files: {e}. Please run the simulation script first.")
        return None, None, None, None, None

model, X_test, explainer, best_recipes, best_recipes_fitness = load_data()

if model is None:
    st.stop()

# --- Page Configuration ---
st.set_page_config(layout="wide")
st.title('🍷 Wine Recipe Optimizer Dashboard')

# --- Section 1: Optimization Results ---
st.header('1. Multi-Objective Optimization Results (Pareto Front)')
st.write("""
The genetic algorithm found optimal recipes by balancing two conflicting objectives:
1.  **Maximize Predicted Quality Score** (higher is better).
2.  **Minimize Mahalanobis Distance** (lower is better, indicating a more 'realistic' recipe).

Each point on the scatter plot below represents an optimal recipe on the Pareto front.
""")

fig, ax = plt.subplots()
ax.scatter(best_recipes_fitness['distance'], best_recipes_fitness['quality'], alpha=0.7, c='royalblue')
ax.set_xlabel('Realism (Lower Distance is Better)')
ax.set_ylabel('Predicted Quality Score (Higher is Better)')
ax.set_title('Optimal Recipes: Quality vs. Realism')
ax.grid(True)
st.pyplot(fig)

# --- Section 2: Recipe Explorer and Simulator ---
st.header('2. Recipe Explorer and Simulator')

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader('👨‍🔬 Select an Optimal Recipe')
    selected_index = st.selectbox(
        'Select a recipe from the Pareto front:',
        best_recipes.index,
        format_func=lambda x: f"Recipe {x} (Quality: {best_recipes_fitness.loc[x, 'quality']:.2f}, Distance: {best_recipes_fitness.loc[x, 'distance']:.2f})"
    )
    
    selected_recipe = best_recipes.loc[selected_index]
    st.write("**Selected Recipe Details:**")
    st.dataframe(selected_recipe)

with col2:
    st.subheader('🔬 Real-time Recipe Simulator')
    st.write("Adjust the sliders to create your own recipe and see the predicted quality.")
    
    current_recipe = {}
    for col in X_test.columns:
        min_val = float(X_test[col].min())
        max_val = float(X_test[col].max())
        default_val = float(selected_recipe[col])
        # Ensure the default value is within the slider's range
        default_val = np.clip(default_val, min_val, max_val)
        current_recipe[col] = st.slider(f'{col}', min_val, max_val, default_val, step=0.01)

# --- Section 3: Prediction and Analysis ---
recipe_df = pd.DataFrame([current_recipe])
predicted_quality = model.predict(recipe_df)[0]
final_score = np.clip(predicted_quality, 0, 10)

st.markdown(f"### 🧪 Predicted Quality Score: **{final_score:.4f}**")

st.subheader('📈 Recipe Contribution Analysis (SHAP Waterfall Plot)')
st.write("This plot shows how each ingredient's value contributes to the final quality score prediction.")

# Generate SHAP values for the current recipe
shap_values = explainer(recipe_df)

# Create and display the waterfall plot
fig_shap, ax_shap = plt.subplots()
shap.plots.waterfall(shap_values[0], max_display=14, show=False)
st.pyplot(fig_shap)