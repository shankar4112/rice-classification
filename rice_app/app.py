from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load models and scaler
nb_model = joblib.load('naive_bayes_model.pkl')
knn_model = joblib.load('knn_model.pkl')
lr_model = joblib.load('logistic_regression_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    predictions = None
    if request.method == 'POST':
        try:
            area = float(request.form['area'])
            perimeter = float(request.form['perimeter'])
            major_axis = float(request.form['major_axis'])
            minor_axis = float(request.form['minor_axis'])
            eccentricity = float(request.form['eccentricity'])
            convex_area = float(request.form['convex_area'])
            extent = float(request.form['extent'])

            input_features = np.array([[area, perimeter, major_axis, minor_axis, eccentricity, convex_area, extent]])
            scaled_input = scaler.transform(input_features)

            predictions = {
                'nb': nb_model.predict(input_features)[0],
                'knn': knn_model.predict(scaled_input)[0],
                'lr': lr_model.predict(scaled_input)[0]
            }

        except Exception as e:
            predictions = {'error': str(e)}

    return render_template('index.html', predictions=predictions)

if __name__ == '__main__':
    app.run(debug=True)

# import streamlit as st
# import joblib
# import numpy as np

# # Load models and scaler
# nb_model = joblib.load('naive_bayes_model.pkl')
# knn_model = joblib.load('knn_model.pkl')
# lr_model = joblib.load('logistic_regression_model.pkl')
# scaler = joblib.load('scaler.pkl')

# st.title("🍚 Rice Classification App")
# st.markdown("Predict whether the rice grain is **Cammeo** or **Osmancik**")

# # Input form
# with st.form("input_form"):
#     area = st.number_input("Area", min_value=0.0)
#     perimeter = st.number_input("Perimeter", min_value=0.0)
#     major_axis = st.number_input("Major Axis Length", min_value=0.0)
#     minor_axis = st.number_input("Minor Axis Length", min_value=0.0)
#     eccentricity = st.number_input("Eccentricity", min_value=0.0)
#     convex_area = st.number_input("Convex Area", min_value=0.0)
#     extent = st.number_input("Extent", min_value=0.0)

#     submit = st.form_submit_button("Predict")

# if submit:
#     input_features = np.array([[area, perimeter, major_axis, minor_axis, eccentricity, convex_area, extent]])
#     scaled_input = scaler.transform(input_features)

#     # Predictions
#     nb_pred = nb_model.predict(input_features)[0]
#     knn_pred = knn_model.predict(scaled_input)[0]
#     lr_pred = lr_model.predict(scaled_input)[0]

#     st.subheader("📊 Prediction Results")
#     st.write(f"**Naive Bayes Prediction:** {nb_pred}")
#     st.write(f"**KNN Prediction:** {knn_pred}")
#     st.write(f"**Logistic Regression Prediction:** {lr_pred}")
