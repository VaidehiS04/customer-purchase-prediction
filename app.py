import streamlit as st
import pandas as pd
import pickle

with open("purchase_prediction_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("model_columns.pkl", "rb") as f:
    model_columns = pickle.load(f)

st.title("Customer Purchase Prediction App")
st.write("Fill in the customer details below and click Predict to see "
         "whether the customer is likely to purchase a product.")


age = st.slider("Age", min_value=18, max_value=70, value=30)

gender = st.selectbox("Gender", ["Male", "Female"])

annual_income = st.number_input("Annual Income ($)", min_value=0,
                            max_value=200000, value=50000, step=1000)

number_of_purchases = st.number_input("Number of Previous Purchases",
                                       min_value=0, max_value=50, value=5)

product_category = st.selectbox(
    "Product Category",
    ["Electronics", "Clothing", "Home Goods", "Beauty", "Sports"]
)

time_spent_on_website = st.slider("Time Spent on Website (minutes)",
                                   min_value=0, max_value=120, value=15)

loyalty_program = st.selectbox("Loyalty Program Member?", ["Yes", "No"])

discounts_availed = st.slider("Number of Discounts Availed",
                               min_value=0, max_value=5, value=1)


gender_code = 0 if gender == "Male" else 1

category_map = {
    "Electronics": 0,
    "Clothing": 1,
    "Home Goods": 2,
    "Beauty": 3,
    "Sports": 4,
}
product_category_code = category_map[product_category]

loyalty_code = 1 if loyalty_program == "Yes" else 0

if st.button("Predict"):

    
    input_data = pd.DataFrame([{
        "Age": age,
        "Gender": gender_code,
        "AnnualIncome": annual_income,
        "NumberOfPurchases": number_of_purchases,
        "ProductCategory": product_category_code,
        "TimeSpentOnWebsite": time_spent_on_website,
        "LoyaltyProgram": loyalty_code,
        "DiscountsAvailed": discounts_availed,
    }])

    
    input_data = input_data[model_columns]

    
    input_scaled = scaler.transform(input_data)

    
    prediction = model.predict(input_scaled)[0]
    prediction_proba = model.predict_proba(input_scaled)[0]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.success("This customer is LIKELY to purchase the product.")
        confidence = prediction_proba[1] * 100
    else:
        st.error("This customer is NOT likely to purchase the product.")
        confidence = prediction_proba[0] * 100

    st.write(f"Confidence: {confidence:.2f}%")