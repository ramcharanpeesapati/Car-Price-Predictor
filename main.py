import tkinter as tk
from tkinter import messagebox
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

np.random.seed(42)
engine_size = np.random.uniform(1.0, 5.0, 1000)
mileage = np.random.uniform(5000, 100000, 1000)
age = np.random.randint(1, 20, 1000)
brand = np.random.choice(['Toyota', 'Ford', 'BMW', 'Audi', 'Mercedes'], 1000)

price = 5000 + (engine_size * 5000) - (mileage * 0.05) - (age * 200) + np.random.normal(0, 10000, 1000)

data = pd.DataFrame({
    'Engine Size': engine_size,
    'Mileage': mileage,
    'Age': age,
    'Brand': brand,
    'Price': price
})

X = data.drop('Price', axis=1)
y = data['Price']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['Engine Size', 'Mileage', 'Age']),
        ('cat', OneHotEncoder(drop='first'), ['Brand'])
    ])

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

model.fit(X, y)

def predict_price():
    try:
        engine_size_input = float(engine_size_entry.get())
        mileage_input = float(mileage_entry.get())
        age_input = int(age_entry.get())
        brand_input = brand_entry.get()

        user_data = pd.DataFrame({
            'Engine Size': [engine_size_input],
            'Mileage': [mileage_input],
            'Age': [age_input],
            'Brand': [brand_input]
        })

        predicted_price = model.predict(user_data)

        result_label.config(text=f"Predicted Car Price: ${predicted_price[0]:,.2f}")
    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numerical values.")

root = tk.Tk()
root.title("Car Price Prediction")

engine_size_label = tk.Label(root, text="Engine Size (liters):")
engine_size_label.grid(row=0, column=0, padx=10, pady=5)
engine_size_entry = tk.Entry(root)
engine_size_entry.grid(row=0, column=1, padx=10, pady=5)

mileage_label = tk.Label(root, text="Mileage (miles):")
mileage_label.grid(row=1, column=0, padx=10, pady=5)
mileage_entry = tk.Entry(root)
mileage_entry.grid(row=1, column=1, padx=10, pady=5)

age_label = tk.Label(root, text="Car Age (years):")
age_label.grid(row=2, column=0, padx=10, pady=5)
age_entry = tk.Entry(root)
age_entry.grid(row=2, column=1, padx=10, pady=5)

brand_label = tk.Label(root, text="Car Brand (Toyota, Ford, BMW, Audi, Mercedes):")
brand_label.grid(row=3, column=0, padx=10, pady=5)
brand_entry = tk.Entry(root)
brand_entry.grid(row=3, column=1, padx=10, pady=5)

predict_button = tk.Button(root, text="Predict Price", command=predict_price)
predict_button.grid(row=4, column=0, columnspan=2, pady=10)

result_label = tk.Label(root, text="Predicted Car Price: $0.00", font=("Helvetica", 14))
result_label.grid(row=5, column=0, columnspan=2, pady=10)

root.mainloop()
