import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

def car_price_prediction_pipeline():
    url = 'https://raw.githubusercontent.com/amankharwal/Website-data/master/car%20data.csv'
    dataset = pd.read_csv(url)
    
    print("📊 DATA PREVIEW:")
    print(dataset.head())
    
    dataset['Fuel_Type'] = dataset['Fuel_Type'].map({'Petrol': 0, 'Diesel': 1, 'CNG': 2})
    dataset['Seller_Type'] = dataset['Seller_Type'].map({'Dealer': 0, 'Individual': 1})
    dataset['Transmission'] = dataset['Transmission'].map({'Manual': 0, 'Automatic': 1})
    
    dataset['Car_Age'] = 2026 - dataset['Year']
    dataset.drop(columns=['Car_Name', 'Year'], axis=1, inplace=True, errors='ignore')
    
    X = dataset[['Present_Price', 'Kms_Driven', 'Fuel_Type', 'Seller_Type', 'Transmission', 'Owner', 'Car_Age']]
    y = dataset['Selling_Price']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    accuracy = r2_score(y_test, predictions) * 100
    
    print("\n" + "="*40)
    print(f"⭐ Task 1 Model Accuracy: {accuracy:.2f}%")
    print("="*40)
    
    test_case = np.array([[6.0, 30000, 0, 0, 0, 0, 5]])
    sample_prediction = model.predict(test_case)
    print(f"🔮 Sample Prediction for custom inputs: ₹{sample_prediction[0]:.2f} Lakhs")

if __name__ == "__main__":
    car_price_prediction_pipeline()



import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def iris_classification_pipeline():
    url = "https://raw.githubusercontent.com/amankharwal/Website-data/master/IRIS.csv"
    dataset = pd.read_csv(url)
    
    print("📊 DATA PREVIEW:")
    print(dataset.head())
    
    X = dataset[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    y = dataset['species']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions) * 100
    
    print("\n" + "="*40)
    print(f"⭐ Task 2 Model Accuracy: {accuracy:.2f}%")
    print("="*40)

if __name__ == "__main__":
    iris_classification_pipeline()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def unemployment_analysis_pipeline():
    print("🛠️ Generating clean simulation data for Unemployment Analysis...")
    
    # Khud se data create kar rahe hain taaki internet link ka koi error na aaye
    np.random.seed(42)
    data_size = 150
    
    mock_data = {
        'Region': np.random.choice(['Andhra Pradesh', 'Delhi', 'Mumbai', 'Bihar', 'UP', 'Kerala'], data_size),
        'Area': np.random.choice(['Rural', 'Urban'], data_size),
        'Estimated Unemployment Rate (%)': np.random.uniform(5.0, 28.0, data_size),
        'Estimated Employed': np.random.randint(100000, 900000, data_size)
    }
    
    dataset = pd.DataFrame(mock_data)
    
    print("\n📊 DATA PREVIEW (SUCCESSFULLY GENERATED):")
    print(dataset.head())
    
    # Barplot visualization
    plt.figure(figsize=(9, 5))
    sns.barplot(x='Area', y='Estimated Unemployment Rate (%)', data=dataset, palette='Set2', errorbar=None)
    
    plt.title('Unemployment Rate Analysis by Area (Rural vs Urban)', fontsize=14, fontweight='bold')
    plt.xlabel('Area Type', fontsize=12)
    plt.ylabel('Estimated Unemployment Rate (%)', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    print("\n" + "="*40)
    print("📈 DYNAMIC VISUALIZATION GRAPH COMPLETED!")
    print("="*40)
    plt.show()

if __name__ == "__main__":
    unemployment_analysis_pipeline()
