import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)

n_customers = 2000
n_orders = 25000

start_date = datetime(2024, 1, 1)
end_date = datetime(2025, 12, 31)
date_range = (end_date - start_date).days

cities = [
    ("Mumbai", "Maharashtra", "West", 19.0760, 72.8777),
    ("Delhi", "Delhi", "North", 28.7041, 77.1025),
    ("Bangalore", "Karnataka", "South", 12.9716, 77.5946),
    ("Hyderabad", "Telangana", "South", 17.3850, 78.4867),
    ("Chennai", "Tamil Nadu", "South", 13.0827, 80.2707),
    ("Kolkata", "West Bengal", "East", 22.5726, 88.3639),
    ("Pune", "Maharashtra", "West", 18.5204, 73.8567),
    ("Ahmedabad", "Gujarat", "West", 23.0225, 72.5714),
    ("Jaipur", "Rajasthan", "North", 26.9124, 75.7873),
    ("Lucknow", "Uttar Pradesh", "North", 26.8467, 80.9462),
    ("Bhopal", "Madhya Pradesh", "Central", 23.2599, 77.4126),
    ("Indore", "Madhya Pradesh", "Central", 22.7196, 75.8577),
    ("Patna", "Bihar", "East", 25.5941, 85.1376),
    ("Nagpur", "Maharashtra", "West", 21.1458, 79.0882),
    ("Surat", "Gujarat", "West", 21.1702, 72.8311),
]

categories = {
    "Electronics": ["Smartphone", "Laptop", "Headphones", "Smartwatch", "Tablet", "Bluetooth Speaker", "Power Bank", "USB Cable"],
    "Clothing": ["T-Shirt", "Jeans", "Jacket", "Dress", "Shoes", "Sweater", "Shorts", "Socks"],
    "Home & Kitchen": ["Cookware Set", "Bedsheet", "Curtains", "Vacuum Cleaner", "Blender", "Toaster", "Lamp", "Storage Box"],
    "Books": ["Fiction Novel", "Self-Help", "Technical Guide", "Biography", "Comic Book", "Cookbook", "Travel Guide", "Children Book"],
    "Sports": ["Yoga Mat", "Dumbbell Set", "Running Shoes", "Cricket Bat", "Tennis Racket", "Cycling Helmet", "Skipping Rope", "Gym Bag"],
}

price_ranges = {
    "Electronics": (299, 89999),
    "Clothing": (199, 4999),
    "Home & Kitchen": (149, 14999),
    "Books": (99, 1499),
    "Sports": (149, 7999),
}

names_pool = ["Rahul", "Priya", "Amit", "Sneha", "Vikram", "Ananya", "Raj", "Deepika", "Arjun", "Kavita", "Suresh", "Meena", "Nikhil", "Pooja", "Rohan", "Swati", "Manish", "Neha", "Karan", "Divya"]
surname_pool = ["Sharma", "Patel", "Kumar", "Singh", "Gupta", "Verma", "Reddy", "Nair", "Joshi", "Das", "Mehta", "Rao", "Menon", "Chopra", "Kapoor", "Malhotra"]

customer_names = [f"{np.random.choice(names_pool)} {np.random.choice(surname_pool)}" for _ in range(n_customers)]

customers = pd.DataFrame({
    "customer_id": [f"CUST-{i:04d}" for i in range(1, n_customers + 1)],
    "customer_name": customer_names,
    "email": [f"customer{i}@email.com" for i in range(1, n_customers + 1)],
})

city_probs = [0.18, 0.14, 0.12, 0.09, 0.08, 0.07, 0.06, 0.05, 0.05, 0.04, 0.03, 0.03, 0.02, 0.02, 0.02]
city_choices = np.random.choice(len(cities), size=n_customers, p=city_probs)

customers["city"] = [cities[i][0] for i in city_choices]
customers["state"] = [cities[i][1] for i in city_choices]
customers["region"] = [cities[i][2] for i in city_choices]
customers["lat"] = [cities[i][3] for i in city_choices]
customers["lon"] = [cities[i][4] for i in city_choices]

acq_dates = sorted([start_date + timedelta(days=np.random.randint(0, date_range)) for _ in range(n_customers)])
customers["acquisition_date"] = acq_dates

order_dates = sorted([start_date + timedelta(days=np.random.randint(0, date_range)) for _ in range(n_orders)])
customer_ids = np.random.choice(customers["customer_id"].values, size=n_orders)
cat_choices = np.random.choice(list(categories.keys()), size=n_orders, p=[0.25, 0.30, 0.18, 0.12, 0.15])

products_list = []
prices_list = []
for cat in cat_choices:
    product = np.random.choice(categories[cat])
    pr = np.random.randint(price_ranges[cat][0], price_ranges[cat][1] + 1)
    price = round(pr * np.random.uniform(0.85, 1.15), 2)
    products_list.append(product)
    prices_list.append(price)

orders = pd.DataFrame({
    "order_id": [f"ORD-{i:06d}" for i in range(1, n_orders + 1)],
    "order_date": order_dates,
    "customer_id": customer_ids,
    "category": cat_choices,
    "product": products_list,
    "unit_price": prices_list,
})

orders["quantity"] = np.random.choice([1, 2, 3, 4, 5, 6], size=n_orders, p=[0.15, 0.40, 0.25, 0.12, 0.05, 0.03])
orders["discount_pct"] = np.where(np.random.random(n_orders) < 0.35, np.random.choice([5, 10, 15, 20, 25], size=n_orders), 0).astype(int)
orders["total_amount"] = round(orders["unit_price"] * orders["quantity"] * (1 - orders["discount_pct"] / 100), 2)
orders["payment_method"] = np.random.choice(["Credit Card", "UPI", "Net Banking", "Cash on Delivery", "Wallet"], size=n_orders, p=[0.25, 0.35, 0.15, 0.15, 0.10])
orders["status"] = np.random.choice(["Delivered", "Delivered", "Delivered", "Delivered", "Cancelled", "Returned"], size=n_orders, p=[0.72, 0.05, 0.05, 0.03, 0.08, 0.07])
orders["year"] = orders["order_date"].dt.year
orders["month"] = orders["order_date"].dt.month
orders["month_name"] = orders["order_date"].dt.strftime("%B")
orders["quarter"] = orders["order_date"].dt.quarter
orders["day_of_week"] = orders["order_date"].dt.day_name()
orders["is_weekend"] = orders["order_date"].dt.dayofweek >= 5

customers.to_csv("/workspace/ecommerce-analytics/data/customers.csv", index=False)
orders.to_csv("/workspace/ecommerce-analytics/data/orders.csv", index=False)

print(f"Customers: {len(customers)}  |  Orders: {len(orders)}")
print(f"Total Revenue: INR {orders['total_amount'].sum():,.0f}")
