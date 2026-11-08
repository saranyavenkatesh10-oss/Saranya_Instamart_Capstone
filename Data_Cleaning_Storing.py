import sqlite3
import pandas as pd
import numpy as np
import os

conn=sqlite3.connect("data/instamart.db")

ticket_df=pd.read_csv("data/raw/tickets.csv",keep_default_na=False)

customers_df=pd.read_csv("data/raw/customers.csv",keep_default_na=False)
products_df=pd.read_csv("data/raw/products.csv",keep_default_na=False)

# Load the tickets.csv into the dataframe
  
order_df=pd.read_csv("data/raw/orders.csv",keep_default_na=False)
    
ticket_df.to_sql("tickets",conn,index=False,if_exists="replace")

   
order_duplicates=order_df.duplicated()
order_duplicate_count=order_duplicates.sum()


orders_pending=order_df[(order_df['status'].isin(["pending"]))].shape[0]

distinct_order_id=len(pd.unique(order_df['order_id']))


non_numeric= pd.to_numeric(order_df['order_amount'], errors='coerce').isna().sum()



    # No of Negative delivery mionutes

negative_delivery = (pd.to_numeric(order_df['delivery_minutes']) < 0).sum()



blank_delivery_minutes=(order_df['delivery_minutes'].isnull() + (order_df['delivery_minutes'].astype(str).str.strip() == '')).sum()

blank_age=(customers_df['age'].isnull() + (customers_df['age'].astype(str).str.strip() == '')).sum()



    #(pd.to_numeric(customers_df['age']) < 10  pd.to_numeric(customers_df['age']) >100 ).count())

age_numeric = pd.to_numeric(customers_df['age'], errors='coerce')
out_of_range_age = ((age_numeric < 10) | (age_numeric > 100)).sum()



raw_city_spellings = customers_df['city'].nunique()


unique_gender = customers_df['gender'].replace('', pd.NA).dropna().nunique()

blank_gender = customers_df['gender'].isna().sum() + (customers_df['gender'].astype(str).str.strip() == '').sum()




audit_table = pd.DataFrame([
    ["Total Orders", "Total number of orders placed", len(order_df)],
    ["Exact Duplicate Rows", "Rows identical across all columns", order_duplicate_count],
    ["Pending Status Rows", "Rows where status = 'pending'", orders_pending],
    ["Distinct Order_id ","Unique Order ID Values",distinct_order_id],
    ["Non Numeric Values", "Entries cannot be converted to number",non_numeric],
    ["Negative Delivery Minutes","Invalid Delivery minutes",negative_delivery],
    ["Blank Delivery Minutes", "No entry for delivery minutes",blank_delivery_minutes],
    ["Blank Age","Age column with no entries",blank_age],
    ["Wrong Age ","Age entries less than 10 and >100 ",out_of_range_age],
    ["Distinct city ","NUmber of distinct raw city spellings",raw_city_spellings],
    ["Distinct gender","Number of distinct non blank spelling",unique_gender],
    ["Blank Gender"," No entry fof Gender coloumn",blank_gender]
    ], columns=["Metric", "Description", "Count"])

print(audit_table)


# Cleaning the Data

# Standardise city by removing leading/trailing spaces and applying title case so exactly six cities remain.


customers_df["city"]=customers_df["city"].astype(str).str.strip().str.title()



# Cleaning the Gender column

clean_map = {
    'MALE': 'male',
    'm': 'male',
    'f': 'female',
    'FEMALE':'female',
    'F':'female',
    'M':'male',
    'Female':'female',
    'Male':'male',
    '':'unknown'
    }

customers_df['gender'] = customers_df['gender'].replace(clean_map)

    # Treat ages outside 10–100 as missing, then fill all missing ages with the median age of the customer's city, computed with groupby and transform.

customers_df['age'] = pd.to_numeric(customers_df['age'], errors='coerce').astype(float)

customers_df['age']=customers_df['age'].where(customers_df['age'].between(10,100))



customers_df['age'] = customers_df.groupby('city')['age'].transform(lambda x: x.fillna(x.median()))



# Convert signup_date to a datetime. 
customers_df['signup_date'] = pd.to_datetime(customers_df['signup_date'])

    #Cleaning Orders table 

    #Convert order_ts to a datetime.
order_df['order_ts'] = pd.to_datetime(order_df['order_ts'])


    #Remove exact duplicate rows.
order_df=order_df.drop_duplicates()


    # Getting the specific orders
    #print(orders_df[orders_df['order_id'].isin(['ORD-100918'])])

order_df = order_df.sort_values(by=['order_ts'], ascending=False)
order_df = order_df.drop_duplicates(subset='order_id', keep='first')

    #Convert order_amount to numeric, turning unparseable entries into missing values, then fill each missing amount 
    # from the product price: order_amount = round(price_inr × quantity × (1 − discount_pct / 100), 2).

    # Convert to numeric, turning unparseable entries into NaN
order_df['order_amount'] = pd.to_numeric(order_df['order_amount'], errors='coerce')



order_df['order_amount'] = order_df['order_amount'].fillna(round(order_df['product_id'].map(products_df.set_index('product_id')['price_inr']) * 
order_df['quantity'] * 
    (1 - order_df['discount_pct'] / 100), 
    2
    ))



    #print(orders_df[orders_df['order_id'].isin(['ORD-109208'])])

order_df['delivery_minutes'] = pd.to_numeric(order_df['delivery_minutes'], errors='coerce')

    #print(orders_df.info())
    #Set negative delivery_minutes to missing

order_df['delivery_minutes'] = np.where(pd.to_numeric(order_df['delivery_minutes'],errors='coerce') < 0, np.nan, order_df['delivery_minutes'] )


    #print(orders_df[orders_df['order_id'].isin(['ORD-109208'])])

    #fill every missing delivery_minutes with the 
    #median delivery time of that order's product category among completed orders (use groupby and transform). 

    # 1. Map category from products onto orders temporarily for grouping
category_series = order_df['product_id'].map(products_df.set_index('product_id')['category'])

    # 2. Compute the median delivery minutes per category for completed orders using transform
completed_mask = order_df['status'] == 'completed'
median_transform = (
    order_df['delivery_minutes']
    .where(completed_mask)
    .groupby(category_series)
    .transform('median')
    )

    # 3. Fill missing values in the original orders DataFrame
order_df['delivery_minutes'] = order_df['delivery_minutes'].fillna(median_transform)

#print(orders_df[orders_df['order_id'].isin(['ORD-111883'])])

#For orders with status cancelled, leave delivery_minutes missing — a cancelled order was never delivered, 
# so imputing a delivery time for it would invent data.


order_df.loc[order_df['status'].str.lower() == 'cancelled', 'delivery_minutes'] = np.nan

    #print(orders_df[orders_df['order_id'].isin(['ORD-111883'])])


orders_df=order_df
order_df.to_csv('data/clean/orders.csv',index=False)
products_df.to_csv('data/clean/products.csv',index=False)
customers_df.to_csv('data/clean/customers.csv',index=False)


def write_to_sqldb():
# Writing the data into SQLite Database

    orders_df.to_sql("orders",conn,index=False,if_exists="replace")
    products_df.to_sql("products",conn,index=False,if_exists="replace")
    customers_df.to_sql("customers",conn,index=False,if_exists="replace")

# YYYY-MM-DD HH:MM:SS
    cursor = conn.cursor()

# Example: Convert a DATE column to a TEXT string format ('YYYY-MM-DD')

    cursor.execute(
    """
    UPDATE orders 
    SET order_ts= STRFTIME('%Y-%m-%d %H:%M:%S', order_ts)
    """
    )

    cursor.execute("""ALTER TABLE orders RENAME TO orders_old""")
    cursor.execute(""" create table orders(
    order_id text,
    customer_id text,
    product_id text,
    quantity integer,
    order_ts text,
    status text,
    discount_pct integer,
    order_amount real,
    delivery_minutes real)

    """)

    cursor.execute("""insert into orders(order_id,customer_id,product_id,quantity,order_ts,status,discount_pct,order_amount,delivery_minutes)
    select order_id,customer_id,product_id,quantity,order_ts,status,discount_pct,order_amount,delivery_minutes from orders_old """)

    cursor.execute(""" drop table orders_old""")

    cursor.close()


    conn.commit()
    conn.close()




DB_FILE_PATH = "data/instamart.db" 
def execute_sqlite_validation(sql_file_path):
    if not os.path.exists(DB_FILE_PATH):
        print(f" SQLite database file not found at: {DB_FILE_PATH}")
        return
    if not os.path.exists(sql_file_path):
        print(f" SQL query file not found at: {sql_file_path}")
        return
        
    with open(sql_file_path, 'r', encoding='utf-8') as file:
        sql_content = file.read()

    # Split queries by semicolon and filter empty blocks
    queries = [q.strip() for q in sql_content.split(';') if q.strip()]

    print(f"\n Found {len(queries)} validation blocks. Executing native SQLite cursor connections...\n")
    all_results = []

    # 2. Open native SQLite Connection and Cursor
    conn=sqlite3.connect("data/instamart.db")
    cursor = conn.cursor()

    for idx, query in enumerate(queries, start=1):
        try:
            # Execute the raw query text string directly on the cursor
            cursor.execute(query)
            
            # Fetch all rows and extract column names from the cursor description
            rows = cursor.fetchall()
            columns = [col[0] for col in cursor.description]
            
            # Convert raw results to a Pandas DataFrame manually
            df = pd.DataFrame(rows, columns=columns)
            
            if not df.empty:
                # Force standard column alignment across all different query outputs
                df.columns = ['test_name', 'actual_value', 'expected_value', 'status']
                all_results.append(df)
            else:
                print(f"Query block {idx} returned no data rows.")
                
        except Exception as e:
            print(f"Error executing Query block {idx}:")
            print("\n".join(query.split("\n")[:3]) + " ...")
            print(f"Error Details: {e}\n" + "-"*40)

    # Close database assets safely
    cursor.close()
    conn.close()

    # 3. Display Summary Report Dashboard
    if all_results:
        final_report = pd.concat(all_results, ignore_index=True)
        
        print("=" * 75)
        print("               SQLITE DATA VALIDATION REPORT ")
        print("=" * 75)
        print(final_report.to_string(index=False))
        print("=" * 75)
        
        failures = final_report[final_report['status'] == 'FAIL']
        if not failures.empty:
            print(f" ALERT: {len(failures)} validation check(s) FAILED.")
        else:
            print(" ALL CHECKPOINTS PASSED")
    else:
        print("No validation results could be processed.")




write_to_sqldb()


execute_sqlite_validation("sql/queries.sql")


