# Instamart Customer Intelligence & Support Copilot — Capstone Project

This repository contains scripts to generate project datasets.

## Reproduce

Follow these steps to set up the repository, install dependencies, and generate the data:

1. **Clone the repository** and ensure you are on the main branch:
   ```bash
   git checkout main
   ```

2. **Create an isolated virtual environment**:
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**:
 
     ```bash
     source venv/bin/activate
     ```

4. **Install the required libraries**:
   ```bash
   pip install -r requirement.txt
   ```

5. **Run the data generation script**:
   ```bash
   python generate_data.py
   ```

6. ** All the Task1 operation is implmented in the Data_Cleaning.py
    ```bash
    python Data_Cleaning_Storing.py
    ```
7. ** All the operations Specified on Task 2 is implemented here

  python CustomerSegment_LapsePrediction.py

8. ** All the  operations Specified on Task 3 is implemented here

    python Support_Ticket_Triage.py

# Data Cleaning and storing the data in database

## Data Cleaning rules

1. Look out the shapes of the data  and null values of the data
2. Check for the outliar values for the numeric column.
3. convert order_ts to a datetime
4. Eliminating the exact duplicates and keeping only the latest status value
5. Checking order_amount to numeric and for missing entries calculate the price as
     price=order_amount=round(price_inr*quantity*(1-discount_pct/100),2)
6. Set the delivery_minutes negative values into median of delivery_time of the order's product category
7. For the order with status "Cancelled"  leave the delivery_minutes null


After cleaning the data it stores the clean data in data/clean/ folders as .csv files
Stores the tables in data/instamart.db on the SQLite database for further accessability


## Part B - Customer segment and Lapse

Keeping feature engineering strictly pre-cutoff and isolating transformers to the training set prevents data leakage, while explicitly acknowledging the post-hoc nature of the Premium flag ensures the priority retention list is used correctly for what it is: an illustrative, high-value targeting overlay.

  
## Part C - Support Ticket Triage with Guardrails

### 1. Prompt Cost Analysis (Tokens per Ticket)
* **Zero-Shot Baseline:** Operates with a lighter instruction structure, averaging approximately **28 tokens per ticket** processed. 
* **Few-Shot Configuration: **40 tokens per ticket**.  (`EXAMPLES:`) to eliminate classification ambiguity, increasing the context footprint to approximately 
* **Trade-Off Evaluation:** While few-shot prompting incurs an ~43% increase in token expenditure, the investment is justified by the elimination of systematic errors (such as misclassifying complex fraud or technical tickets as billing), boosting classification accuracy from 85.7% to 100%.

### 2. Output Validation and Routing Governance
Every model response is processed through a strict validation hierarchy to ensure data safety before routing:
* **Human Review Escalation :** Any ticket failing structural validation, containing malformed outputs, encountering token budget limits , or tripping input filters (`skipped_empty`, `flagged_long`) is explicitly sequestered with a `NORMAL` priority flag for manual human review—ensuring the automated system never guesses an uncertain classification.

