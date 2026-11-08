import json
import os
import random
import re
import time
from urllib import response
import numpy as np
import pandas as pd
import mock_llm


# =============================================================
# TASK 1: INPUT GUARDRAILS
# =============================================================
df_tickets = pd.read_csv("data/raw/tickets.csv")
MAX_INPUT_TOKENS = 400

statuses = []
for idx, row in df_tickets.iterrows():
  txt = str(row["text"])
  if pd.isna(txt) or txt.strip() == "":
    statuses.append("skipped_empty")
  elif mock_llm.estimate_tokens(txt) > MAX_INPUT_TOKENS:
    statuses.append("flagged_long")
  else:
    statuses.append("eligible")

df_tickets["guardrail_status"] = statuses
print("Task 1 - Guardrail Status Counts:")
print(df_tickets["guardrail_status"].value_counts())


# =============================================================
# TASK 2: REDACT PII
# =============================================================
email_pattern = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
phone_pattern = re.compile(r"\b[6-9]\d{9}\b")
order_pattern = re.compile(r"ORD-\d{6}")

email_count, phone_count, order_count = 0, 0, 0
redacted_texts = []

for txt in df_tickets["text"]:
  if not isinstance(txt, str):
    txt = ""
  email_count += len(email_pattern.findall(txt))
  phone_count += len(phone_pattern.findall(txt))
  order_count += len(order_pattern.findall(txt))

  t1 = email_pattern.sub("[EMAIL]", txt)
  t2 = phone_pattern.sub("[PHONE]", t1)
  t3 = order_pattern.sub("[ORDER_ID]", t2)
  redacted_texts.append(t3)

df_tickets["redacted_text"] = redacted_texts
total_subs = email_count + phone_count + order_count
print(
    f"\nTask 2 - PII Substituted: Emails={email_count}, Phones={phone_count},"
    f" Order IDs={order_count}, Total={total_subs}"
)

# Verify re-applying finds nothing
leftover_emails = sum(len(email_pattern.findall(t)) for t in redacted_texts)
leftover_phones = sum(len(phone_pattern.findall(t)) for t in redacted_texts)
leftover_orders = sum(len(order_pattern.findall(t)) for t in redacted_texts)
print(
    "Leftovers after re-applying patterns:"
    f" Emails={leftover_emails}, Phones={leftover_phones},"
    f" Orders={leftover_orders}"
)

# =============================================================
# TASK 3: PROMPTS
# =============================================================
zero_shot_prompt = (
    "You are an AI support agent. CLASSIFY the incoming customer ticket into"
    " one of these four allowed categories: billing, delivery, technical, or"
    ' fraud. You must output a JSON-only reply in the exact format {"category":'
    ' "...", "reason": "..."}.'
)

few_shot_prompt = (
    "You are an AI support agent. CLASSIFY the incoming customer ticket into"
    " one of these four allowed categories: billing, delivery, technical, or"
    ' fraud. You must output a JSON-only reply in the exact format {"category":'
    ' "...", "reason": "..."}.\n\nEXAMPLES:\nTicket: I was charged twice for my'
    ' subscription. -> {"category": "billing", "reason": "Double charge'
    ' complaint."}\nTicket: Someone used my account to buy items without my'
    ' consent. -> {"category": "fraud", "reason": "Unauthorized account'
    " usage.\"}"
)

# =============================================================
# TASKS 4, 5, 6, 7: PIPELINE EXECUTION
# =============================================================
def run_pipeline(prompt_type="zero_shot", token_budget=float("inf")):
  mock_llm.reset()
  system_prompt = (
      zero_shot_prompt if prompt_type == "zero_shot" else few_shot_prompt
  )
  allowed_categories = {"billing", "delivery", "technical", "fraud"}

  results = []
  total_tokens_used = 0

  for idx, row in df_tickets.iterrows():
    t_id = row["ticket_id"]
    c_id = row["customer_id"]
    status = row["guardrail_status"]
    redacted = row["redacted_text"]
    true_cat = row["true_category"]

    if status != "eligible":
      results.append({
          "ticket_id": t_id,
          "customer_id": c_id,
          "status": status,
          "category": None,
          "reason": None,
          "tokens_used": 0,
          "true_category": true_cat,
      })
      continue

    if total_tokens_used >= token_budget:
      results.append({
          "ticket_id": t_id,
          "customer_id": c_id,
          "status": "budget_exhausted",
          "category": None,
          "reason": None,
          "tokens_used": 0,
          "true_category": true_cat,
      })
      continue

    # Resilient call with exponential backoff & jitter
    base_delay = 0.05
    response = None
    success = False
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": redacted},
    ]
    for n in range(3):
      try:
        response = mock_llm.chat(messages)
        success = True
        break
      except mock_llm.TransientLLMError:
        wait = base_delay * (2**n) * random.uniform(0.9, 1.15)
        time.sleep(0.001)  # simulated wait

    if not success or response is None:
      results.append({
          "ticket_id": t_id,
          "customer_id": c_id,
          "status": "flagged_invalid_output",
          "category": None,
          "reason": None,
          "tokens_used": 0,
          "true_category": true_cat,
      })
      continue

    tokens_used = response.usage.total_tokens
    total_tokens_used += tokens_used
    reply_text = response.choices[0].message.content.strip()

    parsed_category = None
    parsed_reason = None
    ticket_status = "flagged_invalid_output"

    # Validate JSON
    try:
      data = json.loads(reply_text)
      cat = data.get("category")
      if isinstance(cat, str) and cat.lower() in allowed_categories:
        parsed_category = cat.lower()
        parsed_reason = data.get("reason")
        ticket_status = "classified_json"
    except json.JSONDecodeError:
      pass

    # Keyword fallback validation
    if ticket_status == "flagged_invalid_output":
      words = re.findall(r"[a-z]+", reply_text.lower())
      matched_cats = [w for w in words if w in allowed_categories]
      if len(matched_cats) == 1:
        parsed_category = matched_cats[0]
        ticket_status = "classified_keyword"

    results.append({
        "ticket_id": t_id,
        "customer_id": c_id,
        "status": ticket_status,
        "category": parsed_category,
        "reason": parsed_reason,
        "tokens_used": tokens_used,
        "true_category": true_cat,
    })

  return pd.DataFrame(results), total_tokens_used


# Run Task 6 Evaluation
df_zero, tokens_zero = run_pipeline("zero_shot")
df_few, tokens_few = run_pipeline("few_shot")
print(f"\nTask 6 - Zero-Shot Total Tokens: {tokens_zero}")
print(f"Task 6 - Few-Shot Total Tokens: {tokens_few}")

# Task 7(a) Budget Demo
df_budget, _ = run_pipeline("zero_shot", token_budget=2500)
exhausted_count = (df_budget["status"] == "budget_exhausted").sum()
print(f"\nTask 7(a) - Budget Exhausted Count (Budget=2500): {exhausted_count}")
df_budget.to_csv("outputs/triage_budget_demo.csv", index=False)

# Task 7(b) Final Routing Output
df_final, _ = run_pipeline("few_shot", token_budget=100000)
if os.path.exists("outputs/customer_segments.csv"):
  segments = pd.read_csv("outputs/customer_segments.csv")
  df_final = df_final.merge(
      segments[["customer_id", "segment_name", "is_premium"]],
      on="customer_id",
      how="left",
  )
else:
  df_final["segment_name"] = ""
  df_final["is_premium"] = False

df_final["is_premium"] = df_final["is_premium"].fillna(False)

routes, priorities = [], []
for idx, row in df_final.iterrows():
  status = row["status"]
  cat = row["category"]
  is_prem = row["is_premium"]

  if status in [
      "skipped_empty",
      "flagged_long",
      "flagged_invalid_output",
      "budget_exhausted",
  ]:
    routes.append("human_review")
    priorities.append("NORMAL")
  elif cat == "fraud":
    routes.append("security_team")
    priorities.append("HIGH")
  elif is_prem:
    routes.append("premium_desk")
    priorities.append("HIGH")
  else:
    routes.append("support_queue")
    priorities.append("NORMAL")

df_final["route"] = routes
df_final["priority"] = priorities

output_cols = [
    "ticket_id",
    "customer_id",
    "status",
    "category",
    "segment_name",
    "priority",
    "route",
    "tokens_used",
]
df_final["category"] = df_final["category"].fillna("")
df_final["segment_name"] = df_final["segment_name"].fillna("")
df_final[output_cols].to_csv("outputs/triage_results.csv", index=False)
print("Saved outputs/triage_results.csv successfully!")