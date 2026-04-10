/** Representative customer payload aligned with backend `CustomerFeatures`. */
export const sampleCustomer = {
  customer_id: "CUST-001",
  age: 34,
  gender: "M" as const,
  location: "Mumbai",
  tenure_months: 18,
  subscription_plan: "Premium",
  monthly_charge: 89.99,
  total_charges: 1619.82,
  num_products: 3,
  contract_type: "One Year",
  payment_method: "Credit Card",
  login_frequency_30d: 22,
  support_tickets_90d: 1,
  avg_session_duration_min: 14.5,
  last_activity_days_ago: 2,
  nps_score: 8,
  late_payments_count: 0,
  discount_used: false,
  referrals_made: 2,
  last_feedback: "Really love the service, very fast and reliable!",
};

export const highRiskCustomer = {
  ...sampleCustomer,
  customer_id: "CUST-HR-001",
  support_tickets_90d: 12,
  late_payments_count: 4,
  last_activity_days_ago: 95,
  nps_score: 2,
  login_frequency_30d: 1,
  last_feedback: "Terrible experience, thinking of cancelling.",
};
