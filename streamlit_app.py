import asyncio
import json
from typing import Any, Optional

import streamlit as st

from app.core.model_registry import ModelRegistry
from app.schemas.schemas import CustomerFeatures
from app.services.monitoring_service import CustomerMonitoringService


def _load_registry_sync() -> ModelRegistry:
    """
    Streamlit runs sync. ModelRegistry.load_all() is async, so we run it
    in a dedicated event loop.
    """
    registry = ModelRegistry()
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(registry.load_all())
    finally:
        loop.close()
    return registry


@st.cache_resource(show_spinner="Loading ML models…")
def get_service() -> CustomerMonitoringService:
    registry = _load_registry_sync()
    return CustomerMonitoringService(registry)


def _parse_customer_json(text: str) -> tuple[Optional[CustomerFeatures], Optional[str]]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        return None, f"Invalid JSON: {e.msg}"

    try:
        # Pydantic validation against CustomerFeatures schema.
        customer = CustomerFeatures.model_validate(data)
        return customer, None
    except Exception as e:  # streamlit display-friendly
        return None, str(e)


def _render_registry_status(registry: ModelRegistry) -> None:
    st.sidebar.subheader("Model status")
    st.sidebar.json(registry._status)
    st.sidebar.caption(f"Loaded at: {getattr(registry, 'loaded_at', None)}")


def main() -> None:
    st.set_page_config(page_title="Customer Monitoring Lab", layout="wide")
    st.title("Customer Monitoring Lab")

    svc = get_service()
    registry = svc.registry

    _render_registry_status(registry)
    st.caption("Inference runs in-app using the same model registry logic as the FastAPI backend.")

    churn_tab, segment_tab, anomaly_tab, sentiment_tab, report_tab = st.tabs(
        ["Churn", "Segmentation", "Anomaly", "Sentiment", "Full report"]
    )

    with churn_tab:
        st.subheader("Churn prediction")
        customer_json = st.text_area(
            "CustomerFeatures JSON",
            key="churn_customer_json",
            value=json.dumps(
                {
                    "customer_id": "CUST-001",
                    "age": 34,
                    "gender": "M",
                    "location": "Mumbai",
                    "tenure_months": 18,
                    "subscription_plan": "Premium",
                    "monthly_charge": 89.99,
                    "total_charges": 1619.82,
                    "num_products": 3,
                    "contract_type": "One Year",
                    "payment_method": "Credit Card",
                    "login_frequency_30d": 22,
                    "support_tickets_90d": 1,
                    "avg_session_duration_min": 14.5,
                    "last_activity_days_ago": 2,
                    "nps_score": 8,
                    "late_payments_count": 0,
                    "discount_used": False,
                    "referrals_made": 2,
                    "last_feedback": "Really love the service, very fast and reliable!",
                },
                indent=2,
            ),
            height=380,
        )

        if st.button("Predict churn", type="primary"):
            customer, err = _parse_customer_json(customer_json)
            if err:
                st.error(err)
            elif customer:
                with st.spinner("Running churn prediction…"):
                    pred = svc.predict_churn(customer)
                st.success("Done")
                st.json(pred.model_dump(mode="json"))

    with segment_tab:
        st.subheader("Customer segmentation")
        customer_json = st.text_area(
            "CustomerFeatures JSON",
            key="segment_customer_json",
            height=380,
            value=json.dumps(
                {
                    "customer_id": "CUST-001",
                    "age": 34,
                    "gender": "M",
                    "location": "Mumbai",
                    "tenure_months": 18,
                    "subscription_plan": "Premium",
                    "monthly_charge": 89.99,
                    "total_charges": 1619.82,
                    "num_products": 3,
                    "contract_type": "One Year",
                    "payment_method": "Credit Card",
                    "login_frequency_30d": 22,
                    "support_tickets_90d": 1,
                    "avg_session_duration_min": 14.5,
                    "last_activity_days_ago": 2,
                    "nps_score": 8,
                    "late_payments_count": 0,
                    "discount_used": False,
                    "referrals_made": 2,
                    "last_feedback": "Great service and reliable support.",
                },
                indent=2,
            ),
        )

        if st.button("Predict segment", type="primary"):
            customer, err = _parse_customer_json(customer_json)
            if err:
                st.error(err)
            elif customer:
                with st.spinner("Running segmentation…"):
                    pred = svc.segment_customer(customer)
                st.success("Done")
                st.json(pred.model_dump(mode="json"))

    with anomaly_tab:
        st.subheader("Anomaly detection")
        customer_json = st.text_area(
            "CustomerFeatures JSON",
            key="anomaly_customer_json",
            height=380,
            value=json.dumps(
                {
                    "customer_id": "CUST-HR-001",
                    "age": 42,
                    "gender": "F",
                    "location": "Berlin",
                    "tenure_months": 4,
                    "subscription_plan": "Standard",
                    "monthly_charge": 55.0,
                    "total_charges": 220.0,
                    "num_products": 2,
                    "contract_type": "Month-to-Month",
                    "payment_method": "Bank transfer",
                    "login_frequency_30d": 1,
                    "support_tickets_90d": 12,
                    "avg_session_duration_min": 3.2,
                    "last_activity_days_ago": 95,
                    "nps_score": 2,
                    "late_payments_count": 4,
                    "discount_used": False,
                    "referrals_made": 0,
                    "last_feedback": "Terrible experience, thinking of cancelling.",
                },
                indent=2,
            ),
        )

        if st.button("Detect anomalies", type="primary"):
            customer, err = _parse_customer_json(customer_json)
            if err:
                st.error(err)
            elif customer:
                with st.spinner("Detecting anomalies…"):
                    pred = svc.detect_anomaly(customer)
                st.success("Done")
                st.json(pred.model_dump(mode="json"))

    with sentiment_tab:
        st.subheader("Sentiment (NLP / fallback keywords)")
        customer_id = st.text_input("Customer ID", value="CUST-001")
        source = st.selectbox("Source", ["feedback", "review", "support_ticket", "chat"], index=0)
        text = st.text_area(
            "Text",
            key="sentiment_text",
            value="Really love the service, very fast and reliable!",
            height=180,
        )

        if st.button("Analyze sentiment", type="primary"):
            with st.spinner("Analyzing…"):
                pred = svc.analyze_sentiment(customer_id=customer_id, text=text, source=source)
            st.success("Done")
            st.json(pred.model_dump(mode="json"))

    with report_tab:
        st.subheader("Full monitoring report")
        customer_json = st.text_area(
            "CustomerFeatures JSON",
            key="report_customer_json",
            height=380,
            value=json.dumps(
                {
                    "customer_id": "CUST-001",
                    "age": 34,
                    "gender": "M",
                    "location": "Mumbai",
                    "tenure_months": 18,
                    "subscription_plan": "Premium",
                    "monthly_charge": 89.99,
                    "total_charges": 1619.82,
                    "num_products": 3,
                    "contract_type": "One Year",
                    "payment_method": "Credit Card",
                    "login_frequency_30d": 22,
                    "support_tickets_90d": 1,
                    "avg_session_duration_min": 14.5,
                    "last_activity_days_ago": 2,
                    "nps_score": 8,
                    "late_payments_count": 0,
                    "discount_used": False,
                    "referrals_made": 2,
                    "last_feedback": "Really love the service, very fast and reliable!",
                },
                indent=2,
            ),
        )

        if st.button("Generate report", type="primary"):
            customer, err = _parse_customer_json(customer_json)
            if err:
                st.error(err)
            elif customer:
                with st.spinner("Generating full report…"):
                    report = svc.full_report(customer)
                st.success("Done")
                st.json(report.model_dump(mode="json"))


if __name__ == "__main__":
    main()

