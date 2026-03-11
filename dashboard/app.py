from __future__ import annotations

import os
import time

import requests
import streamlit as st

API_BASE_URL = os.getenv("DASHBOARD_API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="ParkVision Dashboard", layout="wide")
st.title("ParkVision - Parking Occupancy Dashboard")

refresh_seconds = st.sidebar.slider("Refresh interval (seconds)", 1, 10, 2)

placeholder = st.empty()

while True:
    with placeholder.container():
        st.subheader("Current Occupancy")
        try:
            response = requests.get(f"{API_BASE_URL}/occupancy/latest", timeout=5)
            response.raise_for_status()
            data = response.json()

            col1, col2, col3 = st.columns(3)
            col1.metric("Total spots", data["total_spots"])
            col2.metric("Occupied", data["occupied_spots"])
            col3.metric("Free", data["free_spots"])

            st.caption(f"Timestamp: {data['timestamp']} | Frame: {data['frame_index']}")
            st.dataframe(data["spots"], use_container_width=True)
        except Exception as exc:
            st.error(f"Failed to fetch occupancy data: {exc}")

    time.sleep(refresh_seconds)
    st.rerun()
