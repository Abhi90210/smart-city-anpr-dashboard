import streamlit as st
from google import genai
from google.genai import types

def get_gemini_client():
    api_key = st.secrets.get("GEMINI_API_KEY")
    if not api_key:
        return None
    return genai.Client(api_key=api_key)

def generate_fir_and_dispatch(plate_number: str, anomaly_data: dict, trajectory_summary: list) -> str:
    """
    Generates a structured forensic analysis and official FIR dispatch brief.
    """
    client = get_gemini_client()
    if not client:
        return "⚠️ Error: `GEMINI_API_KEY` not found in Streamlit Secrets. Please configure it to run AI summaries."

    # Format the trail summary for the LLM
    trail_text = "\n".join([
        f"- {pt.get('time')}: Camera '{pt.get('camera')}' (Speed: {pt.get('speed', 'N/A')} km/h)"
        for pt in trajectory_summary
    ]) if trajectory_summary else "No sequence sightings logged."

    system_instruction = """
    You are the Senior Cyber-Forensic Dispatcher for the Smart City Integrated Command and Control Centre (ICCC).
    Analyze traffic anomalies (speeding, impossible travel times, suspected clone plates) and generate an authoritative, 
    actionable dossier for field interception and legal filing under the Indian Motor Vehicles Act.
    """

    prompt = f"""
    ### INCIDENT DATA
    - Target License Plate: {plate_number}
    - Violation / Anomaly Category: {anomaly_data.get('Type', 'Unknown')}
    - Timestamp: {anomaly_data.get('Time', 'N/A')}
    - System Sensor Alert: {anomaly_data.get('Description', 'N/A')}
    - Sensor Trajectory History:
    {trail_text}

    ### REQUIRED OUTPUT FORMAT
    Please produce the report using the following three sections:

    #### 1. Threat & Anomaly Evaluation
    - State clearly whether this indicates physical cloning (impossible physical travel time between junctions), fraudulent registration, or excessive speeding.
    - Risk rating (Low / Medium / High / Critical).

    #### 2. Tactical Interception Directive (Control Room Dispatch)
    - Identify the probable route vector based on camera sightings.
    - Recommend 2 immediate checkpoints/toll plazas for traffic police barricading and vehicle impoundment.

    #### 3. Auto-Drafted E-Challan / Preliminary FIR Notice
    - Draft an official legal incident summary citing sections under the Motor Vehicles (Amendment) Act (e.g., Section 183 for Speeding, Section 192/420 IPC for Number Plate Forgery/Cloning).
    - Offender Plate, Date/Time, and Prescribed Enforcement Action.
    """

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.2, # Lower temperature for formal, deterministic legal/technical output
            )
        )
        return response.text
    except Exception as e:
        return f"🚨 API Generation Failed: {str(e)}"