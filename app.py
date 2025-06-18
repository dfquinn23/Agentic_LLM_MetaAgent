import os
import traceback
from dotenv import load_dotenv  # Used for local .env files

# --- CRITICAL FIX FOR CHROMA/SQLITE3 START ---
# These lines MUST be at the very top of your app.py, before other imports
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass
# --- CRITICAL FIX FOR CHROMA/SQLITE3 END ---

import streamlit as st
from crewai import Crew
# No need to import agents directly here as they are imported by tasks/compare
from tasks import create_response_tasks
from compare import create_comparison_task

# --- Load environment variables for local development ---
# This is here for when you run the app locally using 'streamlit run app.py'
# Streamlit Cloud uses its 'Secrets' mechanism instead of .env files.
load_dotenv()

# --- Define Logo Path (Relative for Deployment) ---
# Assuming your logo is in a subfolder named 'assets' in your repo root.
# Adjust the path if your logo file is located elsewhere relative to app.py
LOGO_PATH = "assets/email_sig.png"

# --- Streamlit Page Configuration (MUST be the first Streamlit command) ---
st.set_page_config(
    page_title="Multi-LLM Response Comparison Tool",
    page_icon=LOGO_PATH,  # Use the relative path for the favicon
    layout="centered"  # Or "wide"
)

# --- Display Logo within the App Body (Optional, if you want it larger) ---
# If you want the logo to appear prominently on the page itself,
# rather than just as the favicon in the browser tab.
st.image(LOGO_PATH, width=150)  # Adjust width as needed

st.title("LLM Response Comparison Tool")  # Title without icon argument
prompt = st.text_area("Enter a question or prompt")


if st.button("Compare Responses") and prompt:
    # Step 1: Generate response tasks for each LLM
    response_tasks = create_response_tasks(prompt)

    print(f"DEBUG: Number of response_tasks defined: {len(response_tasks)}")

    response_crew = Crew(
        tasks=response_tasks,
        agents=[task.agent for task in response_tasks],
        verbose=True
    )

    try:
        crew_result_object = response_crew.kickoff()
        response_outputs = crew_result_object.tasks_output
        print(
            f"DEBUG: Number of individual task outputs from CrewOutput: {len(response_outputs)}")

    except Exception as e:
        st.error("❌ Crew execution failed. Check terminal for traceback.")
        print("=== Full Error Trace ===")
        traceback.print_exc()
        raise e

    agent_responses = {}
    for task_obj, output_task_obj in zip(response_tasks, response_outputs):
        agent_role = task_obj.agent.role
        extracted_content = None
        if hasattr(output_task_obj, 'raw') and output_task_obj.raw is not None:
            extracted_content = output_task_obj.raw
        elif hasattr(output_task_obj, 'description') and output_task_obj.description is not None:
            extracted_content = output_task_obj.description
        else:
            extracted_content = f"Could not extract content for {agent_role}. Full output: {output_task_obj}"
            print(f"DEBUG: {extracted_content}")

        agent_responses[agent_role] = extracted_content

    st.subheader("🔍 Responses by LLM")
    for name, text in agent_responses.items():
        st.markdown(f"### {name}")
        st.markdown(text)

    # Step 2: Compare the responses
    comparison_task = create_comparison_task(agent_responses, prompt)
    compare_crew = Crew(tasks=[comparison_task],
                        agents=[comparison_task.agent],
                        verbose=True)

    try:
        comparison_crew_result_object = compare_crew.kickoff()
        comparison_output = comparison_crew_result_object.raw

        if isinstance(comparison_output, list) and len(comparison_output) > 0:
            comparison_output = comparison_output[0]
        elif not isinstance(comparison_output, str):
            comparison_output = str(comparison_output)

    except Exception as e:
        st.error("❌ Comparison agent failed. Check terminal for traceback.")
        print("=== Comparison Error Trace ===")
        traceback.print_exc()
        raise e

    st.subheader("📋 Comparison Summary")
    st.markdown(comparison_output)


# OLD:
# import os
# import traceback
# from dotenv import load_dotenv

# import streamlit as st
# from crewai import Crew
# from tasks import create_response_tasks
# from compare import create_comparison_task

# # Load environment variables
# load_dotenv()

# st.set_page_config(page_title="Multi-LLM Prompt Comparator")
# st.title("🧠 LLM Response Comparison Tool")
# prompt = st.text_area("Enter a question or prompt")

# if st.button("Compare Responses") and prompt:
#     # Step 1: Generate response tasks for each LLM
#     response_tasks = create_response_tasks(prompt)

#     # Keep this debug print
#     print(f"DEBUG: Number of response_tasks defined: {len(response_tasks)}")

#     response_crew = Crew(
#         tasks=response_tasks,
#         agents=[task.agent for task in response_tasks],
#         verbose=True  # Keep verbose for debugging
#     )

#     try:
#         # kickoff() now returns a CrewOutput object
#         crew_result_object = response_crew.kickoff()

#         # Access the list of individual task outputs from the CrewOutput object
#         response_outputs = crew_result_object.tasks_output

#         # Keep this debug print
#         print(
#             f"DEBUG: Number of individual task outputs from CrewOutput: {len(response_outputs)}")

#     except Exception as e:
#         st.error("❌ Crew execution failed. Check terminal for traceback.")
#         print("=== Full Error Trace ===")
#         traceback.print_exc()
#         raise e

#     # --- REVISED MODIFICATION START: Correctly iterate over tasks_output ---
#     agent_responses = {}

#     # Now, response_outputs should be a list of TaskOutput objects
#     # We can zip them with response_tasks for robust mapping
#     for task_obj, output_task_obj in zip(response_tasks, response_outputs):
#         agent_role = task_obj.agent.role

#         # Safely extract content from the TaskOutput object
#         extracted_content = None
#         if hasattr(output_task_obj, 'raw') and output_task_obj.raw is not None:
#             extracted_content = output_task_obj.raw
#         elif hasattr(output_task_obj, 'description') and output_task_obj.description is not None:
#             # Fallback to description if raw is unexpectedly empty or not available
#             extracted_content = output_task_obj.description
#         else:
#             extracted_content = f"Could not extract content for {agent_role}. Full output: {output_task_obj}"
#             print(f"DEBUG: {extracted_content}")  # More detailed debug output

#         agent_responses[agent_role] = extracted_content
#     # --- REVISED MODIFICATION END ---

#     # Display LLM outputs
#     st.subheader("🔍 Responses by LLM")
#     for name, text in agent_responses.items():
#         st.markdown(f"### {name}")
#         st.markdown(text)

#     # Step 2: Compare the responses
#     comparison_task = create_comparison_task(agent_responses, prompt)
#     compare_crew = Crew(tasks=[comparison_task],
#                         agents=[comparison_task.agent],
#                         verbose=True  # Keep verbose here too
#                         )

#     try:
#         # This kickoff also returns a CrewOutput object for the comparison crew
#         comparison_crew_result_object = compare_crew.kickoff()
#         # For the comparison, we likely want the final raw output of the crew
#         comparison_output = comparison_crew_result_object.raw

#         # CrewAI sometimes returns a list with one item for single tasks even with .raw.
#         # Let's ensure it's a string.
#         if isinstance(comparison_output, list) and len(comparison_output) > 0:
#             comparison_output = comparison_output[0]
#         elif not isinstance(comparison_output, str):
#             # Convert to string as last resort
#             comparison_output = str(comparison_output)

#     except Exception as e:
#         st.error("❌ Comparison agent failed. Check terminal for traceback.")
#         print("=== Comparison Error Trace ===")
#         traceback.print_exc()
#         raise e

#     # Display comparison summary
#     st.subheader("📋 Comparison Summary")
#     # Now comparison_output should consistently be a string
#     st.markdown(comparison_output)
