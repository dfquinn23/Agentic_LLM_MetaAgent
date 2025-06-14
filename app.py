# app.py
import os
import traceback
from dotenv import load_dotenv

import streamlit as st
from crewai import Crew
from tasks import create_response_tasks
from compare import create_comparison_task

# Load environment variables
load_dotenv()

st.set_page_config(page_title="Multi-LLM Prompt Comparator")
st.title("🧠 LLM Response Comparator")
prompt = st.text_area("Enter a question or prompt")

if st.button("Compare Responses") and prompt:
    # Step 1: Generate response tasks for each LLM
    response_tasks = create_response_tasks(prompt)
    response_crew = Crew(
        tasks=response_tasks,
        agents=[task.agent for task in response_tasks]
    )

    try:
        response_outputs = response_crew.kickoff()
    except Exception as e:
        st.error("❌ Crew execution failed. Check terminal for traceback.")
        print("=== Full Error Trace ===")
        traceback.print_exc()
        raise e

    agent_responses = {
        task.agent.role: output for task, output in zip(response_tasks, response_outputs)
    }

    # Display LLM outputs
    st.subheader("🔍 Responses by LLM")
    for name, text in agent_responses.items():
        st.markdown(f"### {name}")
        st.markdown(text)

    # Step 2: Compare the responses
    comparison_task = create_comparison_task(agent_responses, prompt)
    compare_crew = Crew(tasks=[comparison_task],
                        agents=[comparison_task.agent])

    try:
        comparison_output = compare_crew.kickoff()
    except Exception as e:
        st.error("❌ Comparison agent failed. Check terminal for traceback.")
        print("=== Comparison Error Trace ===")
        traceback.print_exc()
        raise e

    # Display comparison summary
    st.subheader("📋 Comparison Summary")
    if isinstance(comparison_output, list) and len(comparison_output) > 0:
        st.markdown(comparison_output[0])
    else:
        st.markdown(comparison_output)
