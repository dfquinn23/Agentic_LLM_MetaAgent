# app.py
import os
from dotenv import load_dotenv

import streamlit as st
from crewai import Crew
from tasks import create_response_tasks
from compare import create_comparison_task

load_dotenv()

st.set_page_config(page_title="Multi-LLM Prompt Comparator")

st.title("🧠 LLM Response Comparator")
prompt = st.text_area("Enter a question or prompt")

if st.button("Compare Responses") and prompt:
    # Step 1: Get responses from all LLMs
    response_tasks = create_response_tasks(prompt)
    response_crew = Crew(tasks=response_tasks, agents=[
                         task.agent for task in response_tasks])
    response_outputs = response_crew.kickoff()

    agent_responses = {
        task.agent.role: output
        for task, output in zip(response_tasks, response_outputs)
    }

    # Display results
    st.subheader("🔍 Responses by LLM")
    for name, text in agent_responses.items():
        st.markdown(f"### {name}")
        st.markdown(text)

    # Step 2: Comparison summary
    comparison_task = create_comparison_task(agent_responses, prompt)
    compare_crew = Crew(tasks=[comparison_task],
                        agents=[comparison_task.agent])
    [comparison_output] = compare_crew.kickoff()

    st.subheader("📋 Comparison Summary")
    st.markdown(comparison_output)
