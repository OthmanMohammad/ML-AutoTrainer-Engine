import streamlit as st
import json
import os
import pandas as pd
import joblib

# File to store project data
PROJECTS_FILE = 'projects.json'
PROJECT_DATA_DIR = 'projects'

def add_project_form():
    with st.form("Add Project Form", clear_on_submit=True):
        project_name = st.text_input("Project Name").strip()
        project_description = st.text_area("Project Description")
        submit_button = st.form_submit_button("Add Project")

        if submit_button and project_name:
            # To ensure project name doesn't conflict with an existing directory
            if os.path.exists(os.path.join(PROJECT_DATA_DIR, project_name)):
                st.error(f"Project '{project_name}' already exists.")
            else:
                save_project(project_name, project_description)
                st.success(f"Project '{project_name}' has been added.")
                st.session_state['selected_project'] = project_name

def save_project(name, description):
    project = {'name': name, 'description': description}
    
    # Load existing projects
    projects = get_projects_list()

    # Add the new project
    projects.append(project)

    # Save the updated projects back to the file
    with open(PROJECTS_FILE, 'w') as f:
        json.dump(projects, f)

def get_projects_list():
    try:
        with open(PROJECTS_FILE, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        # If no file found or file is empty return an empty list
        return []

def get_project_names():
    projects = get_projects_list()
    return [project['name'] for project in projects]

def load_project_data(project_name):
    project_data_path = os.path.join(PROJECT_DATA_DIR, project_name, 'processed_data.csv')
    if os.path.exists(project_data_path):
        return pd.read_csv(project_data_path)
    return None

def save_project_data(project_name, data):
    project_folder = os.path.join(PROJECT_DATA_DIR, project_name)
    os.makedirs(project_folder, exist_ok=True)
    project_data_path = os.path.join(project_folder, 'processed_data.csv')
    data.to_csv(project_data_path, index=False)

def save_model(project_name, model):
    project_folder = os.path.join(PROJECT_DATA_DIR, project_name)
    os.makedirs(project_folder, exist_ok=True)
    model_path = os.path.join(project_folder, 'model.joblib')
    joblib.dump(model, model_path)

def load_model(project_name):
    project_folder = os.path.join(PROJECT_DATA_DIR, project_name)
    model_path = os.path.join(project_folder, 'model.joblib')
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        st.error("Model file not found in the project.")
        return None
