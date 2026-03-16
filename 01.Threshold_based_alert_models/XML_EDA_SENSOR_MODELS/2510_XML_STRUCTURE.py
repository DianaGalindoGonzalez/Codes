import streamlit as st
import pandas as pd
import os
import plotly.express as px
import xml.etree.ElementTree as ET
import networkx as nx
from pyvis.network import Network


# ------------------------------------------------------------------------------
# Streamlit App Configuration
# ------------------------------------------------------------------------------
st.set_page_config(page_title="XML Visualizer", layout="wide")

# ------------------------------------------------------------------------------
# Session State Initialization
# ------------------------------------------------------------------------------
if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame()
if "selected_files" not in st.session_state:
    st.session_state.selected_files = []

# ------------------------------------------------------------------------------
# Fixed Folder Path (as required)
# ------------------------------------------------------------------------------
folder_path = './2510_XML_SAMPLE'


# ------------------------------------------------------------------------------
# XML Parsing Function
# ------------------------------------------------------------------------------
def load_xml_files(selected_files):
    """
    Loads selected XML files and extracts <Sampling> and <Sample> data.
    Each <Sampling> may contain multiple <Sample> entries.
    Sampling attributes + Sample attributes are merged into one row per Sample.

    Parameters:
        selected_files (list[str]): List of XML file names in the folder.

    Returns:
        pd.DataFrame: Flattened XML data.
    """
    all_data = []

    for filename in selected_files:
        file_path = os.path.join(folder_path, filename)

        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
        except Exception as e:
            st.warning(f"Failed to parse {filename}: {e}")
            continue

        for sampling in root.findall('Sampling'):
            parent_attrs = sampling.attrib

            for child in sampling.findall('Sample'):
                row = {**parent_attrs, **child.attrib}
                row['source_file'] = filename
                all_data.append(row)

    return pd.DataFrame(all_data)


# ------------------------------------------------------------------------------
# UI – XML File List
# ------------------------------------------------------------------------------
st.subheader("XML Data Exploration")

# List all .xml files in folder
xml_files = [f for f in os.listdir(folder_path) if f.endswith('.xml')]

select_all = st.checkbox("Select all files")
selected_files = xml_files if select_all else st.multiselect("Select XML files", xml_files)

st.session_state.selected_files = selected_files

# Load temporary dataframe for filtering and summary
temp_df = load_xml_files(selected_files) if selected_files else pd.DataFrame()


# ------------------------------------------------------------------------------
# MAIN TABS
# ------------------------------------------------------------------------------
if not temp_df.empty:

    st.text("Data loaded successfully")

    tab_overview, tab_structure, tab_hierarchy = st.tabs(["Overview", "Structure", "Hierarchy"])

    # --------------------------------------------------------------------------
    # OVERVIEW TAB
    # --------------------------------------------------------------------------
    with tab_overview:
        st.subheader("Column Summary")

        summary_data = []
        for col in temp_df.columns:
            if col == "source_file":
                continue

            unique_vals = temp_df[col].dropna().unique()

            summary_data.append({
                "Column": col,
                "Unique Count": len(unique_vals),
                "Unique Values": ", ".join(map(str, unique_vals[:10])) + ("..." if len(unique_vals) > 10 else "")
            })

        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True)

        # ------------------------ Filtering Controls --------------------------
        st.subheader("Filter Data")

        skip_filter_cols = ["TimeStamp", "Result", "Value", "source_file"]
        filters = {}

        for col in temp_df.columns:
            if col in skip_filter_cols:
                continue

            if temp_df[col].dtype == 'object':
                # Categorical fields -> multiselect
                options = temp_df[col].dropna().unique().tolist()
                selected = st.multiselect(f"Filter {col}", options)
                if selected:
                    filters[col] = selected

            else:
                # Numeric fields -> slider
                min_val, max_val = float(temp_df[col].min()), float(temp_df[col].max())
                range_val = st.slider(f"Filter {col}", min_val, max_val, (min_val, max_val))
                filters[col] = range_val

        # ------------------------ Apply Filters -------------------------------
        filtered_df = temp_df.copy()

        for col, val in filters.items():
            if isinstance(val, list) and val:
                filtered_df = filtered_df[filtered_df[col].isin(val)]
            elif isinstance(val, tuple):
                filtered_df = filtered_df[(filtered_df[col] >= val[0]) & (filtered_df[col] <= val[1])]

        st.subheader("Filtered Data")
        st.dataframe(filtered_df, use_container_width=True)

    # --------------------------------------------------------------------------
    # STRUCTURE TAB – TAG HIERARCHY GRAPH
    # --------------------------------------------------------------------------
    with tab_structure:

        if folder_path and os.path.isdir(folder_path):

            xml_files = [f for f in os.listdir(folder_path) if f.endswith('.xml')]

            if xml_files:
                st.write(f"Found {len(xml_files)} XML files.")

                # Directed graph of XML structure
                G = nx.DiGraph()

                def add_nodes_edges(element, parent=None, prefix=""):
                    node_name = f"{prefix}/{element.tag}"
                    G.add_node(node_name)
                    if parent:
                        G.add_edge(parent, node_name)
                    for child in element:
                        add_nodes_edges(child, node_name, prefix)

                # Build graph for all files
                for xml_file in xml_files:
                    file_path = os.path.join(folder_path, xml_file)

                    try:
                        tree = ET.parse(file_path)
                        root = tree.getroot()
                    except Exception as e:
                        st.warning(f"Cannot parse {xml_file}: {e}")
                        continue

                    add_nodes_edges(root, prefix=xml_file)

                # Create PyVis visualization
                net = Network(height="750px", width="100%", directed=True)
                net.from_nx(G)

                html_path = "xml_graph.html"
                net.save_graph(html_path)

                with open(html_path, 'r', encoding='utf-8') as f:
                    html_content = f.read()

                st.components.v1.html(html_content, height=800, scrolling=True)

            else:
                st.warning("No XML files found in the folder.")

    # --------------------------------------------------------------------------
    # HIERARCHY TAB – SUNBURST CHART
    # --------------------------------------------------------------------------
    with tab_hierarchy:

        if all(col in temp_df.columns for col in ["ID", "Category", "Id"]):

            fig = px.sunburst(
                temp_df,
                path=['ID', 'Category', 'Id'],
                title="XML Data Hierarchy",
                color='Category'
            )

            fig.update_traces(textinfo="label")
            fig.update_layout(
                margin=dict(t=50, l=25, r=25, b=25),
                legend_title_text="Sensor Categories"
            )

            st.plotly_chart(fig, use_container_width=True)

        else:
            st.info("Sunburst requires columns: ID, Category, Id.")