import pandas as pd
import plotly.express as px
import streamlit as st
from datetime import datetime, timedelta

st.set_page_config(layout="wide")

def styled_header(text: str, color: str = "#00126E", size: str = "24px"):
        st.markdown(f"<h4 style='color:{color}; font-size:{size};'>{text}</h4>", unsafe_allow_html=True)

def styled_subheader(text: str, color: str = "#0B5ED7", size: str = "18px"):
    st.markdown(f"<h3 style='color:{color}; font-size:{size};'>{text}</h3>", unsafe_allow_html=True)


# Example usage
styled_header("Sensor data analysis model: Power regulators use case")

data = pd.read_csv("C:/Users/320303731/OneDrive - Philips/Documents/PREG_MD.txt", sep = ",")
data['SamplingDate'] = pd.to_datetime(data['SamplingDate'])
data['uniqueID'] = data['SampleID'].astype(str) + "_" + data['CatalogNumber'].astype(str) + "_" + data['SerialNumber'].astype(str)
df = data.drop_duplicates()

##df = df[df['SerialNumber']=='US119B0591']

def check_alert(group):
    # Ensure datetime and sort
    group = group.drop_duplicates()
    group['SamplingTimestamp'] = pd.to_datetime(group['SamplingTimestamp'])
    group = group.sort_values('SamplingTimestamp').set_index('SamplingTimestamp')

    # Boolean mask for SampleValue <= threshold
    mask = (group['SampleValue'] <= t).astype(int)

    # Rolling count over 48-hour window
    rolling_counts = mask.rolling(f'{w}h').sum()

    # If any rolling window has at least 2 occurrences, flag alert
    return (rolling_counts >= l).any()


tab1, tab2, tab3 = st.tabs(["Alerts by factory defaults", "Alerts by recommended values","SampleValue distribution"])

with tab1:
    t=1026
    w=2
    l=2

    color_map = ['#02ABB1', '#0B5ED7', '#00126E', '#008800', '#00666F']
    # Apply to each uniqueID - No partition filter for graph
    systems = df.groupby('uniqueID').apply(check_alert, include_groups=False)
    alerts = systems[systems == True]
    count_df = df.loc[df['uniqueID'].isin(alerts.index)].drop_duplicates().groupby('SampleID')['uniqueID'].nunique().reset_index(name='Count')

    fig_counts = px.bar(count_df, x='SampleID', y='Count', color='SampleID',color_discrete_sequence = color_map,text_auto=True,
                        title='Number of systems reporting Fan speed threshold or less (2025/12/16)' ) #px.colors.sequential.Blues
    fig_counts.update_layout(showlegend=False)
    st.plotly_chart(fig_counts, key="countSystems_gen")

    # Filter to select partition
    selected_partition_tab1 = st.selectbox("Select PREG to see detail:", df['SampleID'].sort_values().unique(), key='key_tab1')
    DF_df = df[df['SampleID'] == selected_partition_tab1].copy()

    alerts_calc = DF_df.groupby('uniqueID').apply(check_alert, include_groups=False)
    alerts_calc = alerts_calc[alerts_calc == True]
    alerts_DF = DF_df.loc[(DF_df['uniqueID'].isin(alerts_calc.index)) & (DF_df['SampleValue'] <= t)].drop_duplicates()

    list_devices = alerts_DF[['SampleID','CatalogNumber','SerialNumber','uniqueID']].drop_duplicates().sort_values(by='uniqueID')
    list_devices['#'] = range(1, len(list_devices) + 1)

    styled_subheader("Systems reporting alert using factory default values:")
    st.dataframe(list_devices[['#','SampleID','CatalogNumber','SerialNumber','uniqueID']], hide_index=True)

    ## Graphs per system
    styled_subheader("Timeline per system")

    for uid in list_devices['uniqueID']:
        subset = df[df['uniqueID'] == uid].copy()
        subset['SamplingTimestamp'] = pd.to_datetime(subset['SamplingTimestamp'])

        # Create the line chart
        fig = px.scatter(
            subset,
            x='SamplingTimestamp',
            y='SampleValue',
            title=f'Fan speed timeline for Power Regulator in system {uid}',
            #line_shape='hv',
        )
        fig.update_traces(marker=dict(size=3,symbol="diamond",color='#00126E'))
        fig.update_layout(xaxis_autorange=True,yaxis_autorange=True)
        fig.update_xaxes(ticks="inside", tickangle=270)
        #fig.update_yaxes(autorange="reversed") to add in case y-axe need to be displayed decreasing instead increasing
        # Add threshold line
        fig.add_hline(
            y=1026,
            line_color="#F85569",
            annotation_text=f"Alert threshold: {1026}",
            annotation_position="top left",
            layer="below",
            line_width=4
        )

        # Display in Streamlit
        st.plotly_chart(fig)

        # Configuring the data frame of the system to be downloaded as csv
        csv = subset[subset['SampleValue'] <= t].drop_duplicates().to_csv(index=False)
        key_dfd = f"download_{uid}"
        file_name = f"{uid}_data.csv"

        with st.expander("System alert data"):
            st.dataframe(
                subset.loc[subset['SampleValue'] <= t, ['uniqueID', 'SamplingTimestamp', 'SampleValue', 'SampleUnits','CatalogNumber', 'SerialNumber']]
                .drop_duplicates()
                .reset_index(drop=True),
                width='stretch'
            )
            st.download_button(label="Download CSV", data=csv, file_name=file_name, mime="text/csv", icon=":material/download:", key=key_dfd)

with tab2:
    # Dictionaries for dynamic thresholds
    value_thresholds = {
        'PREG_FAN_1': 1026,
        'PREG_FAN_2': 1053,
        'PREG_FAN_3': 1053
    }
    window_hours = {
        'PREG_FAN_1': 2,
        'PREG_FAN_2': 2,
        'PREG_FAN_3': 2
    }
    limit = {
        'PREG_FAN_1': 3,
        'PREG_FAN_2': 3,
        'PREG_FAN_3': 3
    }

    # Select partition
    selected_partition_tab2 = st.selectbox("Select PREG:",df['SampleID'].sort_values().unique(),key='key_tab2')

    w = window_hours.get(selected_partition_tab2, "N/A")
    l = limit.get(selected_partition_tab2, "N/A")
    t = value_thresholds.get(selected_partition_tab2, "N/A")

    REC_df = df[df['SampleID'] == selected_partition_tab2].copy()
    # REC_df = df[df['Sample'] == 'R:'].copy()


    alerts_calcR = REC_df.groupby('uniqueID').apply(check_alert, include_groups=False)
    alerts_calcR = alerts_calcR[alerts_calcR == True]
    alerts_REC = REC_df.loc[(REC_df['uniqueID'].isin(alerts_calcR.index)) & (REC_df['SampleValue'] <= t)].drop_duplicates()
    # alerts_DF['SamplingTimestamp'] = pd.to_datetime(alertsDF['SamplingTimestamp'])

    list_devicesREC = alerts_REC[['SampleID', 'CatalogNumber', 'SerialNumber', 'uniqueID']].drop_duplicates().sort_values(by='uniqueID')
    list_devicesREC['#'] = range(1, len(list_devicesREC) + 1)

    st.text("Values per partition selected:")
    m1, m2, m3, m4 = st.columns(4)

    m1.metric('Threshold (%)', t, border=True)
    m2.metric('Time window (hours)', w, border=True)
    m3.metric('Limit', l, border=True)
    m4.metric('Number of systems', len(list_devicesREC), border=True)

    styled_subheader("Systems reporting alert using recommended values:")
    st.dataframe(list_devicesREC[['#', 'SampleID', 'CatalogNumber', 'SerialNumber', 'uniqueID']], hide_index=True)

    styled_subheader("Timeline per system")
    for uid in list_devicesREC['uniqueID']:
        subset_rec = df[df['uniqueID'] == uid].copy()
        subset_rec['SamplingTimestamp'] = pd.to_datetime(subset_rec['SamplingTimestamp'])

        # Create the line chart
        fig_rec = px.scatter(
            subset_rec,
            x='SamplingTimestamp',
            y='SampleValue',
            title=f'Fan speed timeline for Power Regulator in system {uid}')
        fig_rec.update_layout(
            title_font=dict(size=24),  # Title font size
            font=dict(size=16),  # General font size (axes, legend, etc.)
            legend=dict(font=dict(size=14)),  # Legend font size
            xaxis_title_font=dict(size=18),  # X-axis title font size
            yaxis_title_font=dict(size=18)  # Y-axis title font size
        )
        fig_rec.update_traces(marker=dict(size=3,  symbol="diamond", color='#A80DF2'))
        fig_rec.update_xaxes(ticks="inside", tickangle=270)
        fig_rec.update_layout(xaxis_autorange=True,yaxis_autorange=True,
                          font=dict(size=16)
                              )
        #fig.update_yaxes(autorange="reversed")
        fig_rec.add_hline(y=t,line_color="#F62FBF",
                          annotation_text=f"Alert threshold: {t}",
                          annotation_position="top left",
                          layer="below",
                          line_width=4)
        st.plotly_chart(fig_rec)

        csv = subset[subset['SampleValue'] <= t].drop_duplicates().to_csv(index=False)
        key_download_rec = f"download_r{uid}"
        file_name = f"{uid}_data.csv"

        with st.expander("System alert data"):
            st.dataframe(
                subset.loc[subset['SampleValue'] <= t, ['uniqueID', 'SamplingTimestamp', 'SampleValue', 'SampleUnits',
                                                        'CatalogNumber', 'SerialNumber']]
                .drop_duplicates()
                .reset_index(drop=True),
                width='stretch'
            )
            st.download_button(label="Download CSV", data=csv, file_name=file_name, mime="text/csv",
                               icon=":material/download:", key=key_download_rec)


with tab3:

    # Dictionaries for dynamic thresholds
    value_thresholds = {
        'PREG_FAN_1': 1026,
        'PREG_FAN_2': 1053,
        'PREG_FAN_3': 1053
    }
    selected_partition_tab3 = st.selectbox("Select PREG:", df['SampleID'].sort_values().unique(), key='key_tab3')
    t = value_thresholds.get(selected_partition_tab3, "N/A")

    df_distro = df[df['SampleID'] == selected_partition_tab3].copy()

    m1, m2, m3, m4 = st.columns(4)

    m1.metric('Threshold (RPM)', t, border=True)
    m2.metric('Min value (RPM)', df_distro['SampleValue'].min(skipna=True, numeric_only=False), border=True)
    m3.metric('Max value (RPM)', df_distro['SampleValue'].max(skipna=True, numeric_only=False), border=True)
    m4.metric('Median value (RPM)', df_distro['SampleValue'].median(skipna=True, numeric_only=False), border=True)

    fig_SV = px.histogram(
        df_distro,
        x='SampleValue',
        title=f"Distribution of SampleValue for PREG: {selected_partition_tab3}",
    )

    # Add vertical lines at 1023 and 1056
    fig_SV.add_vline(
        x=t,
        line_color="red",
        line_width=2,
        line_dash="dash",
        annotation_text=t,
        annotation_position="top right"
    )
    fig_SV.update_layout(
        # Optional: make sure the annotations don't overlap the title
        margin=dict(t=80)
    )
    st.plotly_chart(fig_SV)