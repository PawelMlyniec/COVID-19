import streamlit as st
import pydeck as pdk
from typing import List
import pandas as pd
import os


class DataViewer():
    def __init__(self) -> None:
        self._coords = []
        self._data = []

    def import_data(self, node_ids: List, outputs: List, labels: List, mean: float, std: float) -> None:
        print("in import")
        days = len(outputs)
        local_data_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "county_coords.csv")
        df = pd.read_csv(local_data_path)

        coords = []
        for row in node_ids:
            c = (df.loc[(df['county'] == row[0]) & (df['state'] == row[1])][['Lon','Lat','county', 'state']]).values.tolist()
            coords.append(c[0])
        self._coords = pd.DataFrame(coords,columns=['lon','lat','county', 'state'])

        print('out coords')
        for day_output, day_label in zip(outputs, labels):
            data = pd.DataFrame()
            data['output'] = day_output
            data['labels'] = day_label
            data['labels_real'] = data['labels'].apply(lambda x: int(x*std + mean))
            data['output_real'] = data['output'].apply(lambda x: int(x*std + mean))
            self._data.append(data)

        self._days = days
        print("out import")

    def view_plot(self) -> None:
        st.title('Covid cases')
        
        day = st.slider('Show data from day:', 0, self._days-1)
        print(day)
        view = pdk.data_utils.compute_view(self._coords[["lon", "lat"]])
        view.pitch = 75
        plot_result = self._data[day].join(self._coords)
        print("in view")
        column_layer = pdk.Layer(
            "ColumnLayer",
            data=plot_result,
            get_position=["lon", "lat"],
            get_elevation="output_real",
            elevation_scale=500,
            radius=5000,
            get_fill_color=['255*(output+1)/2' ,'255*(1-(output+1)/2)', 0],
            get_line_color=[255, 255, 255],
            pickable=True,
            auto_highlight=True,
        )
        layer = pdk.Layer(
            "ScatterplotLayer",
            plot_result,
            pickable=True,
            opacity=0.8,
            stroked=True,
            filled=True,
            radius_scale=50,
            radius_min_pixels=7,
            radius_max_pixels=80,
            line_width_min_pixels=1,
            get_position=["lon", "lat"],
            get_radius="output_real",
            get_fill_color=['255*(output+1)/2' ,'255*(1-(output+1)/2)', 0],
            get_line_color=[0, 0, 0],
        )

        tooltip = {
            "html": "<b>{county}, {state}</b> Predicted: {output_real}, Real: {labels_real}",
            "style": {"background": "grey", "color": "white", "font-family": '"Helvetica Neue", Arial', "z-index": "10000"},
        }
        if st.button('Change layout'):
            r = pdk.Deck(
                map_style="mapbox://styles/mapbox/light-v9",
                layers=[column_layer],
                initial_view_state=view,
                tooltip=tooltip,
            )
        else:
            r = pdk.Deck(
                map_style="mapbox://styles/mapbox/light-v9",
                layers=[layer],
                initial_view_state=view,
                tooltip=tooltip,
            )

        st.pydeck_chart(r)
