import pandas as pd
import folium
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("stloiusbridges.csv")
df["LATDD"] = df["LATDD"]+0
indexesToDrop = df[df["LATDD"] > 40].index
df.drop(indexesToDrop, inplace=True)

# Print the 'LAT_016' and 'LONG_017' values for each row
for index, row in df.iterrows():
    print(f"Row {index}: y = {row['LATDD']}, x = {row['x']}")

# Static scatter plot of coordinates
latitudes = df["LAT_016"]
longitudes = df["LONG_017"]


# Interactive map using Folium
map_center = [df["y"].mean(), df["x"].mean()]
bridge_map = folium.Map(location=map_center, zoom_start=12)

for _, row in df.iterrows():
    lat = row["LATDD"]
    lon = row["LONGDD"]
    if pd.notnull(lat) and pd.notnull(lon):
        folium.CircleMarker(
            location=[lat, lon],
            radius=3,
            color='blue',
            fill=True,
            fill_opacity=0.6
        ).add_to(bridge_map)

# Save the map to an HTML file
bridge_map.save("st_louis_bridges_map.html")
print("Interactive map saved as 'st_louis_bridges_map.html'.")

#print the bridges to a file
with open("st_louis_bridges.txt", "w") as f:
    f.write("x,y,classes\n")
    for index, row in df.iterrows():
        f.write(f"{row['LONGDD']},{row['LATDD']},1\n")


