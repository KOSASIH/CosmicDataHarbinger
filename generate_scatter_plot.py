import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# Scatter plot
def generate_scatter_plot(x, y, title, x_label, y_label):
    plt.scatter(x, y)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()

# Heatmap
def generate_heatmap(data, title):
    sns.heatmap(data)
    plt.title(title)
    plt.show()

# 3D representation
def generate_3d_plot(x, y, z, title):
    fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z, mode='markers')])
    fig.update_layout(title=title)
    fig.show()

# Example usage
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]
z = [3, 6, 9, 12, 15]
data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

generate_scatter_plot(x, y, "Scatter Plot", "X", "Y")
generate_heatmap(data, "Heatmap")
generate_3d_plot(x, y, z, "3D Plot")
