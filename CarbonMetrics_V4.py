import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


class ModelGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Random Forest Model GUI")
        self.csv_file = None
        self.model = None
        self.df = None
        self.results = {}
        self.tiff_files = {}  # Dictionary to store TIFF file paths for each independent variable

        # GUI Components
        self.load_csv_button = tk.Button(root, text="Load CSV File", command=self.load_csv)
        self.load_csv_button.pack(pady=10)

        self.dependent_label = tk.Label(root, text="Select Dependent Variable:")
        self.dependent_label.pack(pady=5)
        self.dependent_combobox = ttk.Combobox(root, state="disabled")
        self.dependent_combobox.pack(pady=5)

        self.independent_label = tk.Label(root, text="Select Independent Variables:")
        self.independent_label.pack(pady=5)
        self.independent_listbox = tk.Listbox(root, selectmode=tk.MULTIPLE, height=10)
        self.independent_listbox.pack(pady=5)

        self.train_model_button = tk.Button(root, text="Train Model", command=self.train_model, state=tk.DISABLED)
        self.train_model_button.pack(pady=10)

        self.tiff_frame = tk.Frame(root)
        self.tiff_frame.pack(pady=10)

        self.run_model_button = tk.Button(root, text="Run Model", command=self.run_model, state=tk.DISABLED)
        self.run_model_button.pack(pady=10)

        self.visualize_button = tk.Button(root, text="Visualize Output", command=self.visualize_output, state=tk.DISABLED)
        self.visualize_button.pack(pady=10)

        self.save_button = tk.Button(root, text="Save Output as TIFF", command=self.save_as_tiff, state=tk.DISABLED)
        self.save_button.pack(pady=10)

        self.status_label = tk.Label(root, text="Status: Waiting for input")
        self.status_label.pack(pady=10)

    def load_csv(self):
        self.csv_file = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if self.csv_file:
            self.df = pd.read_csv(self.csv_file)
            columns = self.df.columns.tolist()

            # Populate dependent variable combobox
            self.dependent_combobox.config(values=columns, state="normal")

            # Populate independent variables listbox
            self.independent_listbox.delete(0, tk.END)
            for col in columns:
                self.independent_listbox.insert(tk.END, col)

            self.status_label.config(text="CSV Loaded. Select Variables and Train the Model.")
            self.train_model_button.config(state=tk.NORMAL)
        else:
            messagebox.showerror("Error", "No file selected.")

    def train_model(self):
        if self.df is not None:
            dependent_var = self.dependent_combobox.get()
            independent_vars = [self.independent_listbox.get(idx) for idx in self.independent_listbox.curselection()]

            if not dependent_var or not independent_vars:
                messagebox.showerror("Error", "Please select both dependent and independent variables.")
                return

            X = self.df[independent_vars]
            y = self.df[dependent_var]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            self.model = RandomForestRegressor()
            self.model.fit(X_train, y_train)
            score = self.model.score(X_test, y_test)

            self.status_label.config(text=f"Model Trained. R^2 Score: {score:.2f}")
            self.generate_tiff_inputs(independent_vars)
        else:
            messagebox.showerror("Error", "Load a CSV file first.")

    def generate_tiff_inputs(self, independent_vars):
        # Clear previous TIFF input widgets
        for widget in self.tiff_frame.winfo_children():
            widget.destroy()

        # Create file input widgets for each independent variable
        self.tiff_files = {}
        for var in independent_vars:
            frame = tk.Frame(self.tiff_frame)
            frame.pack(pady=2)

            label = tk.Label(frame, text=f"{var}:")
            label.pack(side=tk.LEFT, padx=5)

            entry = tk.Entry(frame, width=30)
            entry.pack(side=tk.LEFT, padx=5)

            button = tk.Button(frame, text="Browse", command=lambda v=var, e=entry: self.browse_tiff(v, e))
            button.pack(side=tk.LEFT, padx=5)

            self.tiff_files[var] = entry

        self.run_model_button.config(state=tk.NORMAL)

    def browse_tiff(self, variable, entry_widget):
        file_path = filedialog.askopenfilename(filetypes=[("TIFF Files", "*.tif")])
        if file_path:
            entry_widget.delete(0, tk.END)
            entry_widget.insert(0, file_path)

    def run_model(self):
        if self.model:
            # Collect TIFF file paths
            tiff_paths = {}
            for var, entry in self.tiff_files.items():
                file_path = entry.get()
                if not file_path:
                    messagebox.showerror("Error", f"No file selected for {var}.")
                    return
                tiff_paths[var] = file_path

            # Read and process TIFF files
            self.results = {}
            tiff_data_list = []
            for var, file_path in tiff_paths.items():
                with rasterio.open(file_path) as src:
                    tiff_data = src.read(1)
                    tiff_data_list.append(tiff_data.flatten())

            # Stack all TIFF data into a single 2D array (rows: pixels, columns: TIFF files)
            tiff_stack = np.stack(tiff_data_list, axis=1)

            # Identify valid pixels (non-NaN in all layers)
            valid_indices = np.all(~np.isnan(tiff_stack), axis=1)
            valid_data = tiff_stack[valid_indices]

            # Create a DataFrame for valid pixels
            feature_df = pd.DataFrame(valid_data, columns=list(tiff_paths.keys()))

            # Predict values for valid pixels
            predictions = np.empty(tiff_stack.shape[0])
            predictions[:] = np.nan  # Initialize with NaN
            predictions[valid_indices] = self.model.predict(feature_df)

            # Reshape predictions to match the original TIFF shape
            result_shape = tiff_data_list[0].shape
            result_data = predictions.reshape(result_shape)

            # Save the results for visualization
            self.results["Prediction"] = result_data

            self.visualize_button.config(state=tk.NORMAL)
            self.save_button.config(state=tk.NORMAL)
            self.status_label.config(text="Model Run Complete. Visualize or Save the Outputs.")
        else:
            messagebox.showerror("Error", "Train the model first.")


    def visualize_output(self):
        if self.results:
            result_data = self.results.get("Prediction")
            if result_data is not None:
                try:
                    # Use the shape of the first TIFF file provided in the dictionary
                    first_tiff_path = list(self.tiff_files.values())[0].get()  # Get the first TIFF file path
                    with rasterio.open(first_tiff_path) as src:
                        original_shape = src.read(1).shape
    
                    # Reshape the result data to match the original TIFF shape
                    reshaped_data = result_data.reshape(original_shape)
    
                    # Visualize the reshaped data
                    fig, ax = plt.subplots(figsize=(10, 6))
                    im = ax.imshow(reshaped_data, cmap='viridis')
                    plt.colorbar(im, ax=ax)
                    ax.set_title("Prediction Output")
                    plt.show()
                except Exception as e:
                    messagebox.showerror("Error", f"Error during visualization: {e}")
            else:
                messagebox.showerror("Error", "No prediction results found.")
        else:
            messagebox.showerror("Error", "Run the model first.")
    

    def save_as_tiff(self):
        if self.results:
            result_data = self.results.get("Prediction")
            if result_data is not None:
                save_path = filedialog.asksaveasfilename(defaultextension=".tif", filetypes=[("TIFF Files", "*.tif")])
                if save_path:
                    try:
                        # Ensure there is at least one TIFF file loaded
                        first_tiff_path = list(self.tiff_files.values())[0].get()
                        if not first_tiff_path:
                            raise ValueError("No TIFF file path found.")
    
                        # Read the first TIFF file to extract metadata
                        with rasterio.open(first_tiff_path) as src:
                            transform = src.transform
                            crs = src.crs
                            width = src.width
                            height = src.height
                            dtype = result_data.dtype
    
                        # Reshape result_data to 2D (height x width) if it's 1D
                        if result_data.ndim == 1:
                            result_data = result_data.reshape((height, width))
    
                        # Save the prediction results as a TIFF file
                        with rasterio.open(
                                save_path, 'w',
                                driver='GTiff',
                                height=height,
                                width=width,
                                count=1,
                                dtype=dtype,
                                crs=crs,
                                transform=transform) as dst:
                            dst.write(result_data, 1)
    
                        self.status_label.config(text="Output saved successfully.")
                        messagebox.showinfo("Success", "Output saved as TIFF file.")
                    except Exception as e:
                        messagebox.showerror("Error", f"Error saving TIFF file: {e}")
                else:
                    messagebox.showwarning("Warning", "No file path specified.")
            else:
                messagebox.showerror("Error", "No prediction results to save.")
        else:
            messagebox.showerror("Error", "Run the model first.")




# Run the GUI
if __name__ == "__main__":
    root = tk.Tk()
    app = ModelGUI(root)
    root.mainloop()
    

