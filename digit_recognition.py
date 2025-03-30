# REVISED code completed 30/3/25

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tensorflow import keras
from PIL import Image, ImageDraw
import numpy as np
import os

# binary shit lol (does not work)
def load_and_preprocess_data():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_train = (x_train > 128).astype(np.float32)
    x_test = (x_test > 128).astype(np.float32)

    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    return (x_train, y_train), (x_test, y_test)

# main
def build_model():
    model = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model

#training
def train_and_evaluate():
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()
    model = build_model()
    model.summary()
    model.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.1)
    model.evaluate(x_test, y_test, verbose=2)
    model.save("digit_recognition_model_binary.h5")
    messagebox.showinfo("Training Complete", "Model trained and saved as 'digit_recognition_model_binary.h5'")

# reading the img
def preprocess_image(img_path):
    img = Image.open(img_path).convert('L')
    img = img.resize((28, 28))
    img_array = np.array(img) / 255.0
    img_array = (img_array > 0.5).astype(np.float32)
    img_array = img_array.reshape(1, 28, 28, 1)
    return img_array

# binary shit (does not work)
def preprocess_drawing(input_grid):
    img = Image.new('L', (28, 28), color=255)
    draw = ImageDraw.Draw(img)
    for i in range(28):
        for j in range(28):
            if input_grid[i][j] == 1:
                draw.rectangle([j, i, j, i], fill=0)
    img_array = np.array(img) / 255.0
    img_array = (img_array > 0.5).astype(np.float32)
    img_array = img_array.reshape(1, 28, 28, 1)
    return img_array

class MinimalDrawingApp:
    def __init__(self, root):
        self.root = root
        self.root.configure(bg="#111111")
        self.input_grid = [[0 for _ in range(28)] for _ in range(28)]

        self.create_main_ui()

    # u and i
    def create_main_ui(self):
        self.title_label = tk.Label(self.root, text="demon's CNN", font=("Helvetica", 24), fg="white", bg="#111111")
        self.title_label.pack(pady=20)

        self.upload_button = self.create_circular_button("Upload Image", self.upload_image)
        self.upload_button.pack(pady=10)

        self.draw_button = self.create_circular_button("Draw Number", self.open_drawing_grid)
        self.draw_button.pack(pady=10)

        self.train_button = self.create_circular_button("Train Model", train_and_evaluate)
        self.train_button.pack(pady=10)

        self.result_label = tk.Label(self.root, text="", font=("Helvetica", 16), fg="white", bg="#111111")
        self.result_label.pack(pady=20)

    def create_circular_button(self, text, command):
        style = ttk.Style()
        style.configure("TButton", font=("Helvetica", 14), padding=10, relief="flat", borderwidth=0)
        style.map("TButton", background=[("active", "#444444")])

        button = ttk.Button(self.root, text=text, command=command, style="TButton")
        return button

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp")])
        if file_path:
            self.predict_image(file_path)

    def predict_image(self, img_path):
        if os.path.exists("digit_recognition_model_binary.h5"):
            model = keras.models.load_model("digit_recognition_model_binary.h5")
        else:
            messagebox.showinfo("Error", "Model not found! Please train the model first.")
            return

        img_array = preprocess_image(img_path)
        predictions = model.predict(img_array)
        predicted_label = np.argmax(predictions)
        self.result_label.config(text=f"Prediction: {predicted_label}")

    def open_drawing_grid(self):
        grid_window = tk.Toplevel(self.root)
        grid_window.title("Draw a Number (doesnt work LMAO)")
        grid_window.configure(bg="#111111")
        grid_frame = tk.Frame(grid_window, bg="#111111")
        grid_frame.pack(pady=10)

        self.canvas_buttons = [[None for _ in range(28)] for _ in range(28)]
        for i in range(28):
            for j in range(28):
                cell = tk.Button(grid_frame, width=2, height=1, bg="white", relief="flat",
                                 command=lambda x=i, y=j: self.toggle_cell(x, y))
                cell.grid(row=i, column=j, padx=1, pady=1)
                self.canvas_buttons[i][j] = cell

        submit_button = self.create_circular_button("Submit", lambda: self.submit_drawing(grid_window))
        submit_button.pack(pady=10)

    def toggle_cell(self, x, y):
        if self.input_grid[x][y] == 0:
            self.input_grid[x][y] = 1
            self.canvas_buttons[x][y].config(bg="black")
        else:
            self.input_grid[x][y] = 0
            self.canvas_buttons[x][y].config(bg="white")

    def submit_drawing(self, window):
        if os.path.exists("digit_recognition_model_binary.h5"):
            model = keras.models.load_model("digit_recognition_model_binary.h5")
        else:
            messagebox.showinfo("Error", "Model not found! Please train the model first.")
            return

        img_array = preprocess_drawing(self.input_grid)
        predictions = model.predict(img_array)
        predicted_label = np.argmax(predictions)
        self.result_label.config(text=f"Prediction: {predicted_label}")
        window.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Demon's NN")
    app = MinimalDrawingApp(root)
    root.mainloop()
