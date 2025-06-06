import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np
import torch
from torch import nn
import os
import sys


class MNISTMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 526),
            nn.ReLU(),
            nn.Linear(526, 126),
            nn.ReLU(),
            nn.Linear(126, 10)
        )

    def forward(self, x):
        return self.layers(x)


class DigitRecognizer:
    def __init__(self):
        self.model = MNISTMLP()

        if getattr(sys, 'frozen', False):
            base_path = sys._MEIPASS
        else:
            base_path = os.path.dirname(os.path.abspath(__file__))

        model_path = os.path.join(base_path, 'mnist_mlp.pth')
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.model.eval()

        self.window = tk.Tk()
        self.window.title("手写数字识别器")

        self.canvas = tk.Canvas(self.window, width=280, height=280, bg='white')
        self.canvas.pack()

        self.image = Image.new('L', (280, 280), 'white')
        self.draw = ImageDraw.Draw(self.image)

        self.label = tk.Label(self.window, text="请绘制数字", font=('Arial', 24))
        self.label.pack()

        self.btn_frame = tk.Frame(self.window)
        self.btn_frame.pack()

        tk.Button(self.btn_frame, text="识别", command=self.predict).pack(side=tk.LEFT)
        tk.Button(self.btn_frame, text="清除", command=self.clear).pack(side=tk.LEFT)

        self.canvas.bind("<B1-Motion>", self.draw_line)

        self.last_x = None
        self.last_y = None

        self.canvas.bind("<Button-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw_line)
        self.canvas.bind("<ButtonRelease-1>", self.reset_last)

    def start_draw(self, event):
        self.last_x, self.last_y = event.x, event.y
        self.draw_line(event)

    def draw_line(self, event):
        if self.last_x is None or self.last_y is None:
            self.last_x, self.last_y = event.x, event.y
            return

        distance = ((event.x - self.last_x)**2 + (event.y - self.last_y)**2)**0.5
        steps = int(distance / 2) + 1

        for i in range(steps):
            x = self.last_x + (event.x - self.last_x) * i / steps
            y = self.last_y + (event.y - self.last_y) * i / steps
            self.draw_point(x, y)

        self.last_x, self.last_y = event.x, event.y

    def draw_point(self, x, y):
        r = 8
        self.canvas.create_oval(x-r, y-r, x+r, y+r,
                               fill='black', outline='black', width=0)
        self.draw.ellipse([x-r, y-r, x+r, y+r], fill='black')

    def reset_last(self, event):
        self.last_x = None
        self.last_y = None

    def preprocess(self):
        img = self.image.resize((28, 28))
        img = np.array(img).astype(np.float32)
        img = 255 - img
        img = img / 255.0
        return torch.tensor(img.flatten()).unsqueeze(0).float()

    def predict(self):
        tensor = self.preprocess()
        with torch.no_grad():
            outputs = self.model(tensor)
            pred = torch.argmax(outputs).item()
        self.label.config(text=f"识别结果：{pred}")

    def clear(self):
        self.canvas.delete("all")
        self.image = Image.new('L', (280, 280), 'white')
        self.draw = ImageDraw.Draw(self.image)
        self.label.config(text="请绘制数字")

    def run(self):
        self.window.mainloop()


if __name__ == "__main__":
    app = DigitRecognizer()
    try:
        app.run()
    except Exception as e:
        with open("error.log", "w") as f:
            f.write(str(e))
