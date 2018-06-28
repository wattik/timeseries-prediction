from PIL import Image, ImageDraw
import numpy as np


def careful_divide(vector, divider):
    if isinstance(divider, np.ndarray):
        where_nonzeros = divider != 0
        outcome = np.zeros(divider.shape)
        if isinstance(vector, np.ndarray):
            outcome[where_nonzeros] = vector[where_nonzeros] / divider[where_nonzeros]
        else:
            outcome[where_nonzeros] = vector / divider[where_nonzeros]
    else:
        if divider != 0:
            outcome = vector / divider
        else:
            if isinstance(vector, np.ndarray):
                outcome = np.zeros(vector.shape)
            else:
                outcome = 0
    return outcome


def _quantile_norm(vector, quantiles):
    scores = np.percentile(vector, quantiles)
    a = scores[0]
    if a == 0:
        a = 1
    b = scores[1]
    if (b == vector).all():
        b = 1
    return careful_divide(vector - b, a - b)


def q95_norm(vector):
    return _quantile_norm(vector, [95, 5])


def q90_norm(vector):
    return _quantile_norm(vector, [90, 10])


def sort_arrays(a, b):
    indices_sorted = np.argsort(-a)
    return a[indices_sorted], b[indices_sorted]


class BarsDrawer(object):
    # Widths and Heights in pixels
    l_column = 70
    l_row = 1
    l_header = 17
    # NumberColor setting
    max_green = 100

    def __init__(self, target, prediction, normalizer=q95_norm):
        assert len(target) == len(prediction)

        prediction, target = sort_arrays(prediction.values.ravel(), target.values.ravel())
        self.normalize = normalizer
        self.target = target
        self.prediction = prediction
        self.image = self._make_image()

    def _size(self):
        width = 2 * self.l_column
        height = self.l_header + len(self.target) * self.l_row
        return width, height

    def _draw_header(self, image: Image):
        draw = ImageDraw.Draw(image)
        draw.text(
            (5, 2), "Prediction", "black"
        )
        draw.text(
            (self.l_column + 5, 2), "Target", "black"
        )

    def _draw_column(self, draw: ImageDraw, offset: int, vector):
        vector = self.normalize(vector.copy())
        for line, element in enumerate(vector):
            xy = [(offset, self.l_header + line), (offset + self.l_column, self.l_header + line + (self.l_row - 1))]
            green_intensity = (1 - element)

            if green_intensity > 1:
                green_intensity = 1
            elif green_intensity < 0:
                green_intensity = 0

            color = (0, int((255 - self.max_green) * green_intensity + self.max_green), 0)
            draw.rectangle(xy, fill=color)

    def _draw_rows(self, image: Image):
        draw = ImageDraw.Draw(image)
        self._draw_column(draw, 0, self.prediction)
        self._draw_column(draw, self.l_column, self.target)

    def _make_image(self) -> Image:
        size = self._size()
        image = Image.new("RGB", size, "white")
        self._draw_header(image)
        self._draw_rows(image)
        return image

    def save(self, path):
        self.image.save(path, format="PNG")

    def show(self):
        self.image.show()
