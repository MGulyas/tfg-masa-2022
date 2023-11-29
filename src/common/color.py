class RGBColor:
    # Initializer
    def __init__(self, red, green, blue):
        self.r = red
        self.g = green
        self.b = blue

    # Operator Overloading
    def __add__(self, c):
        return RGBColor(self.r + c.r, self.g + c.g, self.b + c.b)

    def __sub__(self, c):
        return RGBColor(self.r - c.r, self.g - c.g, self.b - c.b)

    def __mul__(self, s):
        return RGBColor(self.r * s, self.g * s, self.b * s)

    def __truediv__(self, s):
        return RGBColor(self.r / s, self.g / s, self.b / s)

    # this alows us to multipy by another RGBColour
    def multiply(self, c):
        return RGBColor(self.r * c.r, self.g * c.g, self.b * c.b)

    # Member Functions
    def clamp(self, minimum, maximum):
        # red
        if (self.r > maximum): self.r = maximum
        if (self.r < minimum): self.r = minimum
        # green
        if (self.g > maximum): self.g = maximum
        if (self.g < minimum): self.g = minimum
        # blue
        if (self.b > maximum): self.b = maximum
        if (self.b < minimum): self.b = minimum

    def blend(self, colors):
        colors = list(filter(lambda item: item is not None, colors))
        blend = RGBColor(self.r, self.g, self.b)
        n = len(colors)
        for color in colors:
            blend.r += color.r
            blend.g += color.g
            blend.b += color.b
        blend.r = blend.r / n
        blend.g = blend.g / n
        blend.b = blend.b / n
        return blend

    def __repr__(self):
        return f'RGBColor({self.r}, {self.g}, {self.b})'

BLACK = RGBColor(0.0, 0.0, 0.0)
WHITE = RGBColor(1.0, 1.0, 1.0)
GREEN = RGBColor(0.0, 1.0, 0.0)
RED = RGBColor(1.0, 0.0, 0.0)