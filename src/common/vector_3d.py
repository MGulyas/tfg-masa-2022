from math import sqrt


class Vector3D:
    # Initializer
    def __init__(self, x_element, y_element, z_element):
        self.x = x_element
        self.y = y_element
        self.z = z_element

    # Operator Overloading
    def __sub__(self, v):
        return Vector3D(self.x - v.x, self.y - v.y, self.z - v.z)

    def __add__(self, v):
        return Vector3D(self.x + v.x, self.y + v.y, self.z + v.z)

    def __mul__(self, s):
        return Vector3D(self.x * s, self.y * s, self.z * s)

    def __truediv__(self, s):
        return Vector3D(self.x / s, self.y / s, self.z / s)

    def __repr__(self):
        return f'Vector3D({self.x}, {self.y}, {self.z})'


# Return dot product between two vectors
def Dot(a, b):
    return a.x * b.x + a.y * b.y + a.z * b.z


# Return perpendicular vector
def Cross(a, b):
    return Vector3D(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x)


# Return length of vector
def Length(v):
    return sqrt(v.x * v.x + v.y * v.y + v.z * v.z)


# Return normalized vector (unit vector)
def Normalize(v):
    return v * (1.0 / Length(v))


# Return normal that is pointing on the side as the passed direction
def orient_normal(normal, direction):
    if Dot(normal, direction) < 0.0:
        return normal * -1.0  # flip normal
    else:
        return normal

def distance(a, b):
    return sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z + b.z) ** 2)


def direction_vector(origin, end):
    return Normalize(Vector3D(end.x - origin.x, end.y - origin.y, end.z - origin.z))
