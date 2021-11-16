import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib import animation
from itertools import combinations
import sympy as sym


class SingletonMeta(type):
    """
    The Singleton class can be implemented in different ways in Python. Some
    possible methods include: base class, decorator, metaclass. We will use the
    metaclass because it is best suited for this purpose.
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        """
        Possible changes to the value of the `__init__` argument do not affect
        the returned instance.
        """
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class Bottom(metaclass=SingletonMeta):

    def __init__(self):
        x = sym.Symbol('x')
        y_expr = sym.sin(x) + (0.2 * sym.cos(4 * x + sym.sin(4 * x))) - 0.2 * x + 3
        #y_expr = 0.1 * x
        #y_expr = 1+sym.sin(x*np.pi/4)
        y_expr_diff = sym.diff(y_expr, x)

        self.__f = sym.lambdify(x, y_expr, "numpy")
        self.__f_diff = sym.lambdify(x, y_expr_diff, "numpy")

    def normal(self, x):
        vector_normal = np.array([-self.f_diff(x), 1, 0])
        vector_normal = vector_normal / np.sqrt(np.sum(vector_normal ** 2))
        return vector_normal

    @property
    def f(self):
        return self.__f

    @property
    def f_diff(self):
        return self.__f_diff


class Particle:
    """A class representing a two-dimensional particle."""

    def __init__(self, m, x, y, vx, vy, side_length=0.01, color='blue'):
        """Initialize the particle's position, velocity, and radius.

        Any key-value pairs passed in the styles dictionary will be passed
        as arguments to Matplotlib's Circle patch constructor.

        """
        self.center_position = np.array([x, y, 0], dtype=float)
        self.center_velocity = np.array([vx, vy, 0], dtype=float)
        self.theta = float(0)
        self.omega = np.array([0, 0, 0], dtype=float)
        self.side_length = side_length
        self.color = color
        self.elasticity = 1
        self.g = 9.81
        self.mass = m
        self.I = m * (side_length**2) / 6
        self.bottom = Bottom()
        self.energy_kin = float(0)
        self.energy_pot = float(0)
        self.collision_position = np.array([0, 0, 0], dtype=float)
        self.collision = False

        self.calculate_corners

    @property
    def calculate_corners(self):
        delta_x = np.cos(self.theta + np.pi/4)*(self.side_length/np.sqrt(2))
        delta_y = np.sin(self.theta + np.pi/4)*(self.side_length/np.sqrt(2))

        self.corner1 = np.array([self.center_position[0]-delta_x, self.center_position[1]-delta_y, 0])
        self.corner2 = np.array([self.center_position[0]+delta_y , self.center_position[1]-delta_x, 0])
        self.corner3 = np.array([self.center_position[0]+delta_x , self.center_position[1]+delta_y, 0])
        self.corner4 = np.array([self.center_position[0]-delta_y , self.center_position[1]+delta_x, 0])

    def draw(self):
        """Add this Particle's Rectangle patch to the Matplotlib Axes ax."""

        square = Rectangle((self.corner1[0], self.corner1[1]), self.side_length, self.side_length, angle=self.theta*180/np.pi, color=self.color)
        return square

    def draw_center(self):
        """Add this Particle's Rectangle patch to the Matplotlib Axes ax."""
        circle_center = Circle((self.center_position[0], self.center_position[1]), 0.1)
        return circle_center

    def draw_corners(self):
        """Add this Particle's Rectangle patch to the Matplotlib Axes ax."""
        circle_corner1 = Circle((self.corner1[0], self.corner1[1]), 0.1, color='green')
        circle_corner2 = Circle((self.corner2[0], self.corner2[1]), 0.1, color='yellow')
        circle_corner3 = Circle((self.corner3[0], self.corner3[1]), 0.1, color='red')
        circle_corner4 = Circle((self.corner4[0], self.corner4[1]), 0.1, color='black')
        return circle_corner1, circle_corner2, circle_corner3, circle_corner4

    def draw_collision(self):
        """Add this Particle's Rectangle patch to the Matplotlib Axes ax."""
        circle_collision = Circle((self.collision_position[0], self.collision_position[1]), 0.1, color='pink')
        return circle_collision

    def intersectionBottom(self):
        division = 10
        for i in range(division):
            x = self.corner1[0] + i*(self.corner2[0]-self.corner1[0])/division

            y = self.corner1[1] + i*(self.corner2[1]-self.corner1[1])/division
            if y <= self.bottom.f(x):
                return True,  np.array([x, y, 0])
            x = self.corner2[0] + i*(self.corner3[0]-self.corner2[0])/division
            y = self.corner2[1] + i*(self.corner3[1]-self.corner2[1])/division
            if y <=  self.bottom.f(x):
                return True,  np.array([x, y, 0])
            x = self.corner3[0] + i * (self.corner4[0] - self.corner3[0]) / division
            y = self.corner3[1] + i * (self.corner4[1] - self.corner3[1]) / division
            if y <=  self.bottom.f(x):
                return True,  np.array([x, y, 0])
            x = self.corner4[0] + i * (self.corner1[0] - self.corner4[0]) / division
            y = self.corner4[1] + i * (self.corner1[1] - self.corner4[1]) / division
            if y <=  self.bottom.f(x):
                return True,  np.array([x, y, 0])
        return False, np.array([0, 0, 0])

    def intersectionTop(self):
        if self.corner1[1] >= 10:
            return True, self.corner1
        if self.corner2[1] >= 10:
            return True, self.corner2
        if self.corner3[1] >= 10:
            return True, self.corner3
        if self.corner4[1] >= 10:
            return True, self.corner4
        return False, (0, 0)

    def intersectionLeft(self):
        minimum = np.min(np.array([self.corner1[0], self.corner2[0], self.corner3[0], self.corner4[0]]))
        if minimum <= 0:
            if self.corner1[0] == minimum:
                return True, self.corner1
            if self.corner2[0] == minimum:
                return True, self.corner2
            if self.corner3[0] == minimum:
                return True, self.corner3
            if self.corner4[0] == minimum:
                return True, self.corner4
        return False, (0, 0)

    def intersectionRight(self):
        maximum = np.max(np.array([self.corner1[0], self.corner2[0], self.corner3[0], self.corner4[0]]))
        if maximum >= 10:
            corners = []
            if self.corner1[0] == maximum:
                corners.append(self.corner1)
            if self.corner2[0] == maximum:
                corners.append(self.corner2)
            if self.corner3[0] == maximum:
                corners.append(self.corner3)
            if self.corner4[0] == maximum:
                corners.append(self.corner4)
            x=float(0)
            y=float(0)
            for corner in corners:
                x += corner[0]
                y += corner[1]
            return True, (x/len(corners), y/len(corners), 0)
        return False, (0, 0, 0)

    def calculate_velocity_contact(self, position_relative_contact):
        return self.center_velocity + np.cross(self.omega, position_relative_contact)

    def calculate_impulse_J(self, position_relative_contact, velocity_contact, normal_contact):
        crossproduct_position_normal = np.cross(position_relative_contact, normal_contact)
        j = -(1+self.elasticity) * np.dot(velocity_contact, normal_contact) / ((1/self.mass) + (np.dot(crossproduct_position_normal, crossproduct_position_normal)/self.I))
        return j

    def calculate_new_velocity_center(self, normal_contact, j ):
        print("vermenigvuldiging")
        print(normal_contact)
        print(j/self.mass)
        print(np.multiply(normal_contact, j / self.mass))

        return self.center_velocity + np.multiply(normal_contact, j/self.mass)

    def calculate_new_angular_velocity(self, position_relative_contact, normal_contact, j ):
        return self.omega + (np.cross(position_relative_contact, normal_contact)*j/self.I)

    def calculate_collision(self, coordinates_intersection, normal_contact):
        position_relative_contact = coordinates_intersection - self.center_position
        velocity_contact = self.calculate_velocity_contact(position_relative_contact)


        print("Massa", self.mass)
        print("Traagheidsmoment", self.I)

        print("VOOR botsing:")
        print("-------------")
        print("Normal contact: ", normal_contact)
        print("Intersection: ", coordinates_intersection)
        print("Center position: ", self.center_position)
        print("Relative position contact w.r.t. center: ", position_relative_contact)
        print("Velocity contact: ", velocity_contact)
        print("Velocity contact normal: ", np.dot(velocity_contact, normal_contact) * normal_contact)
        print("Velocity contact tangential: ", velocity_contact - np.dot(velocity_contact, normal_contact) * normal_contact)
        print("Center velocity: ", self.center_velocity)
        print("Omega: ", self.omega)

        print("Energy translation: ", (self.mass * np.dot(self.center_velocity, self.center_velocity))/2)
        print("Energy rotation: ", (self.I * (self.omega[2]**2))/2)
        print("Energy potential", self.mass * self.g * self.center_position[1])
        print("Energy total", (self.mass * np.dot(self.center_velocity, self.center_velocity))/2 + (self.I * (self.omega[2]**2))/2 + self.mass * self.g * self.center_position[1])


        j = self.calculate_impulse_J(position_relative_contact, velocity_contact, normal_contact)
        self.center_velocity = self.calculate_new_velocity_center(normal_contact, j)
        self.omega = self.calculate_new_angular_velocity(position_relative_contact, normal_contact, j)
        new_velocity_contact = self.center_velocity + np.cross(self.omega, position_relative_contact)

        print();
        print("NA botsing:")
        print("-----------")
        print("New velocity contact", new_velocity_contact)
        print("New velocity contact normal", np.dot(new_velocity_contact, normal_contact) * normal_contact)
        print("New velocity contact tangential", new_velocity_contact - np.dot(new_velocity_contact, normal_contact) * normal_contact)
        print("New velocity center", self.center_velocity)
        print("New omega", self.omega)
        print("Energy translation: ", (self.mass * np.dot(self.center_velocity, self.center_velocity))/2)
        print("Energy rotation: ", (self.I * (self.omega[2]**2))/2)
        print("Energy potential", self.mass * self.g * self.center_position[1])
        print("Energy total", (self.mass * np.dot(self.center_velocity, self.center_velocity))/2 + (self.I * (self.omega[2]**2))/2 + self.mass * self.g * self.center_position[1])
        print();

    def advance(self, dt):
        """Advance the Particle's position forward in time by dt."""
        self.center_position += self.center_velocity * dt
        self.theta += self.omega[2] * dt
        self.center_velocity[1] += -self.g * dt
        self.calculate_corners
        self.collision = False

        self.energy_kin = (self.mass * (np.dot(self.center_velocity, self.center_velocity)) + self.I * (self.omega[2]**2))/2
        self.energy_pot = self.mass * self.g * self.center_position[1]

        maximum = np.max(np.array([self.corner1[0], self.corner2[0], self.corner3[0], self.corner4[0]]))
        if (maximum > 10):
            self.center_position[0] -= maximum-10

        minimum = np.min(np.array([self.corner1[0], self.corner2[0], self.corner3[0], self.corner4[0]]))
        if (minimum < 0):
            self.center_position[0] -= minimum

        maximum = np.max(np.array([self.corner1[1], self.corner2[1], self.corner3[1], self.corner4[1]]))
        if (maximum > 10):
            self.center_position[1] -= maximum-10

        self.calculate_corners

        intersection, coordinates_intersection = self.intersectionBottom()
        if intersection :
            print("botsing bottom")
            self.collision_position = np.array([coordinates_intersection[0], coordinates_intersection[1], 0], dtype=float)
            self.collision = True

            if (self.collision_position[1] < self.bottom.f(self.collision_position[0])):
                self.center_position[1] += self.bottom.f(self.collision_position[0]) - self.collision_position[1]
                self.calculate_corners

            normal_contact = self.bottom.normal(coordinates_intersection[0])
            self.calculate_collision(coordinates_intersection, normal_contact)


        intersection, coordinates_intersection = self.intersectionTop()
        if intersection :
            print("botsing top")
            self.collision_position = np.array([coordinates_intersection[0], coordinates_intersection[1], 0], dtype=float)
            self.collision = True

            normal_contact = np.array([0, -1, 0],  dtype=float)
            self.calculate_collision(coordinates_intersection, normal_contact)

        intersection, coordinates_intersection = self.intersectionLeft()
        if intersection :
            print("botsing left")
            self.collision_position = np.array([coordinates_intersection[0], coordinates_intersection[1], 0], dtype=float)
            self.collision = True
            normal_contact = np.array([1, 0, 0],  dtype=float)
            self.calculate_collision(coordinates_intersection, normal_contact)

        intersection, coordinates_intersection = self.intersectionRight()
        if intersection :
            print("botsing right")
            self.collision_position = np.array([coordinates_intersection[0], coordinates_intersection[1], 0], dtype=float)
            self.collision = True
            normal_contact = np.array([-1, 0, 0])
            self.calculate_collision(coordinates_intersection, normal_contact)



class Simulation:
    def __init__(self, side_length=0.01):
        self.bottom = Bottom()
        self.particle = Particle(1, 2, 8, 0, -1, side_length, 'blue')

    def init(self):
        """Initialize the Matplotlib animation."""
        objects_to_draw = []
        objects_to_draw.append(self.particle.draw())
        ax.add_patch(objects_to_draw[0])

        self.velocity_x_text = ax.text(1, 9, '')
        self.velocity_y_text = ax.text(1, 8, '')
        self.energy_kin_text = ax.text(1, 7, '')
        self.energy_pot_text = ax.text(1, 6, '')
        self.theta_text = ax.text(6, 9, '')
        self.omega_text = ax.text(6, 8, '')
        self.energy_tot_text = ax.text(6, 7, '')
        return objects_to_draw

    def advance_animation(self, dt):
        self.particle.advance(dt)
        return

    def animate(self, i):
        global fig, ax
        self.advance_animation(0.005)
        objects_to_draw = []
        objects_to_draw.append(self.particle.draw())
        ax.add_patch(objects_to_draw[0])

        objects_to_draw.append(self.particle.draw_center())
        ax.add_patch(objects_to_draw[1])

        corner1, corner2, corner3, corner4 = self.particle.draw_corners()
        objects_to_draw.append(corner1)
        objects_to_draw.append(corner2)
        objects_to_draw.append(corner3)
        objects_to_draw.append(corner4)

        ax.add_patch(objects_to_draw[2])
        ax.add_patch(objects_to_draw[3])
        ax.add_patch(objects_to_draw[4])
        ax.add_patch(objects_to_draw[5])

        if True:
            collision = self.particle.draw_collision()
            objects_to_draw.append(collision)
            ax.add_patch(collision)

        self.velocity_x_text.set_text(f'Vx: {self.particle.center_velocity[0]:.1f} m')
        self.velocity_y_text.set_text(f'Vy: {self.particle.center_velocity[1]:.1f} m')

        self.energy_kin_text.set_text(f'E Kin: {self.particle.energy_kin:.1f} J')
        self.energy_pot_text.set_text(f'E Pot: {self.particle.energy_pot:.1f} J')
        self.energy_tot_text.set_text(f'E Tot: {self.particle.energy_pot + self.particle.energy_kin:.1f} J')
        theta_degrees = self.particle.theta * 180 / np.pi
        self.theta_text.set_text(f'Theta: {theta_degrees:.1f} °')
        self.omega_text.set_text(f'Omega: {self.particle.omega[2]:.1f} rad/s')

        objects_to_draw.append(self.velocity_x_text)
        objects_to_draw.append(self.velocity_y_text)
        objects_to_draw.append(self.energy_pot_text)
        objects_to_draw.append(self.energy_kin_text)
        objects_to_draw.append(self.energy_tot_text)
        objects_to_draw.append(self.theta_text)
        objects_to_draw.append(self.omega_text)

        return objects_to_draw

    def do_animation(self):
        global fig, ax
        anim = animation.FuncAnimation(fig, self.animate, init_func=self.init,
                               frames=50, interval=1, blit=True)
        plt.show()

if __name__ == '__main__':
    global fig, ax
    fig = plt.figure()
    ax = fig.add_subplot()

    x1 = np.arange(0, 10, 0.01)
    bottom = Bottom()
    y1 = bottom.f(x1)
    ax.plot(x1, y1, lw=2)

    for s in ['top', 'bottom', 'left', 'right']:
        ax.spines[s].set_linewidth(2)

    ax.set_aspect('equal', 'box')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])

    side_length = 1
    sim = Simulation(side_length)
    sim.do_animation()