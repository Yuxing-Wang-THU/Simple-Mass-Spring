import taichi as ti
import numpy as np

ti.init(arch=ti.cpu)

resolution = 512
gui = ti.GUI('Mass-Spring System', res=(resolution, resolution), background_color=0xdddddd)

total_m = 9

max_num_particles = 9  

dt = 1e-3  

num_particles = ti.var(ti.i32, shape=())  
spring_stiffness = ti.var(ti.f32, shape=())  
paused = ti.var(ti.i32, shape=()) 
damping = ti.var(ti.f32, shape=())  

particle_mass = total_m / max_num_particles
bottom_y = 0.05  

x = ti.Vector(2, dt=ti.f32, shape=max_num_particles)  
v = ti.Vector(2, dt=ti.f32, shape=max_num_particles)  

rest_length = ti.var(ti.f32, shape=(max_num_particles, max_num_particles))

gravity = [0, -9.8] 

origin_energy = ti.var(ti.f32, shape=())  
current_energy = ti.var(ti.f32, shape=())  
lost_energy = ti.var(ti.f32, shape=()) 
damp_energy = ti.var(ti.f32, shape=())

# For the first scene
connection_radius=0.142


@ti.kernel
def new_particle(pos_x: ti.f32, pos_y: ti.f32):
    new_particle_id = num_particles[None]  
    x[new_particle_id] = [pos_x, pos_y]
    v[new_particle_id] = [0, 0]
    num_particles[None] += 1
    origin_energy[None] += -particle_mass * gravity[1] * pos_y 
    # Connect with existing particles
    for i in range(new_particle_id):
        dist = (x[new_particle_id] - x[i]).norm()
        if dist < connection_radius:
            rest_length[i, new_particle_id] = dist
            rest_length[new_particle_id, i] = dist


# collide with ground
@ti.kernel
def collide_with_ground():
    for i in range(num_particles[None]):
        if x[i].y < bottom_y:
            x[i].y = bottom_y  
            if v[i].y < 0:  
                lost_energy[None] += 0.5 * particle_mass * v[i].y * v[i].y
                v[i].y = 0.0


# compute new position
@ti.kernel
def update_position():
    for i in range(num_particles[None]):
        if x[i].y <= bottom_y:
            v[i].y = 0
        x[i] += v[i] * dt


# (green <-- black --> red)
# red means the spring is elongating
# green means the spring is compressing
@ti.kernel
def calculate_color(delta: ti.f32) -> ti.i32:
    eps = 0.00001
    color = 0x445566
    if delta > eps:
        color = 0xFF0000
    elif delta < -eps:
        color = 0x00FF00
    return color


@ti.kernel
def compute_current_energy():  # Compute current energy
    current_energy[None] = 0
    n = num_particles[None]
    for i in range(n):
        current_energy[None] += -particle_mass * gravity[1] * x[i][1]
        current_energy[None] += 0.5 * particle_mass * v[i].norm() * v[i].norm()
        for j in range(i, n): 
            if rest_length[i, j] != 0:
                x_ij = x[i] - x[j]
                current_energy[None] += 0.5 * spring_stiffness[None] * (x_ij.norm() - rest_length[i, j]) ** 2


@ti.kernel
def compute_damp_energy():
    n = num_particles[None]
    for i in range(n):
        total_force = ti.Vector([0, 0])
        for j in range(n):
            if rest_length[i, j] != 0:
                x_ij = x[i] - x[j]
                v_ij = v[i] - v[j]
                total_force += -damping[None] * x_ij.normalized() * v_ij * x_ij.normalized()  # damping
        damp_energy[None] += abs(total_force.dot(v[i]) * dt)


def init_mass_spring_system():
    # parameters
    spring_stiffness[None] = 1000
    damping[None] = 20

    # First scene
    new_particle(0.3, 0.3)
    new_particle(0.3, 0.4)
    new_particle(0.4, 0.4)
    new_particle(0.4, 0.3)
    new_particle(0.5, 0.3)
    new_particle(0.3, 0.2)
    new_particle(0.4, 0.2)
    new_particle(0.5, 0.2)
    new_particle(0.5, 0.4)

def process_input():
    for e in gui.get_events(ti.GUI.PRESS):
        if e.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
            exit()
        elif e.key == gui.SPACE:
            paused[None] = not paused[None]
        elif e.key == ti.GUI.LMB:
            new_particle(e.pos[0], e.pos[1])
        elif e.key == 'c':
            num_particles[None] = 0
            rest_length.fill(0)
        elif e.key == 's':
            if gui.is_pressed('Shift'):
                spring_stiffness[None] /= 1.1
            else:
                spring_stiffness[None] *= 1.1
        elif e.key == 'd':
            if gui.is_pressed('Shift'):
                damping[None] /= 1.1
            else:
                damping[None] *= 1.1


def process_output():
    X = x.to_numpy()
    gui.circles(X[:num_particles[None]], color=0xffaa77, radius=5)

    gui.line(begin=(0.0, bottom_y), end=(1.0, bottom_y), color=0x0, radius=1)
    
    for i in range(num_particles[None]):
        for j in range(i + 1, num_particles[None]):
            if rest_length[i, j] != 0:
                norm = np.linalg.norm(X[i] - X[j])
                gui.line(begin=X[i], end=X[j], radius=2, color=calculate_color(norm - rest_length[i, j]))
    gui.text(content=f'C: clear all; Space: pause', pos=(0, 0.95), color=0x0)
    gui.text(content=f'S: Spring stiffness {spring_stiffness[None]:.1f}', pos=(0, 0.9), color=0x0)
    gui.text(content=f'D: damping {damping[None]:.2f}', pos=(0, 0.85), color=0x0)
    gui.text(content=f'Number of particles {num_particles[None]:.0f}', pos=(0, 0.80), color=0x0)
    gui.text(content=f'Origin energy {origin_energy[None]:.0f}', pos=(0, 0.75), color=0x0)
    gui.text(content=f'Current energy {current_energy[None]:.0f}', pos=(0, 0.70), color=0x0)
    gui.text(content=f'Lost energy {lost_energy[None]:.0f}', pos=(0, 0.65), color=0x0)
    gui.text(content=f'Damp energy {damp_energy[None]:.0f}', pos=(0, 0.60), color=0x0)
    gui.text(content=f'Total energy {current_energy[None] + lost_energy[None] + damp_energy[None]:.0f}',
             pos=(0, 0.55), color=0x0)
    gui.text(
        content=f'Error energy {origin_energy[None] - current_energy[None] - lost_energy[None] - damp_energy[None]:.0f}',
        pos=(0, 0.50), color=0x0)
    gui.show()
    