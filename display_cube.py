import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import gluPerspective
import numpy as np

# Camera variables
angle_x = 0
angle_y = 0
mouse_dragging = False
last_mouse_x, last_mouse_y = 0, 0
camera_distance = 10  # Adjust this for initial zoom level
camera_translation = [0, 0]  # Translation offsets for panning

# Define the cube's vertices and edges
vertices = [
    [1, 3, -1],   # 0
    [1, 3, 1],    # 1
    [-1, 3, 1],   # 2
    [-1, 3, -1],  # 3
    [1, 1, -1],   # 4
    [1, 1, 1],    # 5
    [-1, 1, -1],  # 6
    [-1, 1, 1]    # 7
]

edges = [
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 0],
    [4, 5],
    [5, 7],
    [7, 6],
    [6, 4],
    [0, 4],
    [1, 5],
    [2, 7],
    [3, 6]
]

def rotate_around_axis(vertex, axis, theta):
    """
    Rotate a vertex around an arbitrary axis.
    """
    axis = np.asarray(axis)
    axis = axis/np.sqrt(np.dot(axis, axis))
    a = np.cos(theta/2)
    b, c, d = -axis*np.sin(theta/2)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    rotation_matrix = np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                                [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                                [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])
    return np.dot(rotation_matrix, vertex)

def get_transformed_vertices(vertices, angle):
    transformed_vertices = []
    for vertex in vertices:
        vertex = np.array(vertex) + [0, 2, 0]  # Translate to cube's center
        rotated_vertex = rotate_around_axis(vertex, [1, 1, 1], angle)
        rotated_vertex = rotated_vertex - [0, 2, 0]  # Translate back
        transformed_vertices.append(rotated_vertex)
    return transformed_vertices

def get_shadow_vertex(vertex, angle):
    # Step 1: Apply the same transformation as the cube
    vertex = rotate_around_axis(vertex, [1, 1, 1], angle)
    
    # Step 2: Flatten the transformed vertex onto the ground
    shadow_vertex = [vertex[0], 0, vertex[2]]  # Simply setting y to 0

    return shadow_vertex

def draw_shadow(vertices):
    """
    Draw the cube's shadow on the ground using the transformed vertices.
    """
    glColor3f(0.3, 0.3, 0.3)  # Shadow color
    glBegin(GL_LINES)
    for edge in edges:
        for vertex_index in edge:
            shadow_vertex = [vertices[vertex_index][0], 0, vertices[vertex_index][2]]  # Flattening y-coordinate
            glVertex3fv(shadow_vertex)
    glEnd()


def draw_checkered_ground(size, squares):
    half_size = size / 2
    square_size = size / squares

    for x in range(squares):
        for z in range(squares):
            # Determine the color
            if (x + z) % 2 == 0:
                glColor3f(0.5, 0.5, 0.5)  # Light gray
            else:
                glColor3f(0.9, 0.9, 0.9)  # Dark gray

            # Draw the square
            glBegin(GL_QUADS)
            glVertex3f(-half_size + x * square_size, 0, -half_size + z * square_size)
            glVertex3f(-half_size + x * square_size, 0, -half_size + (z+1) * square_size)
            glVertex3f(-half_size + (x+1) * square_size, 0, -half_size + (z+1) * square_size)
            glVertex3f(-half_size + (x+1) * square_size, 0, -half_size + z * square_size)
            glEnd()

def draw_cube():
    glColor3f(0, 0, 1)  # Set color to blue
    glBegin(GL_LINES)
    for edge in edges:
        for vertex in edge:
            glVertex3fv(vertices[vertex])
    glEnd()

def draw_spheres_at_vertices():
    glColor3f(1, 0, 0)  # Color of the spheres
    for vertex in vertices:
        glPushMatrix()
        glTranslatef(*vertex)
        glutSolidSphere(0.1, 20, 20)  # Draw a sphere of radius 0.1 with 20 slices and 20 stacks
        glPopMatrix()

def mouse_button_callback(event):
    global mouse_dragging, last_mouse_x, last_mouse_y, camera_distance

    if event.type == pygame.MOUSEBUTTONDOWN:
        if event.button == 1:  # Left button press for rotation
            mouse_dragging = "DRAG"
            last_mouse_x, last_mouse_y = event.pos
        elif event.button == 3:  # Right button press for panning
            mouse_dragging = "PAN"
            last_mouse_x, last_mouse_y = event.pos
        elif event.button == 4:  # Mouse wheel up
            camera_distance -= 1.0
        elif event.button == 5:  # Mouse wheel down
            camera_distance += 1.0

    elif event.type == pygame.MOUSEBUTTONUP:
        if event.button in [1, 3]:  # Left or right button release
            mouse_dragging = False

def mouse_motion_callback(event):
    global angle_x, angle_y, last_mouse_x, last_mouse_y, camera_translation

    dx = event.pos[0] - last_mouse_x
    dy = event.pos[1] - last_mouse_y

    if mouse_dragging == "DRAG":  # Rotation
        angle_x += dy * 0.5
        angle_y += dx * 0.5
    elif mouse_dragging == "PAN":  # Panning
        camera_translation[0] += dx * 0.05
        camera_translation[1] -= dy * 0.05

    last_mouse_x, last_mouse_y = event.pos
    # ... [rest of the code remains unchanged]

def main():
    pygame.init()
    display = (800, 600)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)
    glTranslatef(0.0, -2, -10)  # Adjusted to have a top-down view

    angle = 0
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            mouse_button_callback(event)
            if mouse_dragging:
                mouse_motion_callback(event)

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)
        glTranslatef(camera_translation[0], camera_translation[1], -camera_distance)
        glRotatef(angle_x, 1, 0, 0)
        glRotatef(angle_y, 0, 1, 0)
        
        
        draw_checkered_ground(20, 10)

        # Lift and rotate the cube
        glPushMatrix()
        glTranslatef(0, 2, 0)  # Translate to cube's center
        glRotatef(angle, 1, 1, 1)  # Rotating the cube
        glTranslatef(0, -2, 0)  # Translate back
        glLineWidth(5)  # Make lines thicker
        
        glColor3f(0, 0, 1)  # Restore cube color
        draw_cube()
        draw_spheres_at_vertices()
        glPopMatrix()

        angle += 0.5  # Increment the cube's rotation angle

        pygame.display.flip()
        pygame.time.wait(10)

if __name__ == "__main__":
    main()
