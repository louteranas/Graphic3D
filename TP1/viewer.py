#!/usr/bin/env python3
"""
Python OpenGL practical application.
"""
# Python built-in modules
import os                           # os function, i.e. checking file status

# External, non built-in modules
import OpenGL.GL as GL              # standard Python OpenGL wrapper
import glfw                         # lean window system wrapper for OpenGL
import numpy as np                  # all matrix manipulations & OpenGL arg
from transform import *


# ------------ low level OpenGL object wrappers ----------------------------
class Shader:
    """ Helper class to create and automatically destroy shader program """
    @staticmethod
    def _compile_shader(src, shader_type):
        src = open(src, 'r').read() if os.path.exists(src) else src
        src = src.decode('ascii') if isinstance(src, bytes) else src
        shader = GL.glCreateShader(shader_type)
        GL.glShaderSource(shader, src)
        GL.glCompileShader(shader)
        status = GL.glGetShaderiv(shader, GL.GL_COMPILE_STATUS)
        src = ('%3d: %s' % (i+1, l) for i, l in enumerate(src.splitlines()))
        if not status:
            log = GL.glGetShaderInfoLog(shader).decode('ascii')
            GL.glDeleteShader(shader)
            src = '\n'.join(src)
            print('Compile failed for %s\n%s\n%s' % (shader_type, log, src))
            return None
        return shader

    def __init__(self, vertex_source, fragment_source):
        """ Shader can be initialized with raw strings or source file names """
        self.glid = None
        vert = self._compile_shader(vertex_source, GL.GL_VERTEX_SHADER)
        frag = self._compile_shader(fragment_source, GL.GL_FRAGMENT_SHADER)
        if vert and frag:
            self.glid = GL.glCreateProgram()  # pylint: disable=E1111
            GL.glAttachShader(self.glid, vert)
            GL.glAttachShader(self.glid, frag)
            GL.glLinkProgram(self.glid)
            GL.glDeleteShader(vert)
            GL.glDeleteShader(frag)
            status = GL.glGetProgramiv(self.glid, GL.GL_LINK_STATUS)
            if not status:
                print(GL.glGetProgramInfoLog(self.glid).decode('ascii'))
                GL.glDeleteProgram(self.glid)
                self.glid = None

    def __del__(self):
        GL.glUseProgram(0)
        if self.glid:                      # if this is a valid shader object
            GL.glDeleteProgram(self.glid)  # object dies => destroy GL object

#gl_Position = vec4(position, 1);

# ------------  Simple color shaders ------------------------------------------
COLOR_VERT = """#version 330 core
uniform mat4 viewMatrix;
uniform mat4 projectionMatrix;
layout(location = 0) in vec3 position;
out vec4 ColorPosition;
void main() {
    gl_Position = projectionMatrix * viewMatrix * vec4(position, 1);
    ColorPosition = gl_Position;
}"""

COLOR_FRAG = """#version 330 core
//uniform vec3 color;
in vec4 ColorPosition;
out vec4 outColor;
void main() {
    outColor = vec4(ColorPosition[0]+0.5, ColorPosition[1]+0.5, ColorPosition[2], 1);

}"""


# ------------  Scene object classes ------------------------------------------
class SimpleTriangle:
    """Hello triangle objectNone"""

    def __init__(self):

        # triangle position buffer
        position = np.array(((0, .5, 0), (.5, -.5, 0), (-.5, -.5, 0)), 'f')
        # color = np.array(((1, 0, 0), (0, 1, 0), (0, 0, 1)), 'f')

        self.glid = GL.glGenVertexArrays(1)  # create OpenGL vertex array id
        GL.glBindVertexArray(self.glid)      # activate to receive state below
        self.buffers = [GL.glGenBuffers(1)]  # create buffer for position attrib

        # bind the vbo, upload position data to GPU, declare its size and type
        GL.glEnableVertexAttribArray(0)      # assign to layout = 0 attribute
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.buffers[0])
        GL.glBufferData(GL.GL_ARRAY_BUFFER, position, GL.GL_STATIC_DRAW)
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, False, 0, None)
        #
        # bind the vbo, upload posviewition data to GPU, declare its size and type
        # GL.glEnableVertexAttribArray(1)      # assign to layout = 0 attribute
        # GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.buffers[0])
        # GL.glBufferData(GL.GL_ARRAY_BUFFER, color, GL.GL_STATIC_DRAW)
        # GL.glVertexAttribPointer(1, 3, GL.GL_FLOAT, False, 0, None)

        # cleanup and unbind so no accidental subsequent state update
        GL.glBindVertexArray(0)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)


    def draw(self, projection, view, model, color_shader, color_array=(0.6, 0.6, 0.9)):
        GL.glUseProgram(color_shader.glid)

        #my_color_location = GL.glGetUniformLocation(color_shader.glid, 'color')
        matrix_location = GL.glGetUniformLocation(color_shader.glid, 'viewMatrix')
        GL.glUniformMatrix4fv(matrix_location, 1, True, view)
        matrix_location = GL.glGetUniformLocation(color_shader.glid, 'projectionMatrix')
        GL.glUniformMatrix4fv(matrix_location, 1, True, projection)
        #GL.glUniform3fv(my_color_location, 1, color_array)

        # draw triangle as GL_TRIANGLE vertex array, draw array call
        GL.glBindVertexArray(self.glid)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, 3)
        GL.glBindVertexArray(0)


    def __del__(self):
        GL.glDeleteVertexArrays(1, [self.glid])
        GL.glDeleteBuffers(1, self.buffers)


class SimplePiramid:
    """Hello triangle objectNone"""

    def __init__(self):

        # one time initialization
        position = np.array(((0, 0), (0, 1), (1, 0), (1, 1), (2, 1)), np.float32)
        index = np.array((0, 2, 1, 2, 3, 1, 3, 2, 4), np.uint32)

        glid = GL.glGenVertexArrays(1)            # create a vertex array OpenGL identifier
        GL.glBindVertexArray(glid)                # make it active for receiving state below

        buffers = [GL.glGenBuffers(1)]            # create one OpenGL buffer for our position attribute
        GL.glEnableVertexAttribArray(0)           # assign state below to shader attribute layout = 0
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, buffers[0])                         # our created position buffer
        GL.glBufferData(GL.GL_ARRAY_BUFFER, position, GL.GL_STATIC_DRAW)        # upload our vertex data to it
        GL.glVertexAttribPointer(0, 2, GL.GL_FLOAT, False, 0, None)             # describe array unit as 2 floats
        ...                                       # optionally add attribute buffers here, same nb of vertices

        buffers += [GL.glGenBuffers(1)]                                           # create GPU index buffer
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, buffers[-1])                  # make it active to receive
        GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, index, GL.GL_STATIC_DRAW)     # our index array here


        # when drawing in the rendering loop: use glDrawElements for index buffer
        GL.glBindVertexArray(glid)                                                # activate our vertex array
        GL.glDrawElements(GL.GL_TRIANGLES, index.size, GL.GL_UNSIGNED_INT, None)  # 9 indexed verts = 3 triangles


    def draw(self, projection, view, model, color_shader, color_array=(0.6, 0.6, 0.9)):
        GL.glUseProgram(color_shader.glid)

        #my_color_location = GL.glGetUniformLocation(color_shader.glid, 'color')
        matrix_location = GL.glGetUniformLocation(color_shader.glid, 'viewMatrix')
        GL.glUniformMatrix4fv(matrix_location, 1, True, view)
        matrix_location = GL.glGetUniformLocation(color_shader.glid, 'projectionMatrix')
        GL.glUniformMatrix4fv(matrix_location, 1, True, projection)
        #GL.glUniform3fv(my_color_location, 1, color_array)

        # draw triangle as GL_TRIANGLE vertex array, draw array call
        GL.glBindVertexArray(self.glid)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, 3)
        GL.glBindVertexArray(0)


    def __del__(self):
        GL.glDeleteVertexArrays(1, [self.glid])
        GL.glDeleteBuffers(1, self.buffers)


class GLFWTrackball(Trackball):
    """ Use in Viewer for interactive viewpoint control """

    def __init__(self, win):
        """ Init needs a GLFW window handler 'win' to register callbacks """
        super().__init__()
        self.mouse = (0, 0)
        glfw.set_cursor_pos_callback(win, self.on_mouse_move)
        glfw.set_scroll_callback(win, self.on_scroll)

    def on_mouse_move(self, win, xpos, ypos):
        """ Rotate on left-click & drag, pan on right-click & drag """
        old = self.mouse
        self.mouse = (xpos, glfw.get_window_size(win)[1] - ypos)
        if glfw.get_mouse_button(win, glfw.MOUSE_BUTTON_LEFT):
            self.drag(old, self.mouse, glfw.get_window_size(win))
        if glfw.get_mouse_button(win, glfw.MOUSE_BUTTON_RIGHT):
            self.pan(old, self.mouse)

    def on_scroll(self, win, _deltax, deltay):
        """ Scroll controls the camera distance to trackball center """
        self.zoom(deltay, glfw.get_window_size(win)[1])


# ------------  Viewer class & window management ------------------------------
class Viewer:
    """ GLFW viewer window, with classic initialization & graphics loop """

    def __init__(self, width=640, height=480, color_array = (1, 0.6, 0.9)):

        # version hints: create GL window with >= OpenGL 3.3 and core profile
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL.GL_TRUE)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.RESIZABLE, False)
        self.win = glfw.create_window(width, height, 'Viewer', None, None)
        self.trackball = GLFWTrackball(self.win)

        # make win's OpenGL context current; no OpenGL calls can happen before
        glfw.make_context_current(self.win)

        # register event handlers
        glfw.set_key_callback(self.win, self.on_key)

        # useful message to check OpenGL renderer characteristics
        print('OpenGL', GL.glGetString(GL.GL_VERSION).decode() + ', GLSL',
              GL.glGetString(GL.GL_SHADING_LANGUAGE_VERSION).decode() +
              ', Renderer', GL.glGetString(GL.GL_RENDERER).decode())

        # initialize GL by setting viewport and default render characteristics
        GL.glClearColor(0.1, 0.1, 0.1, 0.1)

        # compile and initialize shader programs once globally
        self.color_shader = Shader(COLOR_VERT, COLOR_FRAG)

        # initially empty list of object to draw
        self.drawables = []
        self.color_array = color_array

    def run(self):
        """ Main render loop for this OpenGL window """
        while not glfw.window_should_close(self.win):
            # clear draw buffer
            GL.glClear(GL.GL_COLOR_BUFFER_BIT)

            winsize = glfw.get_window_size(self.win)
            view = self.trackball.view_matrix()
            projection = self.trackball.projection_matrix(winsize)
            # draw our scene objects
            for drawable in self.drawables:
                drawable.draw(projection, view, identity(), self.color_shader)
                #drawable.draw(None, None, None, self.color_shader, self.color_array)

            # flush render commands, and swap draw buffers
            glfw.swap_buffers(self.win)

            # Poll for and process events
            glfw.poll_events()

    def add(self, *drawables):
        """ add objects to draw in this window """
        self.drawables.extend(drawables)

    def on_key(self, _win, key, _scancode, action, _mods):
        """ 'Q' or 'Escape' quits """
        if action == glfw.PRESS or action == glfw.REPEAT:
            if key == glfw.KEY_ESCAPE or key == glfw.KEY_Q:
                glfw.set_window_should_close(self.win, True)
            if key == glfw.KEY_ENTER:
                self.color_array = ((self.color_array[0]+0.1)%1, (self.color_array[1]+0.1)%1, (self.color_array[2]+0.1)%1)


# -------------- main program and scene setup --------------------------------
def main():
    """ create a window, add scene objects, then run rendering loop """
    viewer = Viewer()

    # place instances of our basic objects
    viewer.add(SimpleTriangle())

    # start rendering loop
    viewer.run()


if __name__ == '__main__':
    glfw.init()                # initialize window system glfw
    main()                     # main function keeps variables locally scoped
    glfw.terminate()           # destroy all glfw windows and GL contexts
