from world import World
from tfbrain import TFBrain
from bot import Bot

import pyglet
from pyglet.gl import *
import numpy
from threading import Thread


class GUI:
    GREEN_GRASS = numpy.array((0.20, 0.67, 0.13))
    BROWN_GRASS = numpy.array((0.47, 0.35, 0.10))

    SCROLL_SPEED = 5
    ZOOM_SPEED = 1.05

    def __init__(self, world, windowwidth, windowheight, x=0, y=0, z=1):
        self.ww, self.wh = windowwidth, windowheight
        self.x, self.y, self.z = x, y, z
        self.dx, self.dy, self.dz = 0, 0, 1
        self.world = world

    def update(self):
        # Change dx, dy
        self.x += self.dx
        self.y += self.dy

        # Change zoom
        self.z *= self.dz

        # Adjust center position: cx = x + ww/2*z ...
        self.x += self.ww * (self.dz - 1) / (2 * self.z)
        self.y += self.wh * (self.dz - 1) / (2 * self.z)

    def draw_world(self):
        tilepercs = world.get_tile_percs()
        botvalues = world.get_bot_values()
        # print(botvalues.shape, botvalues)
        for i in range(self.world.tileshape[0]):
            for j in range(self.world.tileshape[1]):
                x = i * World.TILE_SIZE
                y = j * World.TILE_SIZE
                self._draw_tile(x, y, tilepercs[i,j])
        for i in range(botvalues.shape[0]):
            self._draw_bot(botvalues[i,:])

    def _draw_tile(self, x, y, energyperc, size=World.TILE_SIZE):

        r, g, b = (GUI.GREEN_GRASS - GUI.BROWN_GRASS) * energyperc + GUI.BROWN_GRASS

        glLoadIdentity()
        glTranslatef(self.z * (x - self.x), self.z * (y - self.y), 0.0)
        glScalef(size * self.z, size * self.z, 1.0)

        glColor4f(r, g, b, 1.0)
        glBegin(GL_QUADS)
        glVertex2f(0, 0)
        glVertex2f(1, 0)
        glVertex2f(1, 1)
        glVertex2f(0, 1)
        glEnd()

        glColor4f(0, 0, 0, 1)
        glBegin(GL_LINE_LOOP)
        glVertex2f(0, 0)
        glVertex2f(1, 0)
        glVertex2f(1, 1)
        glVertex2f(0, 1)
        glEnd()

    def _draw_bot(self, botinfo, size=World.ENTITY_SIZE):
        x, y, d, r, g, b = botinfo
        glLoadIdentity()
        glTranslatef(self.z * (x - self.x), self.z * (y - self.y), 0.0)
        glScalef(self.z * size, self.z * size, 1.0)
        glRotatef(d - 90.0, 0, 0, 1)
        glColor4f(r, g, b, 1.0)
        glBegin(GL_TRIANGLES)
        glVertex2f(-0.3, -0.5, 0)
        glVertex2f(0.3, -0.5, 0)
        glVertex2f(0.0, 0.5, 0)
        glEnd()

    def add_translate(self,dx,dy):
        self.dx += dx
        self.dy += dy

    def set_zoom(self, dz):
        self.dz = dz

running = True


def update(world, iters=0):
    iternum = 1
    while iternum != iters and running:
        world.update(1)
        iternum += 1


def make_model():
    world = World(TFBrain,TFBrain)
    return world

# Python multithreading slows stuff down
SINGLE_THREAD = True

if __name__ == "__main__":

    world = make_model()
    try:
        world.startup()
        world.update(1)
        win = pyglet.window.Window(750, 750)
        fps_display = pyglet.clock.ClockDisplay(format='%(fps).2f fps')
        gui = GUI(world, win.width, win.height)

        if not SINGLE_THREAD:
            t = Thread(target=update, args=(world,))
            t.start()
        else:
            pyglet.clock.schedule(world.update)

        @win.event
        def on_draw():
            pyglet.clock.tick()
            gui.update()
            win.clear()
            gui.draw_world()

        @win.event
        def on_key_press(symbol, modifiers):
            if symbol == pyglet.window.key.LEFT:
                gui.add_translate(-GUI.SCROLL_SPEED,0)
            elif symbol == pyglet.window.key.RIGHT:
                gui.add_translate(GUI.SCROLL_SPEED, 0)
            elif symbol == pyglet.window.key.UP:
                gui.add_translate(0,GUI.SCROLL_SPEED)
            elif symbol == pyglet.window.key.DOWN:
                gui.add_translate(0,-GUI.SCROLL_SPEED)
            elif symbol == pyglet.window.key.Q:
                gui.set_zoom(1. / GUI.ZOOM_SPEED)
            elif symbol == pyglet.window.key.E:
                gui.set_zoom(GUI.ZOOM_SPEED)

        @win.event
        def on_key_release(symbol, modifiers):
            if symbol == pyglet.window.key.LEFT:
                gui.add_translate(GUI.SCROLL_SPEED, 0)
            elif symbol == pyglet.window.key.RIGHT:
                gui.add_translate(-GUI.SCROLL_SPEED, 0)
            elif symbol == pyglet.window.key.UP:
                gui.add_translate(0, -GUI.SCROLL_SPEED)
            elif symbol == pyglet.window.key.DOWN:
                gui.add_translate(0, GUI.SCROLL_SPEED)
            elif symbol == pyglet.window.key.Q:
                gui.set_zoom(1)
            elif symbol == pyglet.window.key.E:
                gui.set_zoom(1)

        @win.event
        def on_mouse_press(x, y, button, modifiers):
            if button == pyglet.window.mouse.LEFT:
                print('The left mouse button was pressed: {},{}'.format(x, y))

        pyglet.app.run()
    finally:
        print("Cleaning up...")
        running = False
        if not SINGLE_THREAD:
            t.join()
        world.cleanup()

