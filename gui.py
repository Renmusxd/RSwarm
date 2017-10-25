from world import *
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

    def __init__(self, world, windowwidth, windowheight, x=0, y=0, z=None):
        self.ww, self.wh = windowwidth, windowheight
        self.x, self.y = x, y
        self.dx, self.dy, self.dz = 0, 0, 1
        self.world = world
        self.botvalues = None

        self.focus = None
        self.focusclass = None
        self.focussenses = None
        self.focusactions = None

        self.sense_label_cache = None
        self.action_label_cache = None
        if z is None:
            worldxsize = World.TILE_SIZE * world.tileshape[0]
            worldysize = World.TILE_SIZE * world.tileshape[1]
            self.z = min(self.ww/worldxsize,self.wh/worldysize)

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
        self.botvalues, fsense, fact = world.get_bot_values()

        if fsense is not None:
            ins, vis, dist = Bot.split_senses(fsense)
            self.focussenses = (Bot.label_inputs(ins), vis, dist)
        else:
            self.focussenses = None
        if fact is not None:
            self.focusactions = Bot.label_actions(fact)
        else:
            self.focusactions = None

        for i in range(self.world.tileshape[0]):
            for j in range(self.world.tileshape[1]):
                x = i * World.TILE_SIZE
                y = j * World.TILE_SIZE
                self._draw_tile(x, y, tilepercs[i,j])
        for i in range(self.botvalues.shape[0]):
            eid = self.botvalues[i,0]
            if eid != self.focus:
                self._draw_bot(self.botvalues[i,:])
            else:
                self._draw_bot(self.botvalues[i,:], focus=True,
                               focussenses=self.focussenses, focusactions=self.focusactions)

        self._draw_debug(self.focussenses, self.focusactions)

        if self.focussenses is None and self.focusactions is None:
            for i in range(self.botvalues.shape[0]):
                eid, ecls, ex, ey, ed, er, eg, eb = self.botvalues[i,:]
                if ecls == self.focusclass:
                    self.focus = self.botvalues[i,0]

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

    def _draw_bot(self, botinfo, size=World.ENTITY_SIZE, focus=False, focussenses=None, focusactions=None, drawvis=False):
        eid, ecls, x, y, d, r, g, b = botinfo

        glLoadIdentity()
        glTranslatef(self.z * (x - self.x), self.z * (y - self.y), 0.0)
        glScalef(self.z * size, self.z * size, 1.0)
        glRotatef(d - 90.0, 0, 0, 1)
        glColor4f(r, g, b, 1.0)
        # Bot body
        glBegin(GL_TRIANGLES)
        glVertex2f(-0.3, -0.5, 0)
        glVertex2f(0.3, -0.5, 0)
        glVertex2f(0.0, 0.5, 0)
        glEnd()

        if focus or drawvis:
            if focussenses is not None:
                senses, vis, dist = focussenses
            else:
                senses, vis, dist = None, None, None
            hasdist = dist is not None
            hasvis = vis is not None

            # Vision cones
            vbins = Bot.VISION_BINS
            vlow = -Bot.FOV
            vhigh = Bot.FOV
            binangle = (vhigh - vlow) / vbins
            vdist = Bot.VIEW_DIST / size

            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glColor4f(r, g, b, 0.25)
            glBegin(GL_TRIANGLES)
            for i in range(vbins):
                angleindx = i
                lowangle = binangle*angleindx + vlow
                highangle = binangle*(angleindx+1) + vlow

                # Add 90.0 since bot-drawings are vertical, not rightwards

                lowx = vdist * numpy.cos(numpy.deg2rad(lowangle + 90.0)) * (dist[i] if hasdist else 1.0)
                lowy = vdist * numpy.sin(numpy.deg2rad(lowangle + 90.0)) * (dist[i] if hasdist else 1.0)
                highx = vdist * numpy.cos(numpy.deg2rad(highangle + 90.0)) * (dist[i] if hasdist else 1.0)
                highy = vdist * numpy.sin(numpy.deg2rad(highangle + 90.0)) * (dist[i] if hasdist else 1.0)

                if hasvis:
                    if vis[i] == 0:
                        glColor4f(0., 0., 0., 0.25)
                    elif vis[i] == -1:
                        glColor4f(1., 0., 0., 0.25)
                    elif vis[i] == 1:
                        glColor4f(0., 0., 1., 0.25)

                glVertex2f(0, 0, 0)
                glVertex2f(lowx, lowy, 0)
                glVertex2f(highx, highy, 0)
            glEnd()
            glDisable(GL_BLEND)

        if focus:
            self._draw_circle(100)

    def _draw_circle(self, npoints):
        angl = (2.0*numpy.pi)/npoints
        glColor4f(1.0, 0.0, 0.0, 1.0)
        glBegin(GL_LINE_LOOP)
        for i in range(npoints):
            glVertex2f(numpy.cos(i*angl), numpy.sin(i*angl), 0)
        glEnd()

    def _draw_debug(self, senses, actions, font_size=12):
        glLoadIdentity()
        if senses is not None:
            sense, vis, dist = senses
        else:
            sense, vis, dist = None, None, None

        if sense is not None:
            if self.sense_label_cache is None:
                self.sense_label_cache = []
                for i, k in enumerate(sorted(sense)):
                    label = pyglet.text.Label("",
                                              font_name='Times New Roman', font_size=font_size,
                                              x=10, y=self.wh - ((i+1) * (font_size + 5)),
                                              color=[255, 255, 255, 255])
                    self.sense_label_cache.append(label)
            for label, k in zip(self.sense_label_cache, sorted(sense)):
                label.text = "{}:\t{:5.5f}".format(k, sense[k])
                label.draw()

        if actions is not None:
            if self.action_label_cache is None:
                self.action_label_cache = []
                for i, k in enumerate(reversed(sorted(actions))):
                    label = pyglet.text.Label("",
                                              font_name='Times New Roman', font_size=font_size,
                                              x=10, y=(i+0.1)*(font_size + 5),
                                              color=[255,255,255,255])
                    self.action_label_cache.append(label)
            for label, k in zip(self.action_label_cache, reversed(sorted(actions))):
                label.text = "{}:\t{:5.5f}".format(k, actions[k])
                label.draw()

    def add_translate(self,dx,dy):
        self.dx += dx
        self.dy += dy

    def set_zoom(self, dz):
        self.dz = dz

    def selectnear(self, x, y):
        mapx = (x/self.z) + self.x
        mapy = (y/self.z) + self.y

        closestd2 = sys.maxsize
        for i in range(self.botvalues.shape[0]):
            eid, ecls, ex, ey, ed, er, eg, eb = self.botvalues[i,:]
            dist2 = (mapx - ex)**2 + (mapy - ey)**2
            if dist2 < closestd2:
                self.focus = int(eid)
                closestd2 = dist2
                self.focusclass = ecls

    def get_focus(self):
        return self.focus


# Python multithreading slows stuff down, probably mutex starving. Will swap to RLmutex
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
            updatewithfocus = lambda world: update(world, getfocus=gui.get_focus)

            t = Thread(target=update, args=(world,))
            t.start()
        else:
            updatewithfocus = lambda dt: world.update(dt, gui.get_focus())
            pyglet.clock.schedule(updatewithfocus)

        @win.event
        def on_draw():
            pyglet.clock.tick()
            gui.update()
            win.clear()
            gui.draw_world()

        @win.event
        def on_key_press(symbol, modifiers):
            if symbol == pyglet.window.key.LEFT:
                gui.add_translate(-GUI.SCROLL_SPEED, 0)
            elif symbol == pyglet.window.key.RIGHT:
                gui.add_translate(GUI.SCROLL_SPEED, 0)
            elif symbol == pyglet.window.key.UP:
                gui.add_translate(0, GUI.SCROLL_SPEED)
            elif symbol == pyglet.window.key.DOWN:
                gui.add_translate(0, -GUI.SCROLL_SPEED)
            elif symbol == pyglet.window.key.Q:
                gui.set_zoom(1. / GUI.ZOOM_SPEED)
            elif symbol == pyglet.window.key.E:
                gui.set_zoom(GUI.ZOOM_SPEED)
            elif symbol == pyglet.window.key.D:
                gui.set_debug(not gui.debug)

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
                gui.selectnear(x,y)

        pyglet.app.run()
    finally:
        print("Cleaning up...")
        running = False
        if not SINGLE_THREAD:
            t.join()
        world.cleanup()

