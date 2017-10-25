from bot import Bot
import random
from brain import *
from tfbrain import *
from heuristics import *
from threading import Lock
import numpy
import itertools
import sys


class World:
    MAX_TILE_ENERGY = 10000
    TILE_ENERGY_RECHARGE = 1
    TILE_SIZE = 100  # 100 times entity size
    ENTITY_SIZE = 5

    MIN_BOTS = 10
    MIN_BOT_REGEN = 2
    MAX_BOT_REGEN = 5

    TRAIN_FREQ = 500
    WORLD_RESET_FREQ = 5000

    PRED_COLOR = (1., 0., 0.)
    PREY_COLOR = (0., 0., 1.)

    def __init__(self, predbraincls, preybraincls, tileshape=(5,5), restockbots=True):
        self.time = 0
        self.shouldrestock = restockbots
        self.tileshape = tileshape
        self.tiles = [[(World.MAX_TILE_ENERGY, self.time)
                       for y in range(tileshape[1])]
                      for x in range(tileshape[0])]
        self.predentities = {}
        self.preyentities = {}
        self.predbrain = Bot.make_brain(predbraincls, 'pred')
        self.preybrain = Bot.make_brain(preybraincls, 'prey')

        self.lock = Lock()
        self.tile_buffer = numpy.ones(tileshape)
        self.entity_buffer = numpy.array([])
        self.focus_senses = None
        self.focus_actvals = None
        self.stats = {}

    def update(self, dt, focusid=None):
        """
        A giant ugly update function for game logic. Will clean up later
        :param dt: dt passed by opengl if desired
        """
        # Kill
        self.cleandead()

        entitylists = [self.predentities, self.preyentities]
        entitycolors = [World.PRED_COLOR, World.PREY_COLOR]
        entitygrazes = [False, True]

        # Regen population for each entity list if below MIN_BOTS
        if self.shouldrestock:
            self.restock(entitylists, entitycolors, entitygrazes)

        # Get sensory data from all entities
        allpredsenses = {entityid: self.predentities[entityid].senses()
                         for entityid in self.predentities.keys()}
        allpreysenses = {entityid: self.preyentities[entityid].senses()
                         for entityid in self.preyentities.keys()}
        entitysenses = [allpredsenses, allpreysenses]

        # Feed into brain and get actions
        allpredactions = self.predbrain.think(allpredsenses)
        allpreyactions = self.preybrain.think(allpreysenses)
        entityactions = [allpredactions, allpreyactions]

        if focusid in self.predentities:
            focussenses = allpredsenses[focusid]
            focusactions = self.predbrain.debug(focussenses)
        elif focusid in self.preyentities:
            focussenses = allpreysenses[focusid]
            focusactions = self.preybrain.debug(focussenses)
        else:
            focussenses = None
            focusactions = None

        # Apply actions
        predrewards = {entityid: self.predentities[entityid].act(allpredactions[entityid])
                       for entityid in allpredactions.keys()}
        preyrewards = {entityid: self.preyentities[entityid].act(allpreyactions[entityid])
                       for entityid in allpreyactions.keys()}
        entityrewards = [predrewards, preyrewards]

        # Check for attacks in each list, attack closest within distance+fov
        for entitylist, rewards in zip(entitylists, entityrewards):
            attackers = list(filter(lambda b: b.attacking, entitylist.values()))
            max_d2 = Bot.ACTION_RADIUS ** 2

            # TODO review attack code to make sure none of those annoying vision bugs are here
            # I'm pretty sure I've seen some BS
            for attacker in attackers:
                closest_defender = None
                closest_distance2 = max_d2
                # Vector in direction of sight
                dirvec = numpy.array([numpy.cos(numpy.deg2rad(attacker.d)),
                                      numpy.sin(numpy.deg2rad(attacker.d))])
                # Vector perpendicular (to the left) of the direction of sight
                perpdirvec = numpy.array([numpy.cos(numpy.deg2rad(attacker.d + 90)),
                                          numpy.sin(numpy.deg2rad(attacker.d + 90))])
                # Check in both lists, add if closest and within sight range
                for defender in itertools.chain(self.predentities.values(), self.preyentities.values()):
                    dist2 = (attacker.x - defender.x) ** 2 + (attacker.y - defender.y) ** 2
                    if 0 < dist2 <= closest_distance2:
                        deltvec = numpy.array([defender.x - attacker.x, defender.y - attacker.y])
                        dist = numpy.linalg.norm(deltvec)
                        normdelt = deltvec / dist

                        # First get angle from d = 0, then subtract d
                        z_angle = numpy.rad2deg(numpy.arccos(normdelt[0]))
                        if normdelt[1] < 0:
                            z_angle = -z_angle
                        angle = modrange(z_angle - attacker.d, -180, 180)

                        if -Bot.FOV < angle < Bot.FOV:
                            closest_defender = defender
                            closest_distance2 = dist2
                if closest_defender is not None:
                    self.log("{} attacking {} | {} attacking {}"
                          .format(attacker.id,
                                  closest_defender.id,
                                  "prey" if attacker.can_graze else "pred",
                                  "prey" if closest_defender.can_graze else "pred"))
                    rewards[attacker.id] += attacker.attack_succeed(closest_defender)
                    # Not pretty, is there a "join dict" structure I can use?
                    if closest_defender.id in predrewards:
                        predrewards[closest_defender.id] += closest_defender.was_attacked(attacker)
                    elif closest_defender.id in preyrewards:
                        preyrewards[closest_defender.id] += closest_defender.was_attacked(attacker)

                    self.add_to_stat('Attacks', 1)
                else:
                    rewards[attacker.id] += attacker.attack_failed()

        # MATING MUST BE LAST -> Do not otherwise interact with new children
        # Check for mating (mate with closest)
        for entitylist, rewards, color, can_graze in zip(entitylists, entityrewards,entitycolors, entitygrazes):
            # Only check in same entitylist (pred/prey)
            maters = list(filter(lambda b: b.mating, entitylist.values()))
            max_d2 = Bot.ACTION_RADIUS ** 2
            for i in range(len(maters)):
                mater_a = maters[i]
                # Check if already mated
                if not mater_a.mating:
                    continue
                closest_mater = None
                closest_distance2 = max_d2
                for j in range(i + 1, len(maters)):
                    mater_b = maters[j]
                    # Check if already mated
                    if not mater_b.mating:
                        continue
                    dist2 = (mater_a.x - mater_b.x) ** 2 + (mater_a.y - mater_b.y) ** 2
                    if 0 < dist2 <= closest_distance2:
                        closest_mater = mater_b
                        closest_distance2 = dist2
                if closest_mater is not None:
                    self.log("Successful mating | {}".format("prey" if mater_a.can_graze else "pred"))
                    # Mate mater_a and mater_b
                    avg_x, avg_y = (mater_a.x + closest_mater.x) / 2., (mater_a.y + closest_mater.y) / 2.
                    child_x = avg_x + random.randint(-10, 10)
                    child_y = avg_y + random.randint(-10, 10)
                    # Make child
                    self.make_bot(entitylist, child_x, child_y, random.randint(0, 360),color, can_graze)

                    # Add rewards and notify
                    rewards[mater_a.id] += mater_a.mate_succeed(closest_mater)
                    rewards[closest_mater.id] += closest_mater.mate_succeed(mater_a)

                    self.add_to_stat("Mates", 1)
                else:
                    rewards[mater_a.id] += mater_a.mate_failed()

        # Get sensory data from all entities
        newpredsenses = {entityid: self.predentities[entityid].senses()
                         for entityid in self.predentities.keys()}
        newpreysenses = {entityid: self.preyentities[entityid].senses()
                         for entityid in self.preyentities.keys()}
        newentitysenses = [newpredsenses, newpreysenses]

        # Store state, action, reward
        self.predbrain.reward(allpredsenses, allpredactions, predrewards, newpredsenses)
        self.preybrain.reward(allpreysenses, allpreyactions, preyrewards, newpreysenses)

        self.add_to_stat('Predreward',sum(predrewards.values()))
        self.add_to_stat('Preyreward',sum(preyrewards.values()))

        self.time += 1

        willtrain = (self.time % World.TRAIN_FREQ) == 0
        willreset = (self.time % World.WORLD_RESET_FREQ) == 0

        # Push all to buffer, push focus if available
        self.pushtobuffer(focussenses, focusactions)

        if willtrain:
            self.printdebug()

            self.predbrain.train(Brain.DEFAULTITERS, Brain.DEFAULTBATCH, totreward=self.stats['Predreward'])
            self.preybrain.train(Brain.DEFAULTITERS, Brain.DEFAULTBATCH, totreward=self.stats['Preyreward'])
            # self.predbrain.save()
            # self.preybrain.save()

            if len(self.predentities) > 0:
                randomentity = random.choice(list(self.predentities.values()))
                self.predbrain.print_diag(randomentity.senses())
            if len(self.predentities) > 0:
                randomentity = random.choice(list(self.preyentities.values()))
                self.preybrain.print_diag(randomentity.senses())

            self.clear_stats()

            self.log("Done Training")

        if willreset:
            self.log("Resetting world")
            self.reset()

        self._clear_cache()

    def pushtobuffer(self, focussenses, focusactions):
        with self.lock:
            # Check if need to change buffer
            nentities = len(self.predentities) + len(self.preyentities)
            if nentities != len(self.entity_buffer):
                self.entity_buffer = numpy.zeros((nentities, 2 + 3 + 3))

            for i, entity in enumerate(itertools.chain(self.predentities.values(), self.preyentities.values())):
                self.entity_buffer[i, :] = [entity.id, int(entity.can_graze),
                                            entity.x, entity.y, entity.d,
                                            entity.r, entity.g, entity.b]

            for x in range(self.tileshape[0]):
                for y in range(self.tileshape[1]):
                    self.tile_buffer[x, y] = self.get_tile_perc(x * World.TILE_SIZE, y * World.TILE_SIZE)

            if focussenses is not None and focusactions is not None:
                self.focus_senses = focussenses
                self.focus_actvals = focusactions
            else:
                self.focus_senses = None
                self.focus_actvals = None

    def printdebug(self):
        self.log("Entering training cycle:")
        self.log("Population:")
        self.log("\tPred: {}".format(len(self.predentities)))
        self.log("\tPrey: {}".format(len(self.preyentities)))
        self.log("Stats:")
        for k in sorted(self.stats.keys()):
            self.log("\t{}:\t{}".format(k, self.stats[k]))

    def log(self, txt, end='\n'):
        print("[{}] {}".format(self.time, txt), end=end)

    def cleandead(self):
        predkilllist = []
        preykilllist = []
        for entity in self.predentities.values():
            if entity.dead:
                predkilllist.append(entity)
        for entity in self.preyentities.values():
            if entity.dead:
                preykilllist.append(entity)
        self._kill(predkilllist, preykilllist)  # Do any cleanup needed

    def restock(self,entitylists, entitycolors, entitygrazes):
        for entitylist, color, can_graze in zip(entitylists, entitycolors, entitygrazes):
            while len(entitylist) < World.MIN_BOTS:
                toadd = random.randint(World.MIN_BOT_REGEN, World.MAX_BOT_REGEN)
                x_center = random.randint(50, self.width() - 50)
                y_center = random.randint(50, self.height() - 50)
                for i in range(toadd):
                    xpos = x_center + random.randint(-50, 50)
                    ypos = y_center + random.randint(-50, 50)
                    self.make_bot(entitylist, xpos, ypos, random.randint(0, 360), color, can_graze)

    def reset(self):
        self.time = 0
        self.stats.clear()
        self.predentities.clear()
        self.preyentities.clear()
        self._clear_cache()

    def _kill(self, preds, preys):
        self.add_to_stat("Deaths", len(preds)+len(preys))
        for entity in preds:
            self.predentities.pop(entity.id)
        for entity in preys:
            self.preyentities.pop(entity.id)

    def make_pred(self, x, y, d=0.0):
        return self.make_bot(self.predentities, x, y, d, World.PRED_COLOR, False)

    def make_prey(self, x, y, d=0.0):
        return self.make_bot(self.preyentities, x, y, d, World.PREY_COLOR, True)

    def make_bot(self, entitylist, x, y, d, color, can_graze):
        """
        Make a new bot at a given position
        :param entitylist: list of entities to append to
        :param x: x position
        :param y: y position
        :param d: direction (default 0, right)
        :param color: color of bot
        :param can_graze: pred or prey boolean
        :return: bot object
        """
        bot = Bot(x, y, d, self, color, can_graze)
        entitylist[bot.id] = bot
        return bot

    def get_tile_indx(self, x, y):
        tile_x = int(x / World.TILE_SIZE)
        tile_y = int(y / World.TILE_SIZE)
        return tile_x, tile_y

    def get_tile_energy(self, x, y):
        if not self.out_of_bounds(x, y):
            tile_x, tile_y = self.get_tile_indx(x, y)
            tileenergy, tiletime = self.tiles[tile_x][tile_y]
            if tiletime < self.time:
                growth = (self.time - tiletime) * World.TILE_ENERGY_RECHARGE
                tileenergy = min(tileenergy + growth, World.MAX_TILE_ENERGY)
                self.tiles[tile_x][tile_y] = (tileenergy, self.time)
            return tileenergy
        else:
            return 0

    def get_tile_perc(self,x,y):
        if not self.out_of_bounds(x,y):
            return float(self.get_tile_energy(x,y))/World.MAX_TILE_ENERGY
        else:
            return 0

    def get_vision(self, x, y, centerd, fov, maxdist, nbins):
        """
        :param x: float x position of observer
        :param y: float y position of observer
        :param centerd: float d direction of observer (degrees)
        :param fov: float field of view +- fov degrees
        :param maxdist: float max distance of sight
        :param nbins: number of bins
        :return: bins for colors + distance
        """
        bins = numpy.zeros((nbins,4))  # 3 colors + 1 distance
        bins[:,3] = 1.0  # Set distances to 1

        # Angle bin size
        bind = 2.*fov / nbins

        for entity in itertools.chain(self.predentities.values(), self.preyentities.values()):
            # Skip self
            if entity.x == x and entity.y == y:
                continue
            deltvec = numpy.array([entity.x - x, entity.y - y])
            dist = numpy.linalg.norm(deltvec)
            normdelt = deltvec/dist

            # First get angle from d = 0, then subtract d
            z_angle = numpy.rad2deg(numpy.arccos(normdelt[0]))
            if normdelt[1] < 0:
                z_angle = -z_angle
            angle = modrange(z_angle - centerd, -180, 180)

            # If angle in FOV
            if -fov < angle < fov:
                # Get bin
                binn = int((angle + fov)/bind)
                normdist = dist / maxdist
                last_ndist = bins[binn, 3]
                if 0 < normdist < last_ndist:
                    bins[binn, 0] = entity.r
                    bins[binn, 1] = entity.g
                    bins[binn, 2] = entity.b
                    bins[binn, 3] = normdist

        # Now add end-of-map information
        vlow = -Bot.FOV
        vhigh = Bot.FOV
        binangle = (vhigh - vlow) / Bot.VISION_BINS

        for binn in range(Bot.VISION_BINS):
            centerangle = binangle * (binn + 0.5) + vlow
            edgedist = self.disttoedge(x, y, centerangle + centerd) / maxdist
            if 0.0 < edgedist < bins[binn, 3]:
                bins[binn, 0:3] = 0
                bins[binn, 3] = edgedist
        return bins

    def disttoedge(self, x, y, d):
        rd = numpy.deg2rad(d)
        dx, dy = numpy.cos(rd), numpy.sin(rd)

        maxx = self.width()
        maxy = self.height()

        if dx == 0:
            lefthit, righthit = sys.maxsize, sys.maxsize
            tophit, bothit = (maxy - y) / dy, (-y) / dy
        elif dy == 0:
            lefthit, righthit = (-x) / dx, (maxx - x) / dx
            tophit, bothit = sys.maxsize, sys.maxsize
        else:
            lefthit, righthit = (-x) / dx, (maxx - x) / dx
            tophit, bothit = (maxy - y) / dy, (-y) / dy

        # Return smallest positive
        dists = list(filter(lambda s: s > 0, [lefthit, righthit, tophit, bothit]))
        if len(dists) == 0:
            return 0
        else:
            return min(dists)

    def eat(self,x,y,toeat):
        if not self.out_of_bounds(x,y):
            tile_x = int(x/World.TILE_SIZE)
            tile_y = int(y/World.TILE_SIZE)

            tileenergy, tiletime = self.tiles[tile_x][tile_y]
            if tiletime < self.time:
                growth = (self.time - tiletime) * World.TILE_ENERGY_RECHARGE
                tileenergy = min(tileenergy + growth, World.MAX_TILE_ENERGY)
            # Eats proportional to missing
            toeat = toeat * tileenergy / World.MAX_TILE_ENERGY
            # Cannot eat more than exists on tile
            toeat = min(toeat,tileenergy)
            self.tiles[tile_x][tile_y] = (tileenergy - toeat, self.time)

            self.add_to_stat('Eaten', toeat)
            return toeat
        else:
            return 0

    def out_of_bounds(self,x,y):
        inx = 0 <= x < self.width()
        iny = 0 <= y < self.height()
        return not inx or not iny

    def _clear_cache(self):
        pass

    def startup(self):
        self.predbrain.startup()
        self.preybrain.startup()

    def cleanup(self):
        self.predbrain.cleanup()
        self.preybrain.cleanup()

    def width(self):
        return self.tileshape[0] * World.TILE_SIZE

    def height(self):
        return self.tileshape[1] * World.TILE_SIZE

    def get_tile_percs(self):
        with self.lock:
            return self.tile_buffer[:]

    def get_bot_values(self):
        with self.lock:
            focussensecopy = self.focus_senses[:] if self.focus_senses is not None else None
            focusactcopy = self.focus_actvals[:] if self.focus_actvals is not None else None
            return self.entity_buffer[:], focussensecopy, focusactcopy

    def add_to_stat(self,stat,val):
        if stat not in self.stats:
            self.stats[stat] = 0
        self.stats[stat] += val

    def clear_stats(self):
        for stat in self.stats:
            self.stats[stat] = 0


def modrange(x,low,high):
    delt = high - low
    while x < low:
        x += delt
    while x > high:
        x -= delt
    return x


def update(world, iters=0, getfocus=lambda: None):
    iternum = 1
    while iternum != iters:
        world.update(1, focusid=getfocus())
        iternum += 1


def make_brain_constructor(predprey):
    """
    :param predprey: string "pred" or string "prey"
    :return:
    """
    if predprey == 'pred':
        constructor = CombinedBrain.make_combined_constructor(TFBrain, ToyBrain, 0.95)
        # constructor = TFBrain
        #constructor = ToyBrain
    else:
        constructor = PreyHeuristicBrain
        #constructor = CombinedBrain.make_combined_constructor(TFBrain, ToyBrain, 0.95)
    return constructor


def make_model():
    world = World(make_brain_constructor('pred'), make_brain_constructor('prey'))
    return world
