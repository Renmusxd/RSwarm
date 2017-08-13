from bot import Bot
import random

from threading import Lock
import numpy
import itertools


class World:
    MAX_TILE_ENERGY = 100000  # 1000 seconds at 100fps
    TILE_ENERGY_RECHARGE = 1
    TILE_SIZE = 100  # 100 times entity size
    ENTITY_SIZE = 5

    MIN_BOTS = 10
    MIN_BOT_REGEN = 2
    MAX_BOT_REGEN = 5

    TRAIN_FREQ = 1000

    PRED_COLOR = (1., 0., 0.)
    PREY_COLOR = (0., 0., 1.)

    def __init__(self, predbraincls, preybraincls, tileshape=(5,5)):
        self.time = 0
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

    def update(self,dt):
        # Kill
        predkilllist = []
        preykilllist = []
        for entity in self.predentities.values():
            if entity.dead:
                predkilllist.append(entity)
        for entity in self.preyentities.values():
            if entity.dead:
                preykilllist.append(entity)
        self._kill(predkilllist,preykilllist)  # Do any cleanup needed

        entitylists = [self.predentities, self.preyentities]
        entitycolors = [World.PRED_COLOR, World.PREY_COLOR]
        entitygrazes = [False, True]

        # Regen population for each entity list if below MIN_BOTS
        for entitylist, color, can_graze in zip(entitylists, entitycolors, entitygrazes):
            while len(entitylist) < World.MIN_BOTS:
                toadd = random.randint(World.MIN_BOT_REGEN,World.MAX_BOT_REGEN)
                x_center = random.randint(50, World.TILE_SIZE * self.tileshape[0] - 50)
                y_center = random.randint(50, World.TILE_SIZE * self.tileshape[0] - 50)
                for i in range(toadd):
                    xpos = x_center + random.randint(-50, 50)
                    ypos = y_center + random.randint(-50, 50)
                    self.make_bot(entitylist, xpos, ypos, random.randint(0, 360), color, can_graze)

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

        # Apply actions
        predrewards = {entityid: self.predentities[entityid].act(allpredactions[entityid])
                       for entityid in allpredactions.keys()}
        preyrewards = {entityid: self.preyentities[entityid].act(allpreyactions[entityid])
                       for entityid in allpreyactions.keys()}
        entityrewards = [predrewards, preyrewards]

        # Check for attacks in each list, attack closest within distance
        for entitylist, rewards in zip(entitylists, entityrewards):
            attackers = list(filter(lambda b: b.attacking, entitylist.values()))
            max_d2 = Bot.ACTION_RADIUS ** 2
            for attacker in attackers:
                closest_defender = None
                closest_distance2 = max_d2
                # Check in both lists
                for defender in itertools.chain(self.predentities.values(), self.preyentities.values()):
                    dist2 = (attacker.x - defender.x) ** 2 + (attacker.y - defender.y) ** 2
                    if 0 < dist2 <= closest_distance2:
                        closest_defender = defender
                        closest_distance2 = dist2
                if closest_defender is not None:
                    print("{} attacking {} | {} attacking {}"
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
                    print("Successful mating | {}".format("prey" if mater_a.can_graze else "pred"))
                    # Mate mater_a and mater_b
                    avg_x, avg_y = (mater_a.x + closest_mater.x) / 2., (mater_a.y + closest_mater.y) / 2.
                    child_x = avg_x + random.randint(-10, 10)
                    child_y = avg_y + random.randint(-10, 10)
                    # Make child
                    self.make_bot(entitylist, child_x, child_y, random.randint(0, 360),color, can_graze)

                    # Add rewards and notify
                    rewards[mater_a.id] += mater_a.mate_succeed(closest_mater)
                    rewards[closest_mater.id] += closest_mater.mate_succeed(mater_a)
                else:
                    rewards[mater_a.id] += mater_a.mate_failed()

        # Get sensory data from all entities
        newpredsenses = {entityid: self.predentities[entityid].senses()
                         for entityid in self.predentities.keys()}
        newpreysenses = {entityid: self.preyentities[entityid].senses()
                         for entityid in self.preyentities.keys()}
        newentitysenses = [newpredsenses, newpreysenses]

        self._clear_cache()

        # Store state, action, reward
        self.predbrain.reward(allpredsenses, allpredactions, predrewards, newpredsenses)
        self.preybrain.reward(allpreysenses, allpreyactions, preyrewards, newpreysenses)
        self.time += 1

        willtrain = (self.time % World.TRAIN_FREQ == 0)

        with self.lock:
            # Check if need to change buffer
            nentities = len(self.predentities) + len(self.preyentities)
            if nentities != len(self.entity_buffer):
                self.entity_buffer = numpy.zeros((nentities, 3 + 3))
            for i,entity in enumerate(itertools.chain(self.predentities.values(),self.preyentities.values())):
                self.entity_buffer[i, 0] = entity.x
                self.entity_buffer[i, 1] = entity.y
                self.entity_buffer[i, 2] = entity.d
                self.entity_buffer[i, 3] = entity.r
                self.entity_buffer[i, 4] = entity.g
                self.entity_buffer[i, 5] = entity.b
            for x in range(self.tileshape[0]):
                for y in range(self.tileshape[1]):
                    self.tile_buffer[x, y] = self.get_tile_perc(x * World.TILE_SIZE, y * World.TILE_SIZE)

        if willtrain:
            print("Pred: ", len(self.predentities))
            print("Prey: ", len(self.preyentities))
            print("Entering training cycle")
            self.predbrain.train()
            self.preybrain.train()
            if len(self.predentities) > 0:
                randomentity = random.choice(list(self.predentities.values()))
                self.predbrain.print_diag(randomentity.senses())
            if len(self.predentities) > 0:
                randomentity = random.choice(list(self.preyentities.values()))
                self.preybrain.print_diag(randomentity.senses())

    def _kill(self, preds, preys):
        for entity in preds:
            self.predentities.pop(entity.id)
        for entity in preys:
            self.preyentities.pop(entity.id)

    def make_pred(self, x, y, d=0.0):
        self.make_bot(self.predentities, x, y, d, World.PRED_COLOR, False)

    def make_prey(self, x, y, d=0.0):
        self.make_bot(self.preyentities, x, y, d, World.PREY_COLOR, True)

    def make_bot(self, entitylist, x, y, d, color, can_graze):
        """
        Make a new bot at a given position
        :prarm entitylist: list of entities to append to
        :param x: x position
        :param y: y position
        :param d: direction (default 0, right)
        :return: bot object
        """
        bot = Bot(x,y,d,self,color,can_graze)
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

        # Vector in direction of sight
        dirvec = numpy.array([numpy.cos(numpy.deg2rad(centerd)),
                              numpy.sin(numpy.deg2rad(centerd))])
        # Vector perpendicular (to the left) of the direction of sight
        perpdirvec = numpy.array([numpy.cos(numpy.deg2rad(centerd + 90)),
                                  numpy.sin(numpy.deg2rad(centerd + 90))])
        # Angle bin size
        bind = 2.*fov / nbins

        # I hate these indented ifs too, but I think they're necessary
        for entity in itertools.chain(self.predentities.values(), self.preyentities.values()):
            # Skip self
            if entity.x == x and entity.y == y:
                continue
            deltvec = numpy.array([entity.x - x, entity.y - y])
            dist = numpy.linalg.norm(deltvec)
            angle = numpy.arccos(dirvec.dot(deltvec) / dist) * numpy.sign(perpdirvec.dot(deltvec))
            if centerd - fov < angle < centerd + fov:
                # Get bin
                binn = int((angle - (centerd-fov))/bind)
                normdist = dist / maxdist
                last_ndist = bins[binn, 3]
                if 0 < normdist < last_ndist:
                    bins[binn, 0] = entity.r
                    bins[binn, 1] = entity.g
                    bins[binn, 2] = entity.b
                    bins[binn, 3] = normdist
        return bins

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
            return toeat
        else:
            return 0

    def out_of_bounds(self,x,y):
        inx = 0 <= x < self.tileshape[0] * World.TILE_SIZE
        iny = 0 <= y < self.tileshape[1] * World.TILE_SIZE
        return not inx or not iny

    def _clear_cache(self):
        pass

    def startup(self):
        self.predbrain.startup()
        self.preybrain.startup()

    def cleanup(self):
        self.predbrain.cleanup()
        self.preybrain.cleanup()

    def get_tile_percs(self):
        with self.lock:
            return self.tile_buffer

    def get_bot_values(self):
        with self.lock:
            return self.entity_buffer

    def savebrains(self):
        self.predbrain.save()
        self.preybrain.save()
