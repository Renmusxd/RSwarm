import numpy


class Bot:
    ID_NUM = 0
    VISION_BINS = 5

    # energy + age + food_below + vision
    NINPUTS = 4 + VISION_BINS*4
    NACTIONS = 10

    VIEW_DIST = 300.0
    FOV = 90

    MAX_AGE = 10000
    MATE_TIMER = 100

    MAX_ENERGY = 100
    MOVE_SPEED = 1.0
    SPRINT_SPEED = 3.0
    TURN_SPEED = 5.0

    # Radius for actions like attacking and mating
    ACTION_RADIUS = 10

    EAT_AMOUNT = 20

    # Rewards
    DEATH_REWARD = -100.
    ATTACK_PRED_REWARD = 20.
    ATTACK_PREY_REWARD = 0.
    ATTACKED_REWARD = -50.
    ATTACK_FAILED_REWARD = -5.
    EAT_REWARD = 100.  # Scaled by hunger: R (E - e) / E
    MATE_REWARD = 100.
    FAILED_MATE_REWARD = -10.

    def __init__(self, x, y, d, world, color, can_graze, energy=MAX_ENERGY, ):
        """
        Construct a bot
        :param x: x position
        :param y: y position
        :param d: direction (0-360)[OPENGL]
        :param world: world to ask for information
        """
        self.x, self.y, self.d = x, y, d
        self.world = world
        self.id = Bot.ID_NUM
        Bot.ID_NUM += 1

        self.can_graze = can_graze
        self.energy = energy
        self.r, self.g, self.b = color
        self.dead = False

        # Indicate that this Bot is attempting to mate
        self.mating = False
        self.attacking = False
        self.age = 0
        self.mate_timer = 0

    @classmethod
    def make_brain(cls, braincls):
        """
        Make a brain suitable for this bot
        :param braincls: class of brain to construct
        :return: instance of brain to use
        """
        brain = braincls(Bot.NINPUTS, Bot.NACTIONS)
        return brain

    def senses(self):
        vision = self.world.get_vision(self.x,self.y,self.d,Bot.FOV,Bot.VIEW_DIST,Bot.VISION_BINS)
        fvision = vision.flatten()
        body = numpy.array([
            self.energy/Bot.MAX_ENERGY,
            self.age/Bot.MAX_AGE,
            self.mate_timer/Bot.MATE_TIMER,
            self.world.get_tile_perc(self.x,self.y)
        ])
        return numpy.concatenate((body, fvision))

    def act(self, action):

        reward_acc = 0

        still, left, lmov, forward, rmov, right, sprint, eat, mate, atck = (action == i for i in range(Bot.NACTIONS))

        if eat and self.can_graze:
            toeat = min(Bot.EAT_AMOUNT, Bot.MAX_ENERGY - self.energy)
            eaten = self.world.eat(self.x,self.y,toeat)
            self.energy += eaten
            # reward_acc += (eaten/Bot.EAT_AMOUNT) * ((Bot.MAX_ENERGY - self.energy)/Bot.MAX_ENERGY) * Bot.EAT_REWARD
            reward_acc += eaten * Bot.EAT_REWARD * (Bot.MAX_ENERGY - self.energy)/(Bot.EAT_AMOUNT * Bot.MAX_ENERGY)
            self.energy -= 1
        elif mate:
            # Check if meets mating criteria
            # Reward will be added later if mate is successful
            if self.mate_timer == Bot.MATE_TIMER and self.energy > Bot.MAX_ENERGY/2:
                self.mating = True
            self.energy -= 1
        elif atck:
            self.attacking = True
        elif sprint:
            self.x += Bot.SPRINT_SPEED * numpy.cos(numpy.deg2rad(self.d))
            self.y += Bot.SPRINT_SPEED * numpy.sin(numpy.deg2rad(self.d))
            self.energy -= Bot.SPRINT_SPEED
        elif not still:
            if left or lmov:
                self.d -= Bot.TURN_SPEED
            elif right or rmov:
                self.d += Bot.TURN_SPEED
            if lmov or forward or rmov:
                self.x += Bot.MOVE_SPEED * numpy.cos(numpy.deg2rad(self.d))
                self.y += Bot.MOVE_SPEED * numpy.sin(numpy.deg2rad(self.d))
            self.energy -= 1

        self.age += 1
        self.mate_timer += 1

        self.mate_timer = min(self.mate_timer, Bot.MATE_TIMER)

        # Punish death
        if self.energy <= 0 or self.world.out_of_bounds(self.x,self.y) or self.age >= Bot.MAX_AGE:
            reward_acc += self.DEATH_REWARD
            self.dead = True
        return reward_acc

    def color(self):
        return self.r, self.g, self.b

    def mate_succeed(self, other_bot):
        self.mating = False
        self.mate_timer = 0
        self.energy -= Bot.MAX_ENERGY/2
        return Bot.MATE_REWARD

    def mate_failed(self):
        self.mating = False
        return Bot.FAILED_MATE_REWARD

    def attack_succeed(self, other):
        if self.can_graze:
            self.attacking = False
            return Bot.ATTACK_PREY_REWARD
        else:
            self.energy += 10*Bot.MAX_ENERGY/2
            other.energy -= Bot.MAX_ENERGY/2
            self.attacking = False
            return Bot.ATTACK_PRED_REWARD

    def attack_failed(self):
        self.attacking = False
        return Bot.ATTACK_FAILED_REWARD

    def was_attacked(self, other):
        return Bot.ATTACKED_REWARD

