import numpy
from collections import OrderedDict

class Bot:
    ID_NUM = 0

    # Populated by GET_NINPUTS
    VISION = None
    INPUTS = None
    NINPUTS = None
    # Populated by GET_NACTIONS
    ACTIONS = None
    NACTIONS = None

    VIEW_DIST = 100.0
    FOV = 90
    VISION_BINS = 5

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

    ATTACK_PRED_PRED_REWARD = -20.
    ATTACK_PRED_PREY_REWARD = 50.

    ATTACK_PREY_PRED_REWARD = 5.
    ATTACK_PREY_PREY_REWARD = -20.

    ATTACKED_REWARD = -50.
    ATTACK_FAILED_REWARD = -5.
    EAT_REWARD = 100.  # Scaled by hunger: R (E - e) / E
    MATE_REWARD = 100.
    FAILED_MATE_REWARD = -10.

    def __init__(self, x, y, d, world, color, can_graze, energy=MAX_ENERGY):
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

    def senses(self):
        # Evaluate vision
        vision = Bot.VISION.eval(self)
        # Evaluate introspection
        body = numpy.array([v(self) for v in Bot.INPUTS.values()])

        state = numpy.concatenate((body, vision))

        return state

    def act(self, action):
        reward_acc = 0

        still, left, lmov, forward, \
        rmov, right, sprint, eat, \
        mate, atck = (action == i for i in range(Bot.GET_NACTIONS()))

        if eat:
            if self.can_graze:
                toeat = min(Bot.EAT_AMOUNT, Bot.MAX_ENERGY - self.energy)
                eaten = self.world.eat(self.x,self.y,toeat)
                self.energy += eaten
                # reward_acc += eaten/Bot.EAT_AMOUNT * (Bot.MAX_ENERGY - self.energy)/Bot.MAX_ENERGY * Bot.EAT_REWARD
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
        """
        Callback for successful attacks
        :param other:
        :return: Reward
        """
        self.attacking = False
        if self.can_graze:
            return Bot.ATTACK_PREY_PREY_REWARD if other.can_graze else Bot.ATTACK_PREY_PRED_REWARD
        else:
            self.energy += 10*Bot.MAX_ENERGY/2
            other.energy -= Bot.MAX_ENERGY/2
            return Bot.ATTACK_PRED_PREY_REWARD if other.can_graze else Bot.ATTACK_PRED_PRED_REWARD

    def attack_failed(self):
        self.attacking = False
        return Bot.ATTACK_FAILED_REWARD

    def was_attacked(self, other):
        return Bot.ATTACKED_REWARD

    @staticmethod
    def split_senses(senses):
        """
        Splits senses into introspection senses and vision
        :param senses: raw input
        :return: inputs, vision, distance
        """
        ins = senses[:len(Bot.INPUTS)]
        vis, dist = Bot.VISION.split_vision(senses[len(Bot.INPUTS):])
        return ins, vis, dist

    @staticmethod
    def label_inputs(inputs):
        return {k:v for k,v in zip(Bot.INPUTS.keys(),inputs)}

    @staticmethod
    def make_brain(braincons, name):
        """
        Make a brain suitable for this bot
        :param name: brain name
        :param braincons: brain constructor function
        :return: instance of brain to use
        """
        brain = braincons(name, Bot.GET_NINPUTS(), Bot.GET_NACTIONS())
        return brain

    @staticmethod
    def GET_NINPUTS():
        if Bot.INPUTS is None:
            Bot.INPUTS = OrderedDict()

            # Basic senses
            Bot.INPUTS['energy'] = lambda b: b.energy / Bot.MAX_ENERGY
            Bot.INPUTS['age'] = lambda b: b.age / Bot.MAX_AGE
            Bot.INPUTS['mate'] = lambda b: b.mate_timer / Bot.MATE_TIMER
            Bot.INPUTS['tile'] = lambda b: b.world.get_tile_perc(b.x,b.y)

            # Vision
            Bot.VISION = BotVision("gray")

            Bot.NINPUTS = len(Bot.INPUTS) + len(Bot.VISION)
        return Bot.NINPUTS

    @staticmethod
    def GET_NACTIONS():
        if Bot.ACTIONS is None:
            Bot.ACTIONS = ["still", "left", "lmov", "forward",
                           "rmov", "right", "sprint", "eat",
                           "mate", "atck"]
            Bot.NACTIONS = len(Bot.ACTIONS)
        return Bot.NACTIONS


class BotVision:
    GRAY_SIZE = 2
    RGB_SIZE = 4

    def __init__(self,color='gray'):
        """
        Construct vision mechanic
        :param vbins: number of vision bins
        :param fov: field of view in degrees
        :param color: color format to use (gray or rgb)
        """
        self.color = color
        if self.color == 'gray':
            self.size = Bot.VISION_BINS * BotVision.GRAY_SIZE
            self.shape = (Bot.VISION_BINS, BotVision.GRAY_SIZE)
        elif self.color == 'rgb':
            self.size = Bot.VISION_BINS * BotVision.RGB_SIZE
            self.shape = (Bot.VISION_BINS, BotVision.RGB_SIZE)

    def eval(self, bot):
        # Gets back 3 colors + 1 distance
        vision = bot.world.get_vision(bot.x, bot.y, bot.d, Bot.FOV, Bot.VIEW_DIST, Bot.VISION_BINS)
        if self.color == "gray":
            # Convert to [-1, 1] scale
            vscale = (-vision[:, 0] + vision[:, 2])
            distances = vision[:, 3]
            return numpy.concatenate((vscale,distances))
        else:
            return vision.flatten()

    def split_vision(self, vision):
        """
        Split into vision and distance components
        :param vision: raw vision input (as is output from eval)
        :return: vision, distance
        """
        vis = vision.reshape(self.shape)
        return vis[:,:-1], vis[:,-1]

    def __len__(self):
        return self.size
