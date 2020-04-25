import pygame
import numpy as np

class Bar:
    def __init__(self, x, y, lenght = 20, width = 2, velocity = 2, orientation = 1):
        self.x = int(x)
        self.y = int(y)
        self.lenght = lenght
        self.width = width
        self.velocity = velocity
        self.orientation = orientation # 1 para horizontal, 0 para vertical

    def draw(self, screen, color = (255,255,255)): # desenhar em pygame
        pygame.draw.rect(screen, color, [self.x-self.width/2, self.y-self.lenght/2, self.width, self.lenght])

    def move(self, mode='human', move=None, ball = None): #mode = (human, machine, enemy); move = (0,1,2)
        lookup_table = {pygame.K_s : lambda x: x + self.velocity,
                        1 : lambda x: x + self.velocity, # movimentamos a barra verticalmente
                        pygame.K_w : lambda x: x - self.velocity,
                        2 : lambda x: x - self.velocity} # conforme a tabela indica

        # modos de movimento: o mode 'human' serve para o controle manual,
        # 'machine' diz respeito ao environment e o 'enemy' serve para controlar
        # a barra inimiga
        if mode == 'human':
            pressed = pygame.key.get_pressed()
            for k in lookup_table.keys(): # verificamos se a tecla foi apertada
                if pressed[k]:
                    self.y = lookup_table[k](self.y)
            # clamping
            if self.y >= 600:
                self.y = 600
            elif self.y <= 0:
                self.y = 0


        elif mode == 'machine':
            if move != 0:
                self.y = lookup_table[move](self.y)
            #clamp
            if self.y >= 600:
                self.y = 600
            elif self.y <= 0:
                self.y = 0

        elif mode == 'enemy':
            if self.y != ball.y and np.random.random() < .6 and ball.x >= 400: vec = ((ball.y - self.y)/abs(ball.y - self.y))
            else: vec = 0
            self.y += self.velocity*vec


class Ball:
    def __init__(self, x, y, radius):
        self.x = int(x)
        self.y = int(y)
        self.radius = radius
        rr = [(-1,-1)] # adicione mais velocidades!
        r = np.random.choice(range(len(rr)))
        self.velocity = [rr[r][0],rr[r][1]]

    def move(self):
        self.x = self.x + self.velocity[0]
        self.y = self.y + self.velocity[1]

    def draw(self,screen,color = (255,255,255)):
        pygame.draw.circle(screen, color, [int(self.x), int(self.y)], self.radius)

    def bounce(self, wall):
        lookup_table = {0:[-1,1],
                        1:[1,-1]}
        if abs(self.x - wall.x) <= wall.width/2 and abs(self.y - wall.y) <= wall.lenght/2:
            self.velocity[0] *= lookup_table[wall.orientation][0]
            self.velocity[1] *= lookup_table[wall.orientation][1]

class Environment:
    def __init__(self, HEIGHT=600, WIDTH=800, bar_velocity=3, max_steps = 1000000):

        bar_parameters = [(15,50,100,5,bar_velocity,0),(WIDTH-15,50,100,5,3,0),
                  (WIDTH/2,0,2,WIDTH,0,1),(WIDTH/2,HEIGHT,2,WIDTH,0,1),
                  (0,HEIGHT/2,HEIGHT,2,0,0),(WIDTH,HEIGHT/2,HEIGHT,2,0,0)]

        self.HEIGHT = HEIGHT
        self.WIDTH = WIDTH
        self.max_steps = max_steps
        self.rendered = False

        self.bars = []
        for bar in bar_parameters:
            self.bars.append(Bar(bar[0],bar[1],bar[2],bar[3],bar[4],orientation=bar[-1]))
        self.control_bar = self.bars[0]
        self.other_bar = self.bars[1]

        self.ball = Ball(WIDTH/2,HEIGHT/2,10) #x inicial; y inicial; raio

    def reset(self):
        
        self.ball.x, self.ball.y = self.WIDTH/2, self.HEIGHT/2
        self.steps = 0
        self.control_bar.x, self.control_bar.y = 15,50
        self.other_bar.x, self.other_bar.y = self.WIDTH - 15,50
        rr = [(-1,-1)]
        r = np.random.choice(range(len(rr)))
        self.ball.velocity = [rr[r][0],rr[r][1]]
        self.done = False
        self.score = [0,0]
        
        dx = self.control_bar.x - self.ball.x
        dy = self.control_bar.y - self.ball.y
        
        return ((dx,dy))

    def step(self,action):

        reward = 0
        self.steps += 1
        self.control_bar.move(mode='machine',move=action)
        self.other_bar.move(mode='enemy',ball=self.ball)
        self.ball.move()

        for bar in self.bars:
            self.ball.bounce(bar)

        if self.ball.x <= 4:

            self.ball.x, self.ball.y = self.WIDTH/2, self.HEIGHT/2
            self.control_bar.x, self.control_bar.y = 15,50
            self.other_bar.x, self.other_bar.y = self.WIDTH - 15,50
            self.ball.velocity = [-1,-1]

            self.score[1] += 1
            reward = -500
            if self.score[-1] >= 5: self.done = True; reward -= 5000

        elif self.ball.x >= self.WIDTH - 4:

            self.ball.x, self.ball.y = self.WIDTH/2, self.HEIGHT/2
            self.control_bar.x, self.control_bar.y = 15,50
            self.other_bar.x, self.other_bar.y = self.WIDTH - 15,50
            self.ball.velocity = [-1,-1]
            
            self.score[0] += 1
            reward = +5000
            if self.score[0] >= 5: self.done = True; reward += self.max_steps

        if self.steps >= self.max_steps:
            self.done = True
        
        dx = self.control_bar.x - self.ball.x
        dy = self.control_bar.y - self.ball.y
        
        return ((dx,dy), 1 + reward, self.done, '_')

    def render(self):
        if not self.rendered:
            self.screen = pygame.display.set_mode((self.WIDTH,self.HEIGHT))
            self.rendered = True
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.done = True
        self.screen.fill((100,100,100))
        for bar in self.bars:
            bar.draw(self.screen)
        self.ball.draw(self.screen)
        pygame.display.update()