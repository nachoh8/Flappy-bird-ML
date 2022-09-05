from itertools import cycle
import random
import sys
import pygame
from pygame.locals import *

import numpy as np

from ML.geneticNN import GeneticNN

gnn_params = dict()
gnn_params['population'] = 50
gnn_params['max_generation'] = -1
gnn_params['max_fitness'] = -1
gnn_params['elitism'] = 0.2
gnn_params['mutation_rate'] = 0.1
gnn_params['checkpoint'] = 5
gnn_params['checkpoint_dir'] = 'checkpoints/test_3'

nn_params = dict()
nn_params['layers'] = [2,6,1]
nn_params['activation'] = 'sigmoid'
nn_params['weights'] = []

gnn_params['nn_params'] = nn_params

GNN = GeneticNN(gnn_params)
GNN.load_checkpoint(GNN.checkpoint_dir + '/checkpoint_10_old.json')

FPS = 30
SCREENWIDTH  = 288.0
SCREENHEIGHT = 512.0
PIPEGAPSIZE  = 100.0 # gap between upper and lower part of pipe
BASEY        = SCREENHEIGHT * 0.79
# image and hitmask  dicts
IMAGES, HITMASKS = {}, {}

# list of all possible players (tuple of 3 positions of flap)
PLAYERS_LIST = (
    # red bird
    (
        'assets/sprites/redbird-upflap.png',
        'assets/sprites/redbird-midflap.png',
        'assets/sprites/redbird-downflap.png',
    ),
    # blue bird
    (
        'assets/sprites/bluebird-upflap.png',
        'assets/sprites/bluebird-midflap.png',
        'assets/sprites/bluebird-downflap.png',
    ),
    # yellow bird
    (
        'assets/sprites/yellowbird-upflap.png',
        'assets/sprites/yellowbird-midflap.png',
        'assets/sprites/yellowbird-downflap.png',
    ),
)

# list of backgrounds
BACKGROUNDS_LIST = (
    'assets/sprites/background-day.png',
    'assets/sprites/background-night.png',
)

# list of pipes
PIPES_LIST = (
    'assets/sprites/pipe-green.png',
    'assets/sprites/pipe-red.png',
)

MAX_SCORE = (0,0) # score, generation


try:
    xrange
except NameError:
    xrange = range


def main():
    global SCREEN, FPSCLOCK
    pygame.init()
    FPSCLOCK = pygame.time.Clock()
    SCREEN = pygame.display.set_mode((SCREENWIDTH, SCREENHEIGHT))
    pygame.display.set_caption('Flappy Bird')

    # numbers sprites for score display
    IMAGES['numbers'] = (
        pygame.image.load('assets/sprites/0.png').convert_alpha(),
        pygame.image.load('assets/sprites/1.png').convert_alpha(),
        pygame.image.load('assets/sprites/2.png').convert_alpha(),
        pygame.image.load('assets/sprites/3.png').convert_alpha(),
        pygame.image.load('assets/sprites/4.png').convert_alpha(),
        pygame.image.load('assets/sprites/5.png').convert_alpha(),
        pygame.image.load('assets/sprites/6.png').convert_alpha(),
        pygame.image.load('assets/sprites/7.png').convert_alpha(),
        pygame.image.load('assets/sprites/8.png').convert_alpha(),
        pygame.image.load('assets/sprites/9.png').convert_alpha()
    )

    # game over sprite
    IMAGES['gameover'] = pygame.image.load('assets/sprites/gameover.png').convert_alpha()
    # message sprite for welcome screen
    IMAGES['message'] = pygame.image.load('assets/sprites/message.png').convert_alpha()
    # base (ground) sprite
    IMAGES['base'] = pygame.image.load('assets/sprites/base.png').convert_alpha()

    if GNN.current_generation == 0:
        GNN.first_generation()
    end = False
    while not end:
        # select random background sprites
        randBg = random.randint(0, len(BACKGROUNDS_LIST) - 1)
        IMAGES['background'] = pygame.image.load(BACKGROUNDS_LIST[randBg]).convert()

        # select random player sprites
        randPlayer = random.randint(0, len(PLAYERS_LIST) - 1)
        IMAGES['player'] = (
            pygame.image.load(PLAYERS_LIST[randPlayer][0]).convert_alpha(),
            pygame.image.load(PLAYERS_LIST[randPlayer][1]).convert_alpha(),
            pygame.image.load(PLAYERS_LIST[randPlayer][2]).convert_alpha(),
        )

        # select random pipe sprites
        pipeindex = random.randint(0, len(PIPES_LIST) - 1)
        IMAGES['pipe'] = (
            pygame.transform.flip(
                pygame.image.load(PIPES_LIST[pipeindex]).convert_alpha(), False, True),
            pygame.image.load(PIPES_LIST[pipeindex]).convert_alpha(),
        )

        # hitmask for pipes
        HITMASKS['pipe'] = (
            getHitmask(IMAGES['pipe'][0]),
            getHitmask(IMAGES['pipe'][1]),
        )

        # hitmask for player
        HITMASKS['player'] = (
            getHitmask(IMAGES['player'][0]),
            getHitmask(IMAGES['player'][1]),
            getHitmask(IMAGES['player'][2]),
        )

        movementInfo = showWelcomeAnimation()
        crashInfo = mainGame(movementInfo)
        end = not showGameOverScreen(crashInfo)


def showWelcomeAnimation():
    """Shows welcome screen animation of flappy bird"""
    # index of player to blit on screen
    playerIndex = 0
    playerIndexGen = cycle([0, 1, 2, 1])
    # iterator used to change playerIndex after every 5th iteration
    loopIter = 0

    playerx = int(SCREENWIDTH * 0.2)
    playery = int((SCREENHEIGHT - IMAGES['player'][0].get_height()) / 2)

    messagex = int((SCREENWIDTH - IMAGES['message'].get_width()) / 2)
    messagey = int(SCREENHEIGHT * 0.12)

    basex = 0
    # amount by which base can maximum shift to left
    baseShift = IMAGES['base'].get_width() - IMAGES['background'].get_width()

    # player shm for up-down motion on welcome screen
    playerShmVals = {'val': 0, 'dir': 1}

    first_gen = GNN.current_generation == 1
    t = 0
    while first_gen or (t < 10 and not first_gen):
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                pygame.quit()
                sys.exit()
            if event.type == KEYDOWN and (event.key == K_SPACE or event.key == K_UP):
                # make first flap sound and return values for mainGame
                return {
                    'playery': playery + playerShmVals['val'],
                    'basex': basex,
                    'playerIndexGen': playerIndexGen,
                }

        # adjust playery, playerIndex, basex
        if (loopIter + 1) % 5 == 0:
            playerIndex = next(playerIndexGen)
        loopIter = (loopIter + 1) % 30
        basex = -((-basex + 4) % baseShift)
        playerShm(playerShmVals)

        # draw sprites
        SCREEN.blit(IMAGES['background'], (0,0))
        SCREEN.blit(IMAGES['player'][playerIndex],
                    (playerx, playery + playerShmVals['val']))
        SCREEN.blit(IMAGES['message'], (messagex, messagey))
        SCREEN.blit(IMAGES['base'], (basex, BASEY))

        t += 1
        pygame.display.update()
        FPSCLOCK.tick(FPS)
    
    return {
        'playery': playery + playerShmVals['val'],
        'basex': basex,
        'playerIndexGen': playerIndexGen,
    }


def mainGame(movementInfo):
    global MAX_SCORE
    playerIndex = loopIter = 0
    playerIndexGen = movementInfo['playerIndexGen']
    playerx, playery = int(SCREENWIDTH * 0.2), movementInfo['playery']

    basex = movementInfo['basex']
    baseShift = IMAGES['base'].get_width() - IMAGES['background'].get_width()

    # get 2 new pipes to add to upperPipes lowerPipes list
    newPipe1 = getRandomPipe()
    newPipe2 = getRandomPipe()

    # list of upper pipes
    upperPipes = [
        {'x': SCREENWIDTH + 200, 'y': newPipe1[0]['y']},
        {'x': SCREENWIDTH + 200 + (SCREENWIDTH / 2), 'y': newPipe2[0]['y']},
    ]

    # list of lowerpipe
    lowerPipes = [
        {'x': SCREENWIDTH + 200, 'y': newPipe1[1]['y']},
        {'x': SCREENWIDTH + 200 + (SCREENWIDTH / 2), 'y': newPipe2[1]['y']},
    ]

    dt = FPSCLOCK.tick(FPS)/1000
    pipeVelX = -128 * dt

    # player velocity, max velocity, downward acceleration, acceleration on flap
    playerVelY    =  -9.0   # player's velocity along Y, default same as playerFlapped
    playerMaxVelY =  10.0   # max vel along Y, max descend speed
    playerAccY    =   1.0   # players downward acceleration
    playerFlapAcc =  -9.0   # players speed on flapping
    # playerFlapped = False # True when player flaps

    alive_ids = np.full(GNN.population, True)
    list_alive = np.where(alive_ids)[0].tolist()
    fitness = np.zeros(GNN.population)
    score_vec = np.zeros(GNN.population)

    flapped = np.full(GNN.population, False)

    y_pos = np.full(GNN.population, float(playery)) # position on Y
    y_vel = np.full(GNN.population, playerVelY) # velocity on Y
    
    PLAYER_W_2 = IMAGES['player'][0].get_width() / 2
    PLAYER_H_2 = IMAGES['player'][0].get_height() / 2

    PIPE_W = IMAGES['pipe'][0].get_width()
    PIPE_W_2 = PIPE_W / 2
    PIPEGAPSIZE_2 = PIPEGAPSIZE / 2

    pipe_idx = 0
    pipe_gap_x = lowerPipes[pipe_idx]['x'] + PIPE_W
    pipe_gap_y = lowerPipes[pipe_idx]['y'] - PIPEGAPSIZE_2
    
    MAX_PIPE_DIST = pipe_gap_x - playerx + PLAYER_W_2

    h_dist = (pipe_gap_x - playerx + PLAYER_W_2) / MAX_PIPE_DIST # horizontal distance to pipe gap
    v_dist = np.full(GNN.population, (pipe_gap_y - y_pos + PLAYER_H_2) / SCREENHEIGHT) # vertical distance to pipe gap

    print("----------------------------------------------------")
    print("Generation:", GNN.current_generation)

    while True:
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                pygame.quit()
                sys.exit()
            """if event.type == KEYDOWN and (event.key == K_SPACE or event.key == K_UP): # TODO: remove keyspace control
                idxs = (alive_ids) & (y_pos > -2 * IMAGES['player'][0].get_height())
                y_vel[idxs] = playerFlapAcc
                flapped[idxs] = True"""

        for idx in list_alive:
            input = np.array([h_dist, v_dist[idx]])
            out = GNN.predict(idx, input)[0]
            if out > 0.5 and (y_pos[idx] > -2 * IMAGES['player'][0].get_height()):
                y_vel[idx] = playerFlapAcc
                flapped[idx] = True

        # check for crash here
        checkCrash({'x': playerx, 'y': y_pos, 'index': playerIndex, "alive": alive_ids}, upperPipes, lowerPipes)
        list_alive = np.where(alive_ids)[0].tolist()
        
        if len(list_alive) == 0:
            print([(idx, fitness[idx], score_vec[idx]) for idx in range(GNN.population)])

            return {
                'basex': basex,
                'upperPipes': upperPipes,
                'lowerPipes': lowerPipes,
                'score': score_vec,
                'fitness': fitness
            }

        ### CHECK SCORE ###
        playerMidPos = playerx + PLAYER_W_2
        for pipe in upperPipes:
            pipeMidPos = pipe['x'] + IMAGES['pipe'][0].get_width() / 2
            if pipeMidPos <= playerMidPos < pipeMidPos + 4:
                score_vec[alive_ids] += 1
            if pipe_gap_x <= playerMidPos < pipeMidPos + 4:
                pipe_idx = 1

        # playerIndex basex change
        if (loopIter + 1) % 3 == 0:
            playerIndex = next(playerIndexGen)
        loopIter = (loopIter + 1) % 30
        basex = -((-basex + 100) % baseShift)

        ### MOVE PLAYERS ###
        y_vel[(alive_ids) & (y_vel < playerMaxVelY) & (~flapped)] += playerAccY
        flapped[(alive_ids) & flapped] = False

        playerHeight = IMAGES['player'][playerIndex].get_height()
        y_pos[alive_ids] += np.minimum(y_vel[alive_ids], BASEY - y_pos[alive_ids] - playerHeight)

        ### MOVE PIPES AND UPDATE GAP/DISTANCE ###
        for uPipe, lPipe in zip(upperPipes, lowerPipes):
            uPipe['x'] += pipeVelX
            lPipe['x'] += pipeVelX

        pipe_gap_x = lowerPipes[pipe_idx]['x'] + PIPE_W
        pipe_gap_y = lowerPipes[pipe_idx]['y'] - PIPEGAPSIZE_2

        h_dist = (pipe_gap_x - playerx + PLAYER_W_2) / MAX_PIPE_DIST # horizontal distance to pipe gap
        v_dist[alive_ids] = (pipe_gap_y - y_pos[alive_ids] + PLAYER_H_2) / SCREENHEIGHT # vertical distance to pipe gap

        # add new pipe when first pipe is about to touch left of screen
        if 3 > len(upperPipes) > 0 and 0 < upperPipes[0]['x'] < 5:
            newPipe = getRandomPipe()
            upperPipes.append(newPipe[0])
            lowerPipes.append(newPipe[1])

        # remove first pipe if its out of the screen
        if len(upperPipes) > 0 and upperPipes[0]['x'] < -IMAGES['pipe'][0].get_width():
            upperPipes.pop(0)
            lowerPipes.pop(0)
            pipe_idx = 0

        ### DRAW SPRITES ###
        SCREEN.blit(IMAGES['background'], (0,0))

        for uPipe, lPipe in zip(upperPipes, lowerPipes):
            SCREEN.blit(IMAGES['pipe'][0], (uPipe['x'], uPipe['y']))
            SCREEN.blit(IMAGES['pipe'][1], (lPipe['x'], lPipe['y']))

        SCREEN.blit(IMAGES['base'], (basex, BASEY))
        
        # print score so player overlaps the score
        showScore(int(np.max(score_vec)))
        
        # draw alive players
        for idx in list_alive:
            SCREEN.blit(IMAGES['player'][playerIndex], (playerx, y_pos[idx]))

        # draw info
        font = pygame.font.SysFont(None, 24)
        img_gen = font.render('Generation: ' + str(GNN.current_generation), True, (0,0,0))
        SCREEN.blit(img_gen, (5, 5))
        img_alive = font.render('Alive: ' + str(len(list_alive)) + '/' + str(GNN.population), True, (0,0,0))
        SCREEN.blit(img_alive, (5, 20))
        img_best_score = font.render('Best score: ' + str(MAX_SCORE[0]) + '(Gen ' + str(MAX_SCORE[1]) + ')', True, (0,0,0))
        SCREEN.blit(img_best_score, (5, 35))

        pygame.draw.circle(SCREEN, (255,0,0), (pipe_gap_x, pipe_gap_y), 5)

        pygame.display.update()
        FPSCLOCK.tick(FPS)
        
        fitness[alive_ids] += 1


def showGameOverScreen(crashInfo):
    global MAX_SCORE

    """crashes the player down and shows gameover image"""
    score = crashInfo['score']
    max_score = int(np.max(score))
    fitness = crashInfo['fitness']

    if max_score > MAX_SCORE[0]:
        MAX_SCORE = (max_score, GNN.current_generation)

    next_gen, best = GNN.next_generation(fitness)
    print("BEST PLAYER:", best[0], "FITNESS:", best[2], "SCORE:", score[best[0]])

    basex = crashInfo['basex']

    upperPipes, lowerPipes = crashInfo['upperPipes'], crashInfo['lowerPipes']

    t = 0
    while t < 10:
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                pygame.quit()
                sys.exit()
            if event.type == KEYDOWN and (event.key == K_SPACE or event.key == K_UP):
                return next_gen

        # draw sprites
        SCREEN.blit(IMAGES['background'], (0,0))

        for uPipe, lPipe in zip(upperPipes, lowerPipes):
            SCREEN.blit(IMAGES['pipe'][0], (uPipe['x'], uPipe['y']))
            SCREEN.blit(IMAGES['pipe'][1], (lPipe['x'], lPipe['y']))

        SCREEN.blit(IMAGES['base'], (basex, BASEY))
        showScore(max_score)

        SCREEN.blit(IMAGES['gameover'], (50, 180))

        t += 1
        FPSCLOCK.tick(FPS)
        pygame.display.update()
    
    return next_gen


def playerShm(playerShm):
    """oscillates the value of playerShm['val'] between 8 and -8"""
    if abs(playerShm['val']) == 8:
        playerShm['dir'] *= -1

    if playerShm['dir'] == 1:
         playerShm['val'] += 1
    else:
        playerShm['val'] -= 1


def getRandomPipe():
    """returns a randomly generated pipe"""
    # y of gap between upper and lower pipe
    gapY = random.randrange(0, int(BASEY * 0.6 - PIPEGAPSIZE))
    gapY += int(BASEY * 0.2)
    pipeHeight = IMAGES['pipe'][0].get_height()
    pipeX = SCREENWIDTH + 10

    return [
        {'x': pipeX, 'y': gapY - pipeHeight},  # upper pipe
        {'x': pipeX, 'y': gapY + PIPEGAPSIZE}, # lower pipe
    ]


def showScore(score):
    """displays score in center of screen"""
    scoreDigits = [int(x) for x in list(str(score))]
    totalWidth = 0 # total width of all numbers to be printed

    for digit in scoreDigits:
        totalWidth += IMAGES['numbers'][digit].get_width()

    Xoffset = (SCREENWIDTH - totalWidth) / 2

    for digit in scoreDigits:
        SCREEN.blit(IMAGES['numbers'][digit], (Xoffset, SCREENHEIGHT * 0.1))
        Xoffset += IMAGES['numbers'][digit].get_width()


def checkCrash(player, upperPipes, lowerPipes):
    """returns True if player collides with base or pipes."""
    pi = player['index']
    player['w'] = IMAGES['player'][0].get_width()
    player['h'] = IMAGES['player'][0].get_height()

    alive_ids = player["alive"]
    y_pos = player['y']

    # if player crashes into ground
    alive_ids[(alive_ids) & ((y_pos < 0) | (y_pos + player['h'] >= BASEY - 1))] = False

    # if not
    pipeW = IMAGES['pipe'][0].get_width()
    pipeH = IMAGES['pipe'][0].get_height()
    for idx in np.where(alive_ids)[0].tolist():
        playerRect = pygame.Rect(player['x'], y_pos[idx],
                      player['w'], player['h'])
        
        for uPipe, lPipe in zip(upperPipes, lowerPipes):
            # upper and lower pipe rects
            uPipeRect = pygame.Rect(uPipe['x'], uPipe['y'], pipeW, pipeH)
            lPipeRect = pygame.Rect(lPipe['x'], lPipe['y'], pipeW, pipeH)

            # player and upper/lower pipe hitmasks
            pHitMask = HITMASKS['player'][pi]
            uHitmask = HITMASKS['pipe'][0]
            lHitmask = HITMASKS['pipe'][1]

            # if bird collided with upipe or lpipe
            uCollide = pixelCollision(playerRect, uPipeRect, pHitMask, uHitmask)
            lCollide = pixelCollision(playerRect, lPipeRect, pHitMask, lHitmask)

            if uCollide or lCollide:
                alive_ids[idx] = False


def pixelCollision(rect1, rect2, hitmask1, hitmask2):
    """Checks if two objects collide and not just their rects"""
    rect = rect1.clip(rect2)

    if rect.width == 0 or rect.height == 0:
        return False

    x1, y1 = rect.x - rect1.x, rect.y - rect1.y
    x2, y2 = rect.x - rect2.x, rect.y - rect2.y

    for x in xrange(rect.width):
        for y in xrange(rect.height):
            if hitmask1[x1+x][y1+y] and hitmask2[x2+x][y2+y]:
                return True
    return False


def getHitmask(image):
    """returns a hitmask using an image's alpha."""
    mask = []
    for x in xrange(image.get_width()):
        mask.append([])
        for y in xrange(image.get_height()):
            mask[x].append(bool(image.get_at((x,y))[3]))
    return mask


if __name__ == '__main__':
    main()
