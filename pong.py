import numpy as np
import time

def range_overlap(a1,a2,b1,b2):
    #print(a1,a2,b1,b2,a1 <= b2 and b1 <= a2)

    return a1 <= b2 and b1 <= a2

class Pong:
    def __init__(self):
        self.GOAL_THRESHOLD = np.array(1.1)
        self.GAME_SPEED = 1.0/40.0
    
        self.left_paddle = np.array([-1.0,0.0])
        self.right_paddle = np.array([1.0,0.0])
        self.ball = np.array([0.0,0.0])
        self.ball_speed = self.get_reset_speed()
        
        self.paddle_width = np.array(0.2)
        
        self.score = np.array([0,0])
        
        
    def get_reset_speed(self):
        arr = np.array([np.random.randint(low=0,high=2,size=[]).astype(np.float)*2.0-1.0,np.random.uniform(low=-2.0,high=2.0,size=[])])
        return arr
        
    def step(self, lp_delta, rp_delta):
        self.left_paddle[1] = np.clip(self.left_paddle[1]+lp_delta,-1.0,1.0)
        self.right_paddle[1] = np.clip(self.left_paddle[1]+rp_delta,-1.0,1.0)
        
        new_ball = self.ball+self.ball_speed*self.GAME_SPEED
        
        if self.ball_speed[1] > 0.0 and new_ball[1] > 1.0:
            self.ball_speed[1] = -self.ball_speed[1]
            new_ball[1] = 1.0
        elif self.ball_speed[1] < 0.0 and new_ball[1] < -1.0:
            self.ball_speed[1] = -self.ball_speed[1]
            new_ball[1] = -1.0
        
            
        if new_ball[0] > self.GOAL_THRESHOLD:
            self.score[0] += 1
            new_ball = np.array([0.0,0.0])
            self.ball_speed = self.get_reset_speed()
        elif new_ball[0] < -self.GOAL_THRESHOLD:
            self.score[1] += 1
            new_ball = np.array([0.0,0.0])
            self.ball_speed = self.get_reset_speed()

        elif self.ball[0] <= self.right_paddle[0] and new_ball[0] > self.right_paddle[0]:
            a = self.right_paddle[1]-self.paddle_width
            b = new_ball[1]
            c = self.right_paddle[1]+self.paddle_width
            if a < b < c:
                new_angle = (b-a)/(c-a)*2.0-1.0
                new_ball[0] = 1.0
                self.ball_speed[0] *= -1.0
                self.ball_speed[1] = new_angle
                
        elif self.ball[0] >= self.left_paddle[0] and new_ball[0] < self.left_paddle[0]:
            a = self.left_paddle[1]-self.paddle_width
            b = new_ball[1]
            c = self.left_paddle[1]+self.paddle_width
            if a < b < c:
                new_angle = (b-a)/(c-a)*2.0-1.0
                new_ball[0] = -1.0
                self.ball_speed[0] *= -1.0
                self.ball_speed[1] = new_angle

        self.ball = new_ball
        
        
    def print_state(self):
        printsize = 15
        
        print('-'*printsize)
        
        grid = np.linspace(-self.GOAL_THRESHOLD,self.GOAL_THRESHOLD, printsize+1)
        
        lp = self.left_paddle
        rp = self.right_paddle
        pw = self.paddle_width
        
        for line in range(printsize):
            for col in range(printsize):
                if grid[col] <= self.ball[0] < grid[col+1] and grid[line] <= self.ball[1] < grid[line+1]:
                    print('O',end="")
                elif range_overlap(grid[col],grid[col+1],lp[0],lp[0]) and range_overlap(grid[line],grid[line+1],lp[1]-pw,lp[1]+pw):
                    print('|',end="")
                elif range_overlap(grid[col],grid[col+1],rp[0],rp[0]) and range_overlap(grid[line],grid[line+1],rp[1]-pw,rp[1]+pw):
                    print('|',end="")
                else:
                    print(' ',end="")
            print()
        print('-'*printsize)
        
        
pong = Pong()

for _ in range(3000000):
    pong.step(0.0,0.0)
    pong.print_state()
    time.sleep(0.037)
    #print(pong.ball, pong.score)