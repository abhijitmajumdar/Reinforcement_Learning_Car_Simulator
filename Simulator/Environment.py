import numpy as np
import math
import random

class Point():
    def __init__(self,pt):
        self.x = float(pt[0])
        self.y = float(pt[1])

    def __add__(self,point):
        return Point((self.x+point.x,self.y+point.y))

    def __sub__(self,point):
        return Point((self.x-point.x,self.y-point.y))

    def __mul__(self,value):
        return Point((value*self.x,value*self.y))

    def distance(self,point):
        return math.sqrt((self.x-point.x)**2 + (self.y-point.y)**2)

    def __str__(self):
        return 'P('+'{0:.2f}'.format(self.x)+','+'{0:.2f}'.format(self.y)+')'

class Vector():
    def __init__(self,p1,p2):
        self.point = p1
        self.end_point = p2
        self.vector = p2-p1

    def length(self):
        return math.sqrt(self.vector.x**2 + self.vector.y**2)

    def angle(self):
        if self.vector.x==0:
            return 0.0
        return math.atan2(self.vector.y,self.vector.x)

    def __add__(self,vec):
        return Vector(self.point,self.vector+vec.vector)

    def __sub__(self,vec):
        return Vector(self.point,self.vector-vec.vector)

    def __mul__(self,value):
        return Vector(self.point,self.point+(self.vector*value))

    def dot(self,vec):
        return (self.vector.x*vec.vector.x + self.vector.y*vec.vector.y)

    def cross(self,vec):
        return (self.vector.x*vec.vector.y - self.vector.y*vec.vector.x)

    def perpendicular(self):
        return Vector(self.point,self.point+Point((-self.vector.y,self.vector.x)))

    def unit_vector(self):
        return Vector(self.point,self.point+(self.vector*(1.0/self.length())))

    def in_segment(self,point):
        if self.vector.x != 0: # Not Vertical
            if ((self.point.x <= point.x) and (point.x <= self.point.x+self.vector.x)):
                return True
            if ((self.point.x >= point.x) and (point.x >= self.point.x+self.vector.x)):
                return True
        else:
            if ((self.point.y <= point.y) and (point.y <= self.point.y+self.vector.y)):
                return True
            if ((self.point.y >= point.y) and (point.y >= self.point.y+self.vector.y)):
                return True
        return False

    def intersection(self,vec):
        # http://geomalgorithms.com/a05-_intersect-1.html#intersect2D_2Segments()
        # https://stackoverflow.com/questions/3252194/numpy-and-line-intersections
        dp = Vector(vec.point,self.point)
        denom = self.cross(vec)
        if denom == 0: # Parallel segments
            return None
        X_point = (vec*(self.cross(dp)/denom)).vector + vec.point
        if (self.in_segment(X_point)==True and vec.in_segment(X_point)==True):
            return X_point
        else:
            return None

    def perpendicular_base_length(self,point):
        l_2 = self.vector.x**2 + self.vector.y**2
        l = math.sqrt(l_2)
        d = (l_2 + (self.point.x-point.x)**2 + (self.point.y-point.y)**2 - (self.end_point.x-point.x)**2 - (self.end_point.y-point.y)**2 )/(2*l)
        return d

    def draw_points(self):
        return (self.point.x,self.point.x+self.vector.x),(self.point.y,self.point.y+self.vector.y)

    def __str__(self):
        return 'V:'+str(self.vector.x)+'i'+ ('+' if self.vector.y>=0 else '-') +str(abs(self.vector.y))+'j'

class Car():
    # Bicycle model (Not to be mistaken for Ackerman model)
    # Current version: Robotics, Vision and Control by Peter Croke - 4.2 Car-like Mobile Robots
    # Another version: http://www.me.berkeley.edu/~frborrel/pdfpub/IV_KinematicMPC_jason.pdf
    def __init__(self,car_detail):
        self.detail = car_detail
        # Controls
        self.psi = 0 # Steering angle
        self.v = 0 # Heading velocity
        # States
        self.omega = self.detail['state'][2] # Heading angle
        self.x = self.detail['state'][0] # Position x
        self.y = self.detail['state'][1] # Position y
        self.physical_state = 'running' # or collided
        self.phi = np.array([0,0,0])
        # Sensing
        self.sensor_reading = None
        # Timing
        self.steps = 0
        # Scoring
        self.score = 0
        self.prev_score = 0
        self.total_reward = 0
        self.epoch = 0
        # Configuration
        self.destination = Point(self.detail['destination'])
        self.connection = car_detail['connection']-1 if 'connection' in car_detail else 0

    def constrain(self,quantity,c_min,c_max):
        return min(max(quantity,c_min),c_max)

    def wrap_angle(self,val):
        return( ( val + np.pi) % (2 * np.pi ) - np.pi )

    def update(self,dt):
        self.x += dt*self.v*math.cos(self.omega)
        self.y += dt*self.v*math.sin(self.omega)
        self.omega += (dt*self.v/self.detail['L'])*math.tan(self.psi)
        self.omega = self.wrap_angle(self.omega)

    def set_steering(self,angle):
        self.psi = self.constrain(angle,-self.detail['gamma_limit'],self.detail['gamma_limit'])

    def get_steering(self):
        return self.psi

    def set_velocity(self,velocity):
        self.v = self.constrain(velocity,-self.detail['v_limit'],self.detail['v_limit'])

    def get_velocity(self):
        return self.v

    def set_state(self,state):
        self.x,self.y,self.omega = state

    def get_state(self):
        return self.x,self.y,self.omega

    def set_sensor_reading(self,reading):
        self.sensor_reading = reading

    def get_sensor_reading(self):
        return self.sensor_reading

    def get_partial_state(self):
        return self.phi

    def random_state(self,mean,variance,angle_variance):
        self.set_state([mean[0]+random.uniform(-variance,variance), mean[1]+random.uniform(-variance,variance), mean[2]+random.uniform(-angle_variance,angle_variance)])

    def reset(self):
        self.psi = 0 # Steering angle
        self.v = 0 # Heading velocity
        self.omega = self.detail['state'][2] # Heading angle
        self.x = self.detail['state'][0] # Position x
        self.y = self.detail['state'][1] # Position y
        self.physical_state = 'running' # or collided
        self.steps = 0
        self.score = 0
        self.prev_score = 0
        self.phi = np.array([0,0,0])
        self.total_reward = 0

class Environment():
    def __init__(self,environment_details,env_select):
        env_select = env_select.split(',')
        if environment_details['path_creator']==True:
            self.env_generator(environment_details,env_select[0])
        self.arenas = []
        for env in env_select:
            pts = environment_details[env][:]
            pts.append(pts[0])
            segs = []
            for i in range(1,len(pts)):
                segs.append(Vector(Point(pts[i-1]),Point(pts[i])))
            points_np = np.array(pts)
            limits = (points_np[:,0].min(),points_np[:,0].max(),points_np[:,1].min(),points_np[:,1].max())
            max_delta = math.sqrt((limits[1]-limits[0])**2 + (limits[3]-limits[2])**2)
            self.arenas.append({'segments':segs,'limits':limits,'max_delta':max_delta})
        self.obs = []
        if environment_details['no_obstacles']==False:
            for obs in environment_details['Obstacle']:
                pts = environment_details['Obstacle'][obs][:]
                pts.append(pts[0])
                segs = []
                for i in range(1,len(pts)):
                    segs.append(Vector(Point(pts[i-1]),Point(pts[i])))
                self.obs.append(segs)
        self.max_steps = environment_details['max_steps']
        self.destination_radius = environment_details['dest_radius']
        self.buffer_space = environment_details['buffer_space']

    def constrain(self,quantity,c_min,c_max):
        return min(max(quantity,c_min),c_max)

    def scale(self,val,valmin,valmax,mi,ma):
        return ((val-valmin) * float(ma-mi)/(valmax-valmin))+mi

    def encode_angle(self,val):
        return [math.sin(val),math.cos(val)]

    def rotation_matrix(self,theta):
        ct = math.cos(theta)
        st = math.sin(theta)
        R = np.array([[ct,-st],[st,ct]])
        return R

    def find_range(self,pos,angle,length,segs):
        R = self.rotation_matrix(angle)
        points = np.array([ [0,0],[length,0] ]).T
        points = np.dot(R,points)
        points[0,:] += pos.x
        points[1,:] += pos.y
        sensor = Vector(Point((points[0][0],points[1][0])),Point((points[0][1],points[1][1])))
        intersections = []
        for seg in segs:
            X = sensor.intersection(seg)
            if X is not None:
                intersections.append(X)
        for obs in self.obs:
            for seg in obs:
                X = sensor.intersection(seg)
                if X is not None:
                    intersections.append(X)
        dist = np.array([pos.distance(p) for p in intersections])
        return pos.distance(intersections[np.argmin(dist)]) if len(dist)>0 else length

    def point_orientation_wrt_segment(self,point,segment):
        # From https://www.cs.cmu.edu/~quake/robust.html
        return True if ((segment[0][0]-point[0])*(segment[1][1]-point[1]) - (segment[1][0]-point[0])*(segment[0][1]-point[1]))>0 else False

    def point_projection_on_segment(self,point,segment):
        # https://stackoverflow.com/questions/17581738/check-if-a-point-projected-on-a-line-segment-is-not-outside-it
        dx = segment[1][0] - segment[0][0]
        dy = segment[1][1] - segment[0][1]
        innerProduct = (point[0] - segment[0][0])*dx + (point[1] - segment[0][1])*dy
        return ((0<=innerProduct) and (innerProduct<=(dx*dx+dy*dy)))

    def check_point_inside_polygon(self,point,polygon):
        # From http://www.geeksforgeeks.org/how-to-check-if-a-given-point-lies-inside-a-polygon/
        pos_segment = Vector(point, Point((1000,point.y)))
        intercepts = [pos_segment.intersection(seg) for seg in polygon]
        inside = len([i for i in intercepts if i is None])%2 != 0
        return inside

    def compute_interaction(self,*agents):
        for agent in agents:
            x,y,omega = agent.get_state()
            car_pos = Point((x,y))
            segs = self.arenas[agent.connection]['segments']
            # Sensor update
            sensor_values = []
            for s in agent.detail['sensors']:
                sensor_values.append(self.find_range(car_pos,omega+s['angle'],s['range'],segs))
            agent.set_sensor_reading(sensor_values)
            # Calculate euclidian distance from destination
            agent.prev_score = agent.score
            agent.score = car_pos.distance(agent.destination)
            # Collision update
            inside_env = self.check_point_inside_polygon(car_pos,segs)
            outside_obs = all([not self.check_point_inside_polygon(car_pos,obs) for obs in self.obs])
            # Delta state
            delta = Vector(car_pos,agent.destination)
            s,c = self.encode_angle(delta.angle()-omega)
            dist = self.scale(delta.length(),0,self.arenas[agent.connection]['max_delta'],0,1)
            #agent.phi = np.concatenate(([dist,s,c],self.scale(np.array(sensor_values),0,agent.detail['sensors']['S1']['range'],1,0)))
            agent.phi = np.concatenate(([dist,s,c],self.scale(np.array(sensor_values),0,2,1,0)))
            # Update agent condition
            if (not inside_env) or (not outside_obs): agent.physical_state = 'collided'
            # Timing
            elif agent.steps > self.max_steps: agent.physical_state = 'timeup'
            # Check destination reached
            elif agent.score<self.destination_radius: agent.physical_state = 'destination'
            if agent.physical_state=='running': agent.steps += 1

    def set_max_steps(self,value):
        self.max_steps = value

    def check_valid_point(self,pt,agent):
        if self.check_point_inside_polygon(pt,self.arenas[agent.connection]['segments'])==True:
            if all([not self.check_point_inside_polygon(pt,obs) for obs in self.obs])==True:
                return True
        return False

    def change_destination(self,agent,x,y):
        p = Point((self.constrain(x,self.arenas[agent.connection]['limits'][0]+self.buffer_space,self.arenas[agent.connection]['limits'][1]-self.buffer_space),self.constrain(y,self.arenas[agent.connection]['limits'][2]+self.buffer_space,self.arenas[agent.connection]['limits'][3]-self.buffer_space)))
        if self.check_valid_point(p,agent)==True: agent.destination.x, agent.destination.y = p.x, p.y

    def random_valid_position(self,agent):
        p = Point((0,0))
        while(True):
            p.x,p.y = self.constrain(
                        random.uniform(self.arenas[agent.connection]['limits'][0],self.arenas[agent.connection]['limits'][1]),
                        self.arenas[agent.connection]['limits'][0]+self.buffer_space,
                        self.arenas[agent.connection]['limits'][1]-self.buffer_space
                        ), self.constrain(
                        random.uniform(self.arenas[agent.connection]['limits'][2],self.arenas[agent.connection]['limits'][3]),
                        self.arenas[agent.connection]['limits'][2]+self.buffer_space,
                        self.arenas[agent.connection]['limits'][3]-self.buffer_space
                        )
            if self.check_valid_point(p,agent)==True: break
        return p

    def randomize(self,agent_position,destination_position,*agents):
        for agent in agents:
            if agent_position==True:
                rp = self.random_valid_position(agent)
                agent.x,agent.y = rp.x,rp.y
                agent.omega = random.uniform(-np.pi,np.pi)
            if destination_position==True:
                rp = self.random_valid_position(agent)
                agent.destination.x,agent.destination.y = rp.x,rp.y

    def env_generator(self,env_dict,select):
        path = env_dict[select][:]
        d = env_dict['track_width']
        start_angle = 0.0
        side_L,side_R = [],[]
        for i in range(len(path)):
            p1,p2,p3 = None,Point(path[i]),None
            if i==0:
                p3 = Point(path[i+1])
                u = Vector(p3,p2).unit_vector()
                p1 = u.vector*d + p2
                start_angle = Vector(p2,p3).angle()
            elif i==(len(path)-1):
                p1 = Point(path[i-1])
                u = Vector(p1,p2).unit_vector()
                p3 = u.vector*d + p2
            else:
                p1 = Point(path[i-1])
                p3 = Point(path[i+1])
            perp_unit_vec = Vector(p1,p3).perpendicular().unit_vector()
            pl,pr = p2+perp_unit_vec.vector*d, p2-perp_unit_vec.vector*d
            side_L.append((pl.x,pl.y))
            side_R.append((pr.x,pr.y))
        env_dict[select] = side_R + side_L[::-1]
