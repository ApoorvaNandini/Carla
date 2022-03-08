#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import glob
import os
import sys
import csv
import torch
from torch.autograd import Variable
from cebgan_model import netG, netD, args


try:
    sys.path.append(glob.glob('../carla-simulator/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
from agents.navigation.basic_agent import BasicAgent
from agents.navigation.behavior_agent import BehaviorAgent

import random
import PIL
from PIL import Image
import scipy.misc as misc
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.backends.backend_agg as agg
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_q
    from pygame.locals import K_LEFT
    from pygame.locals import K_RIGHT
    from pygame.locals import K_m
    from pygame.locals import K_a
    from pygame.locals import K_l
    from pygame.locals import K_r
    from pygame.locals import K_s
    from pygame.locals import K_j
    from pygame.locals import K_e
    from pygame.locals import K_g
    from pygame.locals import K_c
    from pygame.locals import K_w
    from pygame.locals import K_t
    from pygame.locals import K_u
    from pygame.locals import K_p
    from pygame.locals import K_SPACE
    from pygame.locals import K_UP
    from pygame.locals import K_DOWN
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

try:
    import queue
except ImportError:
    import Queue as queue

class CarlaSyncMode(object):
    """
    Context manager to synchronize output from different sensors. Synchronous
    mode is enabled as long as we are inside this context

        with CarlaSyncMode(world, sensors) as sync_mode:
            while True:
                data = sync_mode.tick(timeout=1.0)

    """

    def __init__(self, world, *sensors, **kwargs):
        self.world = world
        self.sensors = sensors
        self.frame = None
        self.delta_seconds = 1.0 / kwargs.get('fps', 20)
        self._queues = []
        self._settings = None

    def __enter__(self):
        self._settings = self.world.get_settings()
        self.frame = self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode=False,
            synchronous_mode=True,
            fixed_delta_seconds=self.delta_seconds))

        def make_queue(register_event):
            q = queue.Queue()
            register_event(q.put)
            self._queues.append(q)

        make_queue(self.world.on_tick)
        for sensor in self.sensors:
            make_queue(sensor.listen)
        return self

    def tick(self, timeout):
        self.frame = self.world.tick()
        data = [self._retrieve_data(q, timeout) for q in self._queues]
        assert all(x.frame == self.frame for x in data)
        return data

    def __exit__(self, *args, **kwargs):
        self.world.apply_settings(self._settings)

    def _retrieve_data(self, sensor_queue, timeout):
        while True:
            data = sensor_queue.get(timeout=timeout)
            if data.frame == self.frame:
                return data


# ==============================================================================
# -- KeyboardControl -----------------------------------------------------------
# ==============================================================================


class KeyboardControl(object):
    def __init__(self, world, vehicle, autopilot_enabled=True):
        self.vehicle = vehicle
        self.autopilot_enabled = autopilot_enabled
        self.control = carla.VehicleControl()
        self.steer_cache = 0.0
        self.inbuilt_autopilot_enabled = 0
        self.set_orientation = False
        self.get_waypoint = False

    def parse_events(self, clock):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            if event.type == pygame.KEYDOWN:
                if event.key == K_m:
                    self.autopilot_enabled = False
                    self.vehicle.set_autopilot(self.autopilot_enabled)
                    print('Autopilot Off')
                if event.key == K_a:
                    self.autopilot_enabled = True
                    self.vehicle.set_autopilot(self.autopilot_enabled)
                    print('Autopilot On')               
                if event.key == K_j:
                    print('Inbuilt autopilot enabled')
                    self.inbuilt_autopilot_enabled = 1
                if event.key == K_s:
                    print('Inbuilt autopilot disabled')
                    self.inbuilt_autopilot_enabled = 0
                if event.key == K_c:
                    print("set car orientation")
                    self.set_orientation = True
                if event.key == K_g:
                    print("getting waypoint")
                    self.get_waypoint = True
            if event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    return True
        
        if not self.autopilot_enabled:
            self._parse_vehicle_keys(pygame.key.get_pressed(), clock.get_time())
            self.control.reverse = self.control.gear < 0           
            self.vehicle.apply_control(self.control)

    
    def _parse_vehicle_keys(self, keys, milliseconds):
        self.control.throttle = 1.0 if keys[K_UP] else 0.0
        steer_increment = 5e-4 * milliseconds
        if keys[K_LEFT]:
            if self.steer_cache > 0:
                self.steer_cache = 0
            else:
                self.steer_cache -= steer_increment
        elif keys[K_RIGHT]:
            if self.steer_cache < 0:
                self.steer_cache = 0
            else:
                self.steer_cache += steer_increment
        else:
            self.steer_cache = 0.0
        self.steer_cache = min(0.7, max(-0.7, self.steer_cache))
        self.control.steer = round(self.steer_cache, 1)
        self.control.brake = 1.0 if keys[K_DOWN] else 0.0
        self.control.hand_brake = keys[K_SPACE]

    
    @staticmethod
    def _is_quit_shortcut(key):
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)

# ==============================================================================
# ------------------------------------------------------------------------------
# ==============================================================================

def draw_image(surface, image, blend=False):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    if blend:
        image_surface.set_alpha(100)
    surface.blit(image_surface, (0, 0))
    return image.raw_data, array

def draw_image2(surface, image, blend=False):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    #image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    #if blend:
    #    image_surface.set_alpha(100)
    #surface.blit(image_surface, (0, 0))
    return image.raw_data, array


def get_font():
    fonts = [x for x in pygame.font.get_fonts()]
    default_font = 'ubuntumono'
    font = default_font if default_font in fonts else fonts[0]
    font = pygame.font.match_font(font)
    return pygame.font.Font(font, 14)

def should_quit():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_ESCAPE:
                return True
    return False


def main():
    keyboard_controller = True
    actor_list = []
    pygame.init()
    display = pygame.display.set_mode(
        (800+640, 600),
        pygame.HWSURFACE | pygame.DOUBLEBUF)
    font = get_font()
    clock = pygame.time.Clock()

    client = carla.Client('localhost', 2000)
    client.set_timeout(2.0)
    world = client.load_world('Town01') # get_world() #
    world.set_weather(carla.WeatherParameters.ClearSunset)
    tot_target_reached = 0
    num_min_waypoints = 21
    
    output = None
    failures = 0
    #total_lane_changes = 0

    random_state = np.random.RandomState(seed=1)
    X = np.concatenate([random_state.normal(-1, 0.2, 500),
                        random_state.normal(1, 0.2, 250)]).reshape(-1, 1)

    gm = GaussianMixture(n_components=3, random_state=0).fit(X)

    try:
        m = world.get_map()
        spawn_points = m.get_spawn_points()
        print(len(spawn_points))
        # hair pin curve - town03 - 88, 225
        # 90 degree turn - town01 - 72, 52
        # T shape junction - town01 - 241, 237

     
        # works - town01 - (241, 237)
        start_pose =  spawn_points[241] #297 261
        end_location = spawn_points[237].location #333 310
        blueprint_library = world.get_blueprint_library()
        
        vehicle = world.spawn_actor(
            random.choice(blueprint_library.filter('vehicle.audi.a2')),
            start_pose)
        actor_list.append(vehicle)
        world.player = vehicle
        vehicle.set_simulate_physics(True)

        #agent = BasicAgent(vehicle)#, ignore_traffic_light=True, behavior='cautious')
        #agent.set_destination([end_location.x, end_location.y, end_location.z])
        agent = BehaviorAgent(vehicle, ignore_traffic_light=True, behavior='cautious')
        agent.set_destination(agent.vehicle.get_location(), end_location, clean=True)

        camera_rgb = world.spawn_actor(
            blueprint_library.find('sensor.camera.rgb'),
            carla.Transform(carla.Location(x=2.5, z=1.5)),
            attach_to=vehicle)
        actor_list.append(camera_rgb)

        camera_top_view = world.spawn_actor(
            blueprint_library.find('sensor.camera.rgb'),
            carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)),
            attach_to=vehicle)
        actor_list.append(camera_top_view)


        if keyboard_controller:
            controller = KeyboardControl(world, vehicle, True)
        

        np_scale = 200
        lin_points = torch.linspace(-1, 1, np_scale).view(np_scale, 1)
        lin_points_V = Variable(lin_points)
        dup_images = torch.FloatTensor(np_scale, args.imgCh, args.imgH, args.imgW)
        anomaly_indicators = []
        steering_deviations = []
        mul_factor = 1.0
        

        # Create a synchronous mode context.
        with CarlaSyncMode(world, camera_rgb, camera_top_view, fps=30) as sync_mode:
            while True:
                if keyboard_controller:
                    if controller.parse_events(clock):
                        return
                else:
                    if should_quit():
                        return

                agent.update_information(world)
                clock.tick()

                if vehicle.is_at_traffic_light():
                    traffic_light = vehicle.get_traffic_light()
                    if traffic_light.get_state() == carla.TrafficLightState.Red:
                        traffic_light.set_state(carla.TrafficLightState.Green)
                        traffic_light.set_green_time(10.0)

                
                if len(agent.get_local_planner().waypoints_queue) < num_min_waypoints:
                    agent.reroute(spawn_points)
                    tot_target_reached += 1
                    print("ReRouting")
                

                if controller.get_waypoint:
                    location = vehicle.get_location()
                    ego_vehicle_wp = m.get_waypoint(location)
                    next_location = list(ego_vehicle_wp.next(5))[0].transform.location
                    vehicle.set_location(next_location)
                    controller.get_waypoint = False

                # Advance the simulation and wait for the data.
                snapshot, image_rgb, image_topview = sync_mode.tick(timeout=2.0)
   
                # Draw the display.
                raw, img = draw_image2(display, image_rgb)
                raw2, img2 = draw_image(display, image_topview)

                img1 = Image.fromarray(img)
                img1 = img1.resize((128, 128), PIL.Image.ANTIALIAS)
                img1 = np.array(img1)
                img1 = img1.transpose(2, 0, 1)
                img1 = np.reshape(img1/255.0, (1, 3,128, 128))
                img1 = torch.from_numpy(img1).type(torch.FloatTensor)#.cuda()

                control = agent.run_step()
                
                steering_control = control.steer*mul_factor
                steering_control = torch.tensor(steering_control).type(torch.FloatTensor).cuda()
                for i in range(np_scale):
                    dup_images.select(0, i).copy_(img1.select(0, 0))

                dup_images_V = Variable(dup_images)
                
                latent_input = torch.FloatTensor(200, args.zSize)
                samples =  gm.sample(200)[0]
                samples_t = torch.tensor(samples)
                latent_input.copy_(samples_t)
                #latent_input.uniform_(-1.0, 1.0)
                latent_input = latent_input.cuda()
                
                latent_input_V = Variable(latent_input)
                img1 = img1.cuda()
                img1_V = Variable(img1)
                generator_output = netG(latent_input_V, dup_images_V.cuda())
                outputs = netD(lin_points_V.cuda(), dup_images_V.cuda())
                argmin = torch.argmin(outputs).item()
                #energy = outputs[argmin].item()
                min_energy_str_command = lin_points[argmin][0].item()
                if abs(control.steer * mul_factor - min_energy_str_command) < 0.3:
                    anomaly_indicators.append(0)
                else:
                    anomaly_indicators.append(1)
                steering_deviations.append(control.steer * mul_factor - min_energy_str_command)
                
                figure = plt.figure(figsize=(6.4, 6))
                plot = figure.add_subplot(211)
                plot.plot(lin_points.type(torch.FloatTensor).numpy(), outputs.data.type(torch.FloatTensor).numpy())
                plot.plot(steering_control.cpu().numpy(), torch.FloatTensor(1).fill_(0).numpy(), 'b^-')
                plot.set_xlim([-1.0, 1.0])
                plot.set_ylim([0.0, 1.5])
                plot.set_xlabel('steering input')
                plot.set_ylabel('Energy')


                plot = figure.add_subplot(212)
                plot.plot(generator_output.select(1,0).cpu().detach().numpy(), color='red') #latent_input.select(1,0).cpu().detach().numpy(),
                #plot.plot(, color= 'blue')
                plot.set_ylim([-1.0, 1.0])
                plot.set_xlabel('latent input')
                plot.set_ylabel('steering command')
                plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
                canvas = agg.FigureCanvasAgg(figure)
                canvas.draw()
                renderer = canvas.get_renderer()
                raw_data = renderer.tostring_rgb()
                size = canvas.get_width_height()
                #print(size)
                surf = pygame.image.fromstring(raw_data, size, "RGB")
                display.blit(surf, (800,0))
                pygame.display.flip()
                #plt.show()
                
                #control.steer = generator_output[0][-1].item()/mul_factor
                vehicle.apply_control(control)
                
                fps = round(1.0 / snapshot.timestamp.delta_seconds)
 
                display.blit(
                    font.render('% 5d FPS (real)' % clock.get_fps(), True, (255, 255, 255)),
                    (8, 10))
                display.blit(
                    font.render('% 5d FPS (simulated)' % fps, True, (255, 255, 255)),
                    (8, 28))
                display.blit(
                    font.render('% 5f (steering)' % control.steer, True, (255, 255, 255)),
                    (8, 48))
                display.blit(
                    font.render('% 5f (gen. steering)' % generator_output[0][-1].item(), True, (255, 255, 255)),
                    (8, 68))
                pygame.display.flip()
                plt.close()

    finally:
        print('destroying actors.')
        
        for actor in actor_list:
            actor.destroy()

        pygame.quit()
        print('done.')


if __name__ == '__main__':

    try:

        main()

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
