# coding=utf-8
# Copyright 2018 The Dopamine Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Module defining classes and helper methods for general agents."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
import shutil


import sys
import time
import datetime
import csv, json, pickle

import random
from dopamine.agents.dqn import dqn_agent
from dopamine.agents.implicit_quantile import implicit_quantile_agent
from dopamine.agents.rainbow import rainbow_agent
from dopamine.agents.fqf import fqf_agent
from dopamine.discrete_domains import atari_lib
from dopamine.discrete_domains import checkpointer
from dopamine.discrete_domains import iteration_statistics
from dopamine.discrete_domains import logger
from dopamine.discrete_domains.tictactoe.examples.td_agent import TDAgent
import numpy as np
import tensorflow as tf

import gin.tf

from copy import deepcopy

def load_gin_configs(gin_files, gin_bindings):
  """Loads gin configuration files.
  Args:
    gin_files: list, of paths to the gin configuration files for this
      experiment.
    gin_bindings: list, of gin parameter bindings to override the values in
      the config files.
  """
  gin.parse_config_files_and_bindings(gin_files,
                                      bindings=gin_bindings,
                                      skip_unknown=False)


@gin.configurable
def create_agent(sess, environment, agent_name=None, summary_writer=None,
                 debug_mode=False):
  """Creates an agent.
  Args:
    sess: A `tf.Session` object for running associated ops.
    environment: A gym environment (e.g. Atari 2600).
    agent_name: str, name of the agent to create.
    summary_writer: A Tensorflow summary writer to pass to the agent
      for in-agent training statistics in Tensorboard.
    debug_mode: bool, whether to output Tensorboard summaries. If set to true,
      the agent will output in-episode statistics to Tensorboard. Disabled by
      default as this results in slower training.
  Returns:
    agent: An RL agent.
  Raises:
    ValueError: If `agent_name` is not in supported list.
  """

  #return dqn_agent.DQNAgent(graph=tf.Graph(),  num_actions=environment.action_space.n, environment=environment,
  #                            summary_writer=summary_writer)

  assert agent_name is not None
  if not debug_mode:
    summary_writer = None
  if agent_name == 'dqn':
    return dqn_agent.DQNAgent(num_actions=environment.action_space.n,
                              summary_writer=summary_writer)
  elif agent_name == 'rainbow':
    return rainbow_agent.RainbowAgent(
         num_actions=environment.action_space.n,
        summary_writer=summary_writer)
  elif agent_name == 'implicit_quantile':
    return implicit_quantile_agent.ImplicitQuantileAgent(
         num_actions=environment.action_space.n, environment=environment, #CHANGE
        summary_writer=summary_writer)
  elif agent_name == 'fqf':
    return fqf_agent.FQFAgent(
        sess, num_actions=environment.action_space.n,
        summary_writer=summary_writer)
  else:
    raise ValueError('Unknown agent: {}'.format(agent_name))


@gin.configurable
def create_runner(base_dir, schedule='continuous_train_and_eval'):
  """Creates an experiment Runner.
  Args:
    base_dir: str, base directory for hosting all subdirectories.
    schedule: string, which type of Runner to use.
  Returns:
    runner: A `Runner` like object.
  Raises:
    ValueError: When an unknown schedule is encountered.
  """
  assert base_dir is not None
  # Continuously runs training and evaluation until max num_iterations is hit.
  if schedule == 'continuous_train_and_eval':
    return Runner(base_dir, create_agent)
  # Continuously runs training until max num_iterations is hit.
  elif schedule == 'continuous_train':
    return TrainRunner(base_dir, create_agent)
  else:
    raise ValueError('Unknown schedule: {}'.format(schedule))


@gin.configurable
class Runner(object):
  """Object that handles running Dopamine experiments.
  Here we use the term 'experiment' to mean simulating interactions between the
  agent and the environment and reporting some statistics pertaining to these
  interactions.
  A simple scenario to train a DQN agent is as follows:
  ```python
  import dopamine.discrete_domains.atari_lib
  base_dir = '/tmp/simple_example'
  def create_agent(sess, environment):
    return dqn_agent.DQNAgent(sess, num_actions=environment.action_space.n)
  runner = Runner(base_dir, create_agent, atari_lib.create_atari_environment)
  runner.run()
  ```
  """

  def __init__(self,
               base_dir,
               create_agent_fn,
               create_environment_fn=atari_lib.create_atari_environment,
               checkpoint_file_prefix='ckpt',
               logging_file_prefix='log',
               log_every_n=1,
               num_iterations=200,
               runtype='run',
               game='Pong',
               training_steps=250000,
               evaluation_steps=125000,
               max_steps_per_episode=27000):
    """Initialize the Runner object in charge of running a full experiment.
    Args:
      base_dir: str, the base directory to host all required sub-directories.
      create_agent_fn: A function that takes as args a Tensorflow session and an
        environment, and returns an agent.
      create_environment_fn: A function which receives a problem name and
        creates a Gym environment for that problem (e.g. an Atari 2600 game).
      checkpoint_file_prefix: str, the prefix to use for checkpoint files.
      logging_file_prefix: str, prefix to use for the log files.
      log_every_n: int, the frequency for writing logs.
      num_iterations: int, the iteration number threshold (must be greater than
        start_iteration).
      training_steps: int, the number of training steps to perform.
      evaluation_steps: int, the number of evaluation steps to perform.
      max_steps_per_episode: int, maximum number of steps after which an episode
        terminates.
    This constructor will take the following actions:
    - Initialize an environment.
    - Initialize a `tf.Session`.
    - Initialize a logger.
    - Initialize an agent.
    - Reload from the latest checkpoint, if available, and initialize the
      Checkpointer object.
    """
    assert base_dir is not None
    self._logging_file_prefix = logging_file_prefix
    self._log_every_n = log_every_n
    self._num_iterations = num_iterations
    self._runtype = runtype
    self._game = game
    self._training_steps = training_steps
    self._evaluation_steps = evaluation_steps
    self._max_steps_per_episode = max_steps_per_episode
    self._base_dir = base_dir
    self._create_directories()
    #self._summary_writer = tf.contrib.summary.FileWriter(self._base_dir) CHANGE
    self.testing = False
    self.opponent=TDAgent('X',0,0)
    self.PATH='/home/neriait/tic_tac_toe_IQN_2/dopamine/discrete_domains/'
    self.opponent.load_model(self.PATH + 'tictactoe/examples/mirai')
    if 'sticky' in runtype:
        self.sticky_actions = True
    else:
        self.sticky_actions = False
    self._environment = create_environment_fn(sticky_actions=self.sticky_actions)
    # Set up a session and initialize variables.
    
    self._agent = create_agent_fn(None, self._environment,
                                  summary_writer=None)
    
    
    
    
    self._agent2 = create_agent_fn(None, self._environment,
                                  summary_writer=None)
    
    
    self._agent11 = create_agent_fn(None, self._environment,
                                  summary_writer=None)
    
    
    
    
    self._agent22 = create_agent_fn(None, self._environment,
                                  summary_writer=None)
    
    self._agent.testing = self.testing
    self._agent2.testing = self.testing
    
    self._agent11.testing = self.testing
    self._agent22.testing = self.testing
    
    self.create_agent_fn=create_agent_fn
    
    
    #self._summary_writer.add_graph(graph=tf.get_default_graph()) CHANGE
    self._initialize_checkpointer_and_maybe_resume(checkpoint_file_prefix)

    self.step_count_total = 0
    #game = self._environment.name
    print (">>>>>>>>>", self._game)
    self.date = date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    self.job_name = self._game + '-' + self._runtype + '-' + date
    self.filename = './fout-results/results-%s_%s-%s.txt' % (self._game, self._runtype, date)
    self.filename_test = './fout-results/testresults-%s_%s-%s.txt' % (self._game, self._runtype, date)
    self.filename_test_raw = 'testresults-%s_%s-%s.txt' % (self._game, self._runtype, date)
    if not self.testing:
        self.fout = open(self.filename, 'w+')
        self.fout_test = open(self.filename_test, 'w+')

  def _create_directories(self):
    """Create necessary sub-directories."""
    self._checkpoint_dir = os.path.join(self._base_dir, 'checkpoints')
    print("CHECKPOINTDIR:", self._checkpoint_dir)
    self._logger = logger.Logger(os.path.join(self._base_dir, 'logs'))

  def _initialize_checkpointer_and_maybe_resume(self, checkpoint_file_prefix):
    """Reloads the latest checkpoint if it exists.
    This method will first create a `Checkpointer` object and then call
    `checkpointer.get_latest_checkpoint_number` to determine if there is a valid
    checkpoiRnt in self._checkpoint_dir, and what the largest file number is.
    If a valid checkpoint file is found, it will load the bundled data from this
    file and will pass it to the agent for it to reload its data.
    If the agent is able to successfully unbundle, this method will verify that
    the unbundled data contains the keys,'logs' and 'current_iteration'. It will
    then load the `Logger`'s data from the bundle, and will return the iteration
    number keyed by 'current_iteration' as one of the return values (along with
    the `Checkpointer` object).
    Args:
      checkpoint_file_prefix: str, the checkpoint file prefix.
    Returns:
      start_iteration: int, the iteration number to start the experiment from.
      experiment_checkpointer: `Checkpointer` object for the experiment.
    """
    self._checkpointer = checkpointer.Checkpointer(self.PATH + "checkpoints",
                                                   checkpoint_file_prefix)
    
    
    self._checkpointer2 = checkpointer.Checkpointer(self.PATH + "player2", checkpoint_file_prefix)
    self._checkpointerlatest1 = checkpointer.Checkpointer(self.PATH + "latest3", checkpoint_file_prefix)
    self._checkpointerlatest2 = checkpointer.Checkpointer(self.PATH + "latest4", checkpoint_file_prefix)
    self._checkpointertest = checkpointer.Checkpointer(self.PATH + "test", checkpoint_file_prefix)
    
    self._start_iteration = 0
    # Check if checkpoint exists. Note that the existence of checkpoint 0 means
    # that we have finished iteration 0 (so we will start from iteration 1).
    latest_checkpoint_version = checkpointer.get_latest_checkpoint_number(
        self.PATH + "checkpoints")
    print("Latest version:",latest_checkpoint_version)
    return 5
    if latest_checkpoint_version >= 0:
      experiment_data = self._checkpointer.load_checkpoint(
          latest_checkpoint_version)
      print("Read to apple")
      if self._agent.unbundle(
          self.PATH + "checkpoints", latest_checkpoint_version, experiment_data):
        
        assert 'logs' in experiment_data
        assert 'current_iteration' in experiment_data
        self._logger.data = experiment_data['logs']
        self._start_iteration = experiment_data['current_iteration'] + 1
        tf.logging.info('Reloaded checkpoint and will start from iteration %d',
                        self._start_iteration)
        #self.testing = True
        #self._agent.testing = True
        print ('TESTING:', self.testing)
        print("BANANA")
  
  
  
  def _my_initialize_resume(self,  checkpoint_file_prefix, folder_type, iteration, player):
    
    self._start_iteration = 0
    # Check if checkpoint exists. Note that the existence of checkpoint 0 means
    # that we have finished iteration 0 (so we will start from iteration 1).
    if player==11 or player==1:
      if folder_type=='latest':
        self._delete_everything_in_folder(self.PATH + ""+ folder_type +str(1+2))
        self._move_everything_in_folder(self.PATH + ""+ folder_type +str(1), self.PATH + ""+ folder_type +str(1+2))
    
      latest_checkpoint_version = iteration
      if True:
        if folder_type=='latest':
          experiment_data = self._checkpointerlatest1.load_checkpoint(iteration)
        else:
          experiment_data = self._checkpointer1.load_checkpoint(iteration)
        x=False
        if player==11:
          x=self._agent11.unbundle(self.PATH + ""+folder_type+str(1+2), iteration, experiment_data)
        else:  
          x=self._agent.unbundle(self.PATH + ""+folder_type+str(1+2), iteration, experiment_data)
        if x:
          assert 'logs' in experiment_data
          assert 'current_iteration' in experiment_data
          self._logger.data = experiment_data['logs']
          self._start_iteration = experiment_data['current_iteration'] + 1
          #self.testing = True
          #self._agent.testing = True
    else:
      if folder_type=='latest':
        self._delete_everything_in_folder(self.PATH + ""+ folder_type +str(2+2))
        self._move_everything_in_folder(self.PATH + ""+ folder_type +str(2), self.PATH + ""+ folder_type +str(2+2))
    
      
      latest_checkpoint_version = iteration
      if True:
        if folder_type=='latest':
          experiment_data = self._checkpointerlatest2.load_checkpoint(iteration)
        else:
          moop=poopo
        if player==22:
          x=self._agent22.unbundle(self.PATH + ""+folder_type+str(2+2), iteration, experiment_data)
        else:
          x=self._agent2.unbundle(self.PATH + ""+folder_type+str(2+2), iteration, experiment_data)
        if x:
          assert 'logs' in experiment_data
          assert 'current_iteration' in experiment_data
          self._logger.data = experiment_data['logs']
          self._start_iteration = experiment_data['current_iteration'] + 1
          #self.testing = True
          #self._agent.testing = True

  
  def _initialize_episode(self, run_mode_str):
    """Initialization for a new episode.
    Returns:
      action: int, the initial action chosen by the agent.
    """
    print("Start episode initaliaztion")
    initial_observation = self._environment.reset()
    initial_observation=initial_observation[0]
    
    print("Finish episode initaliaztion")
    return self._agent.get_action_from_observation(initial_observation)

  def _initialize_former_agents(self):
    
    
    
    self._my_initialize_resume('crpt', 1)
    self._my_initialize_resume('crpt', 2)
    

  def _run_one_step(self, action):
    """Executes a single step in the environment.
    Args:
      action: int, the action to perform in the environment.
    
    Returns:
      The observation, reward, and is_terminal values returned from the
        environment.
    """
    observation, reward, is_terminal, _ = self._environment.step(action)
    '''
    if self.testing:
        image = self._environment.render('rgb_array')
        self._agent.vis['s'].append(image)
        self._agent.vis['state_input'].append(image)
        #print (image.shape)
        self._agent.vis['r'].append(reward)
        quantiles, values = self._sess.run([
            self._agent._net_outputs.quantiles, 
            self._agent._net_outputs.quantile_values], 
            {self._agent.state_ph: self._agent.state}
        )
        self._agent.vis['fraction'].append(quantiles)
        self._agent.vis['values'].append(values)
        #print (quantiles.shape)
    '''
    return observation, reward, is_terminal

  def _end_episode(self, reward):
    """Finalizes an episode run.
    Args:
      reward: float, the last reward from the environment.
    """
    self._agent.end_episode(reward)


  def observation_to_tensorlike(self, observation):
    x=0
    if observation[1]=='O':
      x=1
    else:
      x=-1
    return observation[0]+(50*x,)
    
  def _run_one_episode(self, run_mode_str):
    """Executes a full trajectory of the agent interacting with the environment.
    Returns:
      The number of steps taken and the total reward.
    """
    step_number = 0
    total_reward = 0.
    
    
    
    action=None
    switchedobservation2=None
    gameplayer=[]
    gameopponent=[]
    pi_data=[]
    count=0
    
      
    initial_observation = self._environment.reset()
    
    is_terminal = False
    observation2=initial_observation
    
    first_time=True
    while True:
    #Player1
      originalobservation2=observation2
      observation2=self.observation_to_tensorlike(observation2)
      if run_mode_str=='oldvsnew':
        pass
      #    observation, reward, is_terminal = self._run_one_step(self._agent11.get_action_from_observation(observation2))
      #    action=2
      #elif run_mode_str=='oldvsboss':
      #    observation, reward, is_terminal = self._run_one_step(self._agent11.get_action_from_observation(observation2))
      #    action=2
      #elif run_mode_str=='oldvsold':
      #    observation, reward, is_terminal = self._run_one_step(self._agent11.get_action_from_observation(observation2))
      #    action=2
      #elif run_mode_str=='newvsold':
      #    observation, reward, is_terminal = self._run_one_step(self._agent.get_action_from_observation(observation2))
      #    action=2
      
      elif run_mode_str=='evalrandom':
          observation, reward, is_terminal = self._run_one_step(self._agent.get_action_from_observation(observation2))
          action=2
      elif run_mode_str=='eval':
        action=self._agent.get_action_from_observation(observation2)
        observation, reward, is_terminal = self._run_one_step(action)
      elif run_mode_str=='train1':
        self._agent.eval_mode=False
        
        if count < CFG.temp_thresh:
                best_child = self.mcts1.search(game, node, CFG.temp_init)
        else:
                best_child = self.mcts1.search(game, node, CFG.temp_final)
        
        action = best_child.action
        observation, reward, is_terminal = self._run_one_step(action) 
      elif run_mode_str=='train2':
        self._agent.eval_mode=True
        if count < CFG.temp_thresh:
                best_child = self.mcts.search(game, node, CFG.temp_init)
        else:
                best_child = self.mcts.search(game, node, CFG.temp_final)
        action = best_child.action
        observation, reward, is_terminal = self._run_one_step(action) 
      else:
        print("Marco polo")
        action = self._agent.get_action_from_observation(observation2)
        observation, reward, is_terminal = self._run_one_step(action) 
      
      observationbeforefirstplayeraction=observation2
      originalobservation=observation
      observation=self.observation_to_tensorlike(observation)
      if not is_terminal:
      #Player2
        if run_mode_str=='oldvsnew':
          pass
        #  observation2, reward2, is_terminal = self._run_one_step(self._agent2.get_action_from_observation(observation))
        #  action2=2
        #elif run_mode_str=='oldvsold':
        #  observation2, reward2, is_terminal = self._run_one_step(self._agent22.get_action_from_observation(observation))
        #  action2=2
        #elif run_mode_str=='newvsold':
        #  observation2, reward2, is_terminal = self._run_one_step(self._agent22.get_action_from_observation(observation))
        #  action2=2
        #elif run_mode_str=='evalrandom':
        #  observation2, reward2, is_terminal = self._run_one_step(random.choice(self._environment.available_actions()))
        #  action2=2
        #elif run_mode_str=='eval':
        #  observation2, reward2, is_terminal = self._run_one_step(self.opponent.act(originalobservation, self._environment.available_actions()))
        #  action2=2
        #elif run_mode_str=='oldvsboss':
        #  observation2, reward2, is_terminal = self._run_one_step(self.opponent.act(originalobservation, self._environment.available_actions()))
        #  action2=2
        elif run_mode_str=='train1':
          self._agent.eval_mode=True
          if count < CFG.temp_thresh:
                best_child = self.mcts.search(game, node, CFG.temp_init)
          else:
                best_child = self.mcts.search(game, node, CFG.temp_final)
          action = best_child.action
          observation, reward, is_terminal = self._run_one_step(action) 
          
        elif run_mode_str=='train2':
          self._agent.eval_mode=False
          if count < CFG.temp_thresh:
                best_child = self.mcts.search(game, node, CFG.temp_init)
          else:
                best_child = self.mcts.search(game, node, CFG.temp_final)
        
          action = best_child.action
          observation, reward, is_terminal = self._run_one_step(action)
        
        else:
          print("Marco polo2")
          action2 = self._agent2.get_action_from_observation(observation)
          observation2, reward2, is_terminal = self._run_one_step(action2)
        
        reward=reward+reward2
        gameopponent.append([observation,-reward, is_terminal,action2])
        gameplayer.append([observationbeforefirstplayeraction,reward,is_terminal,action])
      
      else:
        gameplayer.append([observationbeforefirstplayeraction,reward,is_terminal,action])
        gameopponent[len(gameopponent)-1][1]=-reward
        gameopponent[len(gameopponent)-1][2]=True

      total_reward += reward
      step_number += 1
      first_time=False
      # Perform reward clipping.
      reward = np.clip(reward, -1, 1)
      if (is_terminal or
          step_number == self._max_steps_per_episode):
        # Stop the run loop once we reach the true end of episode.
        break
    
    if run_mode_str=='train1':
      for item in gameplayer:
          self._agent._store_transition([item[0]], item[3], item[1], item[2])
          if not item[2]:
            self._agent._train_step()
      
    if run_mode_str=='train2':
      for item in gameopponent:
            
          self._agent2._store_transition([item[0]], item[3], item[1], item[2])
          if not item[2]:
            self._agent2._train_step()

    if total_reward==-2:
      total_reward=-1
    
    return step_number, total_reward, gameplayer, gameopponent

  def _run_one_phase(self, min_steps, statistics, run_mode_str, is_silent):
    """Runs the agent/environment loop until a desired number of steps.
    We follow the Machado et al., 2017 convention of running full episodes,
    and terminating once we've run a minimum number of steps.
    Args:
      min_steps: int, minimum number of steps to generate in this phase.
      statistics: `IterationStatistics` object which records the experimental
        results.
      run_mode_str: str, describes the run mode for this agent.
    Returns:
      Tuple containing the number of steps taken in this phase (int), the sum of
        returns (float), and the number of episodes performed (int).
    """
    step_count = 0
    num_episodes = 0
    sum_returns = 0.
    
    while step_count < min_steps:
      start_time = time.time()
      episode_length, episode_return, self.gameplayer, self.gameopponent = self._run_one_episode(run_mode_str)
      time_delta = time.time() - start_time
      statistics.append({
          '{}_episode_lengths'.format(run_mode_str): episode_length,
          '{}_episode_returns'.format(run_mode_str): episode_return
      })
      step_count += episode_length
      self.step_count_total += episode_length
      sum_returns += episode_return
      num_episodes += 1
      # We use sys.stdout.write instead of tf.logging so as to flush frequently
      # without generating a line break.
      if not self.testing:
          self.fout.write('%d %f %d\n' % (self.step_count_total, episode_return, episode_length))
          self.fout.flush()
      if not is_silent:
        sys.stdout.write('Steps executed: {} '.format(step_count) +
                       'Episode length: {} '.format(episode_length) +
                       'Return: {} '.format(episode_return) +
                       'Average time one episode: {}\r'.format(episode_length/time_delta))
      sys.stdout.flush()
    if run_mode_str!='evalrandom' or run_mode_str!='train':
      for item in self.gameplayer:
        print(item)
    return step_count, sum_returns, num_episodes

  def _run_train_phase(self, statistics):
    """Run training phase.
    Args:
      statistics: `IterationStatistics` object which records the experimental
        results. Note - This object is modified by this method.
    Returns:
      num_episodes: int, The number of episodes run in this phase.
      average_reward: The average reward generated in this phase.
    """
    # Perform the training phase, during which the agent learns.
    self._agent.eval_mode = False
    self._agent2.eval_mode = False
    
    start_time = time.time()
    number_steps, sum_returns, num_episodes = self._run_one_phase(
        2000, statistics, 'train',False)
    average_return = sum_returns / num_episodes if num_episodes > 0 else 0.0
    statistics.append({'train_average_return': average_return})
    time_delta = time.time() - start_time
    tf.logging.info('Average undiscounted return per training episode: %.2f',
                    average_return)
    tf.logging.info('Average training steps per second: %.2f',
                    number_steps / time_delta)
    return num_episodes, average_return

  def _run_eval_phase(self, statistics):
    """Run evaluation phase.
    Args:
      statistics: `IterationStatistics` object which records the experimental
        results. Note - This object is modified by this method.
    Returns:
      num_episodes: int, The number of episodes run in this phase.
      average_reward: float, The average reward generated in this phase.
    """
    # Perform the evaluation phase -- no learning.
    self._agent.eval_mode = True
    self._agent2.eval_mode = True
    
    _, sum_returns, num_episodes = self._run_one_phase(
        80, statistics, 'eval',False)
    average_return = sum_returns / num_episodes if num_episodes > 0 else 0.0
    tf.logging.info('Average undiscounted return per evaluation episode: %.2f',
                    average_return)
    statistics.append({'eval_average_return': average_return})
    return num_episodes, average_return

  def _run_generic_phase(self, statistics, name):
    """Run evaluation phase.
    Args:
      statistics: `IterationStatistics` object which records the experimental
        results. Note - This object is modified by this method.
    Returns:
      num_episodes: int, The number of episodes run in this phase.
      average_reward: float, The average reward generated in this phase.
    """
    # Perform the evaluation phase -- no learning.
    self._agent.eval_mode = True
    self._agent2.eval_mode = True
    self._agent11.eval_mode = True
    self._agent22.eval_mode = True
    _, sum_returns, num_episodes = self._run_one_phase(
        1000, statistics, name,True)
    average_return = sum_returns / num_episodes if num_episodes > 0 else 0.0
    tf.logging.info('Average undiscounted return per evaluation episode: %.2f',
                    average_return)
    statistics.append({'eval_average_return': average_return})
    return num_episodes, average_return

  def _move_everything_in_folder(self, src, dst):
    src_files = os.listdir(src)
    for file_name in src_files:
      full_file_name = os.path.join(src, file_name)
      if os.path.isfile(full_file_name):
        shutil.copy(full_file_name, dst)

  def _delete_everything_in_folder(self, path):
    
    files = glob.glob(path + '/*')
    for f in files:
      os.remove(f)
    
    files = glob.glob(path + '/.*')
    for f in files:
      os.remove(f)

  def _run_one_iteration(self, iteration, firstiteration):
    """Runs one iteration of agent/environment interaction.
    An iteration involves running several episodes until a certain number of
    steps are obtained. The interleaving of train/eval phases implemented here
    are to match the implementation of (Mnih et al., 2015).
    Args:
      iteration: int, current iteration number, used as a global_step for saving
        Tensorboard summaries.
    Returns:
      A dict containing summary statistics for this iteration.
    """
    
    #if firstiteration:
    #  experiment_data = self._agent2.bundle_and_checkpoint(self.PATH + "test",
    #                                                    iteration)
    #  if experiment_data:
    #    experiment_data['current_iteration'] = iteration
    #    experiment_data['logs'] = self._logger.data
    #    self._checkpointertest.save_checkpoint(iteration, experiment_data)
    #if iteration==25: 
    #    experiment_data = self._checkpointertest.load_checkpoint(0)
    #    x=False
    #   x=self._agent2.unbundle(self.PATH + "test", 0, experiment_data)
    # 
    
    #self.to_graph = tf.Graph() # graph where everything above will be copied to

    #self._q_argmax = tf.contrib.copy_graph.copy_op_to_graph(self._agent._q_argmax, self.to_graph,[])
    #self._q_argmax = tf.contrib.copy_graph.copy_op_to_graph(self._agent.state_ph, self.to_graph,[])
    #self._q_argmax = tf.contrib.copy_graph.copy_op_to_graph(self._agent.validmoves_ph, self.to_graph,[])
    
    
    if firstiteration:
      self._my_checkpoint_experiment(iteration, 'latest', 1)
      self._my_checkpoint_experiment(iteration, 'latest', 2)
    self._my_checkpoint_experiment(iteration, 'player', 1)
    
    
    q_argmax2 = self._agent._sess.run(self._agent._q_values, {self._agent.state_ph: [[[[0],[0],[0],[0],[0],[0],[0],[0],[0],[50]]]], self._agent.validmoves_ph:
       
         [1,3,5,7,8]})
         
    #print("Q Before loading", q_argmax2)
    
    
    self._my_initialize_resume('crpt', 'latest',  self.latest1, 11)
    self._my_initialize_resume('crpt', 'latest',  self.latest2, 22)
    
    
    statistics = iteration_statistics.IterationStatistics()
    tf.logging.info('Starting iteration %d', iteration)
    print("SELF.COUNTER:", self.counter)
    if iteration>50 and self.counter>=1:
      self.counter=0
      self.player1_turn_training=not self.player1_turn_training
    
    if iteration<50000:
      num_episodes_train, average_reward_train = self._run_generic_phase(
        statistics,'train1')
    
      num_episodes_train, average_reward_train = self._run_generic_phase(
        statistics,'train2')
      print("TRAIN TRAIN1")
    else:
      if self.player1_turn_training:
        num_episodes_train, average_reward_train = self._run_generic_phase(
        statistics,'train1')
        print("TRAIN TRAIN2")
      else:
        num_episodes_train, average_reward_train = self._run_generic_phase(
        statistics,'train2')
        print("TRAIN TRAIN3")    
    
    
    num_episodes_eval, average_reward_eval = self._run_eval_phase(
        statistics)
    
    
    print("PLAYER1 TURN TRAINING:", self.player1_turn_training)
    print("EVAL EVAL")
    
    
    #numep, evalaveragereward =  self._run_generic_phase(
    #    statistics,'preveval')
    
    
    
    print("LATEST1:", self.latest1)
    print("LATEST2:", self.latest2)
    
    
    numep, oldvsnew = self._run_generic_phase(
        statistics,'oldvsnew')
    numep, newvsold = self._run_generic_phase(
        statistics,'newvsold')
    numep, oldvsold = self._run_generic_phase(
        statistics,'oldvsold')
    print("OLD VS NEW:",oldvsnew)
    print("NEW VS OLD:",newvsold)
    print("OLD VS OLD:", oldvsold)
    numep, oldvsboss = self._run_generic_phase(
        statistics,'oldvsboss')
    print("OLD VS BOSS", oldvsboss)
    numep, evalrandom = self._run_generic_phase(
        statistics,'evalrandom')
    print("EVALRANDOM:", evalrandom)
    print("PLAYER1 TURN TRAINING2:", self.player1_turn_training)
    if self.player1_turn_training:
      self.compare_result_against=newvsold
      self.who_to_change=1
      self.who_to_change_latest=self.latest1
      if newvsold>0.05:
        self.counter+=1
      else:
        self.counter=0
    else:
      self.compare_result_against=oldvsnew
      self.who_to_change=2
      self.who_to_change_latest=self.latest2
      if oldvsnew<0.05:
        self.counter+=1
      else:
        self.counter=0
    
    if iteration<50000:
      self._my_checkpoint_experiment(iteration, 'latest', 1)
      self.latest1=iteration
      self._my_checkpoint_experiment(iteration, 'latest', 2)
      self.latest2=iteration
      self.counter=0
    else:
      if self.player1_turn_training:
        if oldvsold + 0.05 <  self.compare_result_against:
          self._my_checkpoint_experiment(iteration, 'latest', self.who_to_change)
          if self.who_to_change==1:
            self.latest1=iteration
          else:
            self.latest2=iteration
          print("CROOK: Changing player "+ str(self.who_to_change)  +" version")
        else:
          self._my_initialize_resume('crpt','latest',self.who_to_change_latest, self.who_to_change)
          print("CROOK: Staying with old player "+str(self.who_to_change)  )
      else:
        if oldvsold - 0.05 >  self.compare_result_against:
          self._my_checkpoint_experiment(iteration, 'latest', self.who_to_change)
          if self.who_to_change==1:
            self.latest1=iteration
          else:
            self.latest2=iteration
          print("CROOK: Changing player "+ str(self.who_to_change)  +" version")
        else:
          self._my_initialize_resume('crpt','latest',self.who_to_change_latest, self.who_to_change)
          print("CROOK: Staying with old player "+str(self.who_to_change)  )
      
    
    #variables = tf.trainable_variables()
    #print("Weight matrix: {0}".format(self._agent._sess.run(variables[0]))) 
    
    print("EPSILON:")
    print(self._agent.epsilon_fn(
          self._agent.epsilon_decay_period,
          self._agent.training_steps,
          self._agent.min_replay_history,
          self._agent.epsilon_train))
    
    
    
    #self._save_tensorboard_summaries(iteration, num_episodes_train,
    #                                 average_reward_train, num_episodes_eval,
    #                                 average_reward_eval)
    if not self.testing:
        self.fout_test.write('%d %f %d\n' % (iteration, average_reward_eval, num_episodes_eval))
        self.fout_test.flush()
    return statistics.data_lists

  def _save_tensorboard_summaries(self, iteration,
                                  num_episodes_train,
                                  average_reward_train,
                                  num_episodes_eval,
                                  average_reward_eval):
    """Save statistics as tensorboard summaries.
    Args:
      iteration: int, The current iteration number.
      num_episodes_train: int, number of training episodes run.
      average_reward_train: float, The average training reward.
      num_episodes_eval: int, number of evaluation episodes run.
      average_reward_eval: float, The average evaluation reward.
    """
    summary = tf.Summary(value=[
        tf.Summary.Value(tag='Train/NumEpisodes',
                         simple_value=num_episodes_train),
        tf.Summary.Value(tag='Train/AverageReturns',
                         simple_value=average_reward_train),
        tf.Summary.Value(tag='Eval/NumEpisodes',
                         simple_value=num_episodes_eval),
        tf.Summary.Value(tag='Eval/AverageReturns',
                         simple_value=average_reward_eval)
    ])
    self._summary_writer.add_summary(summary, iteration)
  def _log_experiment(self, iteration, statistics):
    """Records the results of the current iteration.
    Args:
      iteration: int, iteration number.
      statistics: `IterationStatistics` object containing statistics to log.
    """
    self._logger['iteration_{:d}'.format(iteration)] = statistics
    if iteration % self._log_every_n == 0:
      self._logger.log_to_file(self._logging_file_prefix, iteration)


  def _my_checkpoint_experiment(self,  iteration, folder_type, player):
    """Checkpoint experiment data.
    Args:
      iteration: int, iteration number for checkpointing.
    """
    
    if player==1:
      experiment_data = self._agent.bundle_and_checkpoint(self.PATH + ""+ folder_type +"3",  iteration)
      if experiment_data:
        experiment_data['current_iteration'] = iteration
        experiment_data['logs'] = self._logger.data
        if folder_type=='latest':
          self._checkpointerlatest1.save_checkpoint(iteration, experiment_data)
        else:
          self._checkpointer2.save_checkpoint(iteration, experiment_data)      
      if folder_type=='latest':
        self._delete_everything_in_folder(self.PATH + ""+ folder_type +"1")
        self._move_everything_in_folder(self.PATH + ""+ folder_type +"3", self.PATH + ""+ folder_type +"1")
    
    else:
      experiment_data = self._agent2.bundle_and_checkpoint(self.PATH + ""+folder_type+"4", iteration)
      if experiment_data:
        experiment_data['current_iteration'] = iteration
        experiment_data['logs'] = self._logger.data
        if folder_type=='latest':
          self._checkpointerlatest2.save_checkpoint(iteration, experiment_data)
        else:
          self._checkpointer2.save_checkpoint(iteration, experiment_data)
      if folder_type=='latest':
        self._delete_everything_in_folder(self.PATH + ""+ folder_type +"2")
        self._move_everything_in_folder(self.PATH + ""+ folder_type +"4", self.PATH + ""+ folder_type +"2")
    
    
    
  def _checkpoint_experiment(self, iteration):
    """Checkpoint experiment data.
    Args:
      iteration: int, iteration number for checkpointing.
    """
    experiment_data = self._agent.bundle_and_checkpoint(self.PATH + "checkpoints",
                                                        iteration)
    if experiment_data:
      experiment_data['current_iteration'] = iteration
      experiment_data['logs'] = self._logger.data
      self._checkpointer.save_checkpoint(iteration, experiment_data)

  def run_experiment(self):
    """Runs a full experiment, spread over multiple iterations."""
    i=3
    j=i
    i=2
    print("MAKEUP:", j)
    print(tf.get_default_graph())
    tf.logging.info('Beginning training...')
    if self._num_iterations <= self._start_iteration:
      tf.logging.warning('num_iterations (%d) < start_iteration(%d)',
                         self._num_iterations, self._start_iteration)
      return
    
    self._delete_everything_in_folder(self.PATH + "player2")
    self._delete_everything_in_folder(self.PATH + "latest1")
    self._delete_everything_in_folder(self.PATH + "latest2")
    self._delete_everything_in_folder(self.PATH + "latest3")
    self._delete_everything_in_folder(self.PATH + "latest4")
    self.counter=0
    self.evalwhoamiplaying=1
    self.latest1=0
    self.latest2=0
    self.player1_turn_training=False
    first=True
    for iteration in range(self._start_iteration, 2000):
      statistics = self._run_one_iteration(iteration, first)
      first=False
      self._log_experiment(iteration, statistics)
      self._checkpoint_experiment(iteration)


@gin.configurable
class TrainRunner(Runner):
  """Object that handles running experiments.
  The `TrainRunner` differs from the base `Runner` class in that it does not
  the evaluation phase. Checkpointing and logging for the train phase are
  preserved as before.
  """

  def __init__(self, base_dir, create_agent_fn,
               create_environment_fn=atari_lib.create_atari_environment):
    """Initialize the TrainRunner object in charge of running a full experiment.
    Args:
      base_dir: str, the base directory to host all required sub-directories.
      create_agent_fn: A function that takes as args a Tensorflow session and an
        environment, and returns an agent.
      create_environment_fn: A function which receives a problem name and
        creates a Gym environment for that problem (e.g. an Atari 2600 game).
    """
    tf.logging.info('Creating TrainRunner ...')
    super(TrainRunner, self).__init__(base_dir, create_agent_fn,
                                      create_environment_fn)
    self._agent.eval_mode = False
    

  def _run_one_iteration(self, iteration):
    """Runs one iteration of agent/environment interaction.
    An iteration involves running several episodes until a certain number of
    steps are obtained. This method differs from the `_run_one_iteration` method
    in the base `Runner` class in that it only runs the train phase.
    Args:
      iteration: int, current iteration number, used as a global_step for saving
        Tensorboard summaries.
    Returns:
      A dict containing summary statistics for this iteration.
    """
    statistics = iteration_statistics.IterationStatistics()
    num_episodes_train, average_reward_train = self._run_train_phase(
        statistics)

    self._save_tensorboard_summaries(iteration, num_episodes_train,
                                     average_reward_train)
    return statistics.data_lists

  def _save_tensorboard_summaries(self, iteration, num_episodes,
                                  average_reward):
    """Save statistics as tensorboard summaries."""
    summary = tf.Summary(value=[
        tf.Summary.Value(tag='Train/NumEpisodes', simple_value=num_episodes),
        tf.Summary.Value(
            tag='Train/AverageReturns', simple_value=average_reward),
    ])
    self._summary_writer.add_summary(summary, iteration)