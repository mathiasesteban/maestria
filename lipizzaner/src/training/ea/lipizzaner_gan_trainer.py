import random
from time import time
from collections import OrderedDict

import numpy as np
import torch

from distribution.concurrent_populations import ConcurrentPopulations
from distribution.neighbourhood import Neighbourhood
from helpers.configuration_container import ConfigurationContainer
from helpers.db_logger import DbLogger
from helpers.population import TYPE_GENERATOR, TYPE_DISCRIMINATOR
from helpers.pytorch_helpers import to_pytorch_variable
from helpers.pytorch_helpers import noise
from training.ea.ea_trainer import EvolutionaryAlgorithmTrainer
from training.mixture.mixed_generator_dataset import MixedGeneratorDataset
from training.mixture.score_factory import ScoreCalculatorFactory

from data.network_data_loader import generate_random_sequences


class LipizzanerGANTrainer(EvolutionaryAlgorithmTrainer):
    """
    Distributed, asynchronous trainer for coevolutionary GANs. Uses the standard Goodfellow GAN approach.
    (Without discriminator mixture)
    """

    def __init__(self, dataloader, network_factory, population_size=10, tournament_size=2, mutation_probability=0.9,
                 n_replacements=1, sigma=0.25, alpha=0.25, default_adam_learning_rate=0.001, calc_mixture=False,
                 mixture_sigma=0.01, score_sample_size=10000, discriminator_skip_each_nth_step=0,
                 enable_selection=True, fitness_sample_size=10000, calculate_net_weights_dist=False,
                 fitness_mode='worst',  es_generations=10, es_score_sample_size=10000, es_random_init=False,
                 checkpoint_period=0):

        super().__init__(dataloader, network_factory, population_size, tournament_size, mutation_probability,
                         n_replacements, sigma, alpha)

        self.batch_number = 0
        self.cc = ConfigurationContainer.instance()

        self._default_adam_learning_rate = self.settings.get('default_adam_learning_rate', default_adam_learning_rate)
        self._discriminator_skip_each_nth_step = self.settings.get('discriminator_skip_each_nth_step',
                                                                   discriminator_skip_each_nth_step)
        self._enable_selection = self.settings.get('enable_selection', enable_selection)
        self.mixture_sigma = self.settings.get('mixture_sigma', mixture_sigma)

        self.neighbourhood = Neighbourhood.instance()

        for i, individual in enumerate(self.population_gen.individuals):
            individual.learning_rate = self._default_adam_learning_rate
            individual.id = '{}/G{}'.format(self.neighbourhood.cell_number, i)
        for i, individual in enumerate(self.population_dis.individuals):
            individual.learning_rate = self._default_adam_learning_rate
            individual.id = '{}/D{}'.format(self.neighbourhood.cell_number, i)

        self.concurrent_populations = ConcurrentPopulations.instance()
        self.concurrent_populations.generator = self.population_gen
        self.concurrent_populations.discriminator = self.population_dis
        self.concurrent_populations.unlock()

        experiment_id = self.cc.settings['general']['logging'].get('experiment_id', None)
        self.db_logger = DbLogger(current_experiment=experiment_id)

        if 'fitness' in self.settings:
            self.fitness_sample_size = self.settings['fitness'].get('fitness_sample_size', fitness_sample_size)
            self.fitness_loaded = self.dataloader.load()
            self.fitness_iterator = iter(self.fitness_loaded)  # Create iterator for fitness loader
            self.fitness_batch_size = self.settings['fitness'].get('fitness_batch_size', None)

            # Determine how to aggregate fitness calculated among neighbourhood
            self.fitness_mode = self.settings['fitness'].get('fitness_mode', fitness_mode)
            if self.fitness_mode not in ['worse', 'best', 'average']:
                raise NotImplementedError("Invalid argument for fitness_mode: {}".format(self.fitness_mode))
        else:
            # TODO: Add code for safe implementation & error handling
            raise KeyError("Fitness section must be defined in configuration file")

        if 'score' in self.settings and self.settings['score'].get('enabled', calc_mixture):
            self.score_calc = ScoreCalculatorFactory.create()
            self.score_sample_size = self.settings['score'].get('sample_size', score_sample_size)
            self.score = float('inf') if self.score_calc.is_reversed else float('-inf')
            self.mixture_generator_samples_mode = self.cc.settings['trainer']['mixture_generator_samples_mode']
        elif 'optimize_mixture' in self.settings:
            self.score_calc = ScoreCalculatorFactory.create()
            self.score = float('inf') if self.score_calc.is_reversed else float('-inf')
        else:
            self.score_sample_size = score_sample_size
            self.score_calc = None
            self.score = 0

        if 'optimize_mixture' in self.settings:
            self.optimize_weights_at_the_end = True
            self.score_sample_size = self.settings['optimize_mixture'].get('sample_size', es_score_sample_size)
            self.es_generations = self.settings['optimize_mixture'].get('es_generations', es_generations)
            self.es_random_init = self.settings['optimize_mixture'].get('es_random_init', es_random_init)
            self.mixture_sigma = self.settings['optimize_mixture'].get('mixture_sigma', mixture_sigma)
            self.mixture_generator_samples_mode = self.cc.settings['trainer']['mixture_generator_samples_mode']
        else:
            self.optimize_weights_at_the_end = False

        n_iterations = self.cc.settings['trainer'].get('n_iterations', 0)
        assert 0 <= checkpoint_period <= n_iterations, 'Checkpoint period paramenter (checkpoint_period) should be ' \
                                                       'between 0 and the number of iterations (n_iterations).'
        self.checkpoint_period = self.cc.settings['general'].get('checkpoint_period', checkpoint_period)

        self.evaluate_subpopulations_every = self.settings.get('evaluate_subpopulations_every', 1)

        self.individuals_sampling_size = self.settings.get('subpopulation_sample_size', None)


    def train(self, n_iterations, stop_event=None):
        loaded = self.dataloader.load()
        selection_applied_apply_replacement = False

        for iteration in range(n_iterations):
            self._logger.debug('Iteration {} started'.format(iteration + 1))
            self.cc.settings['network']['iteration'] = iteration
            start_time = time()

            if self.evaluate_subpopulations_every == 1 or (iteration % self.evaluate_subpopulations_every == 0):
                all_generators = self.neighbourhood.all_generators(self.individuals_sampling_size)
                all_discriminators = self.neighbourhood.all_discriminators(self.individuals_sampling_size)
                local_generators = self.neighbourhood.local_generators
                local_discriminators = self.neighbourhood.local_discriminators

                # Log the name of individuals in entire neighborhood and local individuals for every iteration
                # (to help tracing because individuals from adjacent cells might be from different iterations)
                self._logger.info('Neighborhood located in possition {} of the grid'.format(self.neighbourhood.grid_position))
                self._logger.info('Generators in current neighborhood are {}'.format([
                    individual.name for individual in all_generators.individuals
                ]))
                self._logger.info('Discriminators in current neighborhood are {}'.format([
                    individual.name for individual in all_discriminators.individuals
                ]))
                self._logger.info('Local generators in current neighborhood are {}'.format([
                    individual.name for individual in local_generators.individuals
                ]))
                self._logger.info('Local discriminators in current neighborhood are {}'.format([
                    individual.name for individual in local_discriminators.individuals
                ]))

                self._logger.info('L2 distance between all generators weights: {}'.format(all_generators.net_weights_dist))
                self._logger.info(
                    'L2 distance between all discriminators weights: {}'.format(all_discriminators.net_weights_dist))

                self._logger.info('Iteration: {}. ----------------------------------- Applying selection'.format(iteration+1))
                selection_applied_apply_replacement = True
                new_populations = {}

                # Create random dataset to evaluate fitness in each iterations
                fitness_samples = self.generate_random_fitness_samples(self.fitness_sample_size)

                # Fitness evaluation
                self._logger.debug('Evaluating fitness')
                # Splitting fitness_samples
                self._logger.debug('Non-split fitness samples size: {}. {}'.format(len(fitness_samples), fitness_samples[0].size()))
                split = False
                if self.fitness_batch_size is not None:
                    split = True
                    fitness_samples = torch.split(fitness_samples, int(self.fitness_batch_size))
                    # fitness_samples = torch.split(fitness_samples, int(len(fitness_samples)/self.fitness_batch_size))
                    self._logger.debug('split fitness samples size: {}. {}'.format(len(fitness_samples), fitness_samples[0].size()))
                self.evaluate_fitness(all_generators, all_discriminators, fitness_samples, self.fitness_mode, split)
                self.evaluate_fitness(all_discriminators, all_generators, fitness_samples, self.fitness_mode, split)
                self._logger.debug('Finished evaluating fitness')

                # Tournament selection
                if self._enable_selection:
                    self._logger.debug('Started tournament selection')
                    new_populations[TYPE_GENERATOR] = self.tournament_selection(all_generators,
                                                                                TYPE_GENERATOR,
                                                                                is_logging=True)
                    new_populations[TYPE_DISCRIMINATOR] = self.tournament_selection(all_discriminators,
                                                                                    TYPE_DISCRIMINATOR,
                                                                                    is_logging=True)
                    self._logger.debug('Finished tournament selection')

            self.batch_number = 0
            data_iterator = iter(loaded)
            while self.batch_number < len(loaded):
                if self.cc.settings['dataloader']['dataset_name'] == 'network_traffic':
                    input_data = to_pytorch_variable(next(data_iterator))
                    batch_size = input_data.size(0)
                else:
                    input_data = next(data_iterator)[0]
                    batch_size = input_data.size(0)
                    input_data = to_pytorch_variable(self.dataloader.transpose_data(input_data))

                # Quit if requested
                if stop_event is not None and stop_event.is_set():
                    self._logger.warning('External stop requested.')
                    return self.result()

                attackers = new_populations[TYPE_GENERATOR] if self._enable_selection else local_generators
                defenders = new_populations[TYPE_DISCRIMINATOR] if self._enable_selection else all_discriminators
                input_data = self.step(local_generators, attackers, defenders, input_data, self.batch_number, loaded,
                                       data_iterator, iteration)

                if self._discriminator_skip_each_nth_step == 0 or self.batch_number % (
                        self._discriminator_skip_each_nth_step + 1) == 0:
                    self._logger.debug('Skipping discriminator step')

                    attackers = new_populations[TYPE_DISCRIMINATOR] if self._enable_selection else local_discriminators
                    defenders = new_populations[TYPE_GENERATOR] if self._enable_selection else all_generators
                    input_data = self.step(local_discriminators, attackers, defenders, input_data, self.batch_number,
                                           loaded, data_iterator, iteration)

                self._logger.info('Iteration {}, Batch {}/{}'.format(iteration + 1, self.batch_number, len(loaded)))

                # If n_batches is set to 0, all batches will be used
                if self.is_last_batch(self.batch_number):
                    break

                self.batch_number += 1

            # Perform selection first before mutation of mixture_weights
            # Replace the worst with the best new
            if self._enable_selection:
                if selection_applied_apply_replacement and ((iteration+1) % (self.evaluate_subpopulations_every) == 0
                        or (iteration+1) == n_iterations):
                    selection_applied_apply_replacement = False

                    if (iteration+1) == n_iterations:
                        all_generators = self.neighbourhood.all_generators(None)
                        all_discriminators = self.neighbourhood.all_discriminators(None)

                    self._logger.info('Iteration: {}. -----------------------------Applying Replacement'.format(iteration+1))

                    # Evaluate fitness of new_populations against neighborhood
                    self.evaluate_fitness(new_populations[TYPE_GENERATOR], all_discriminators, fitness_samples,
                                          self.fitness_mode, split)
                    self.evaluate_fitness(new_populations[TYPE_DISCRIMINATOR], all_generators, fitness_samples,
                                          self.fitness_mode, split)
                    self.concurrent_populations.lock()
                    local_generators.replacement(new_populations[TYPE_GENERATOR], self._n_replacements, is_logging=True)
                    local_generators.sort_population(is_logging=True)
                    local_discriminators.replacement(new_populations[TYPE_DISCRIMINATOR], self._n_replacements,
                                                     is_logging=True)
                    local_discriminators.sort_population(is_logging=True)
                    self.concurrent_populations.unlock()

                    # Update individuals' iteration and id after replacement and logging to ease tracing
                    for i, individual in enumerate(local_generators.individuals):
                        individual.id = '{}/G{}'.format(self.neighbourhood.cell_number, i)
                        individual.iteration = iteration + 1
                    for i, individual in enumerate(local_discriminators.individuals):
                        individual.id = '{}/D{}'.format(self.neighbourhood.cell_number, i)
                        individual.iteration = iteration + 1

                    del fitness_samples
                    torch.cuda.empty_cache()
                # else: # If there is not replacement, we update the fitness
                #     self._logger.info(
                #         'Iteration: {}. -----------------------------Evaluating fitness because there is not replacement'.format(iteration+1))
                #     self.evaluate_fitness(local_generators, all_discriminators, fitness_samples, self.fitness_mode,
                #                           split)
                #     self.evaluate_fitness(local_discriminators, all_generators, fitness_samples, self.fitness_mode,
                #                           split)
            else:
                # Re-evaluate fitness of local_generators and local_discriminators against neighborhood
                if (iteration+1) % (self.evaluate_subpopulations_every) == 0 or (iteration+1) == n_iterations:
                    self.evaluate_fitness(local_generators, all_discriminators, fitness_samples, self.fitness_mode, split)
                    self.evaluate_fitness(local_discriminators, all_generators, fitness_samples, self.fitness_mode, split)
                    del fitness_samples
                    torch.cuda.empty_cache()


            # Mutate mixture weights after selection
            if (self.evaluate_subpopulations_every == 0 or ((iteration) % self.evaluate_subpopulations_every == 0)) and not self.optimize_weights_at_the_end:
                self.mutate_mixture_weights_with_score(input_data)  # self.score is updated here

            stop_time = time()

            path_real_images, path_fake_images = \
                self.log_results(batch_size, iteration, input_data, loaded,
                                 lr_gen=self.concurrent_populations.generator.individuals[0].learning_rate,
                                 lr_dis=self.concurrent_populations.discriminator.individuals[0].learning_rate,
                                 score=self.score, mixture_gen=self.neighbourhood.mixture_weights_generators,
                                 mixture_dis=None)

            if self.db_logger.is_enabled:
                self.db_logger.log_results(iteration, self.neighbourhood, self.concurrent_populations,
                                           self.score, stop_time - start_time,
                                           path_real_images, path_fake_images)

            if self.checkpoint_period>0 and (iteration+1)%self.checkpoint_period==0:

                self.save_checkpoint(all_generators.individuals, all_discriminators.individuals,
                                     self.neighbourhood.cell_number, self.neighbourhood.grid_position)


        if self.optimize_weights_at_the_end:
            all_generators = self.neighbourhood.all_generators(None)
            all_discriminators = self.neighbourhood.all_discriminators(None)
            local_generators = self.neighbourhood.local_generators
            local_discriminators = self.neighbourhood.local_discriminators

            self.optimize_generator_mixture_weights()

            path_real_images, path_fake_images = \
                self.log_results(batch_size, iteration+1, input_data, loaded,
                                 lr_gen=self.concurrent_populations.generator.individuals[0].learning_rate,
                                 lr_dis=self.concurrent_populations.discriminator.individuals[0].learning_rate,
                                 score=self.score, mixture_gen=self.neighbourhood.mixture_weights_generators,
                                 mixture_dis=self.neighbourhood.mixture_weights_discriminators)

            if self.db_logger.is_enabled:
                self.db_logger.log_results(iteration+1, self.neighbourhood, self.concurrent_populations,
                                           self.score, stop_time - start_time,
                                           path_real_images, path_fake_images)

        torch.cuda.empty_cache()
        return self.result()

    def optimize_generator_mixture_weights(self):
        generators = self.neighbourhood.best_generators
        weights_generators = self.neighbourhood.mixture_weights_generators

        # Not necessary for single-cell grids, as mixture must always be [1]
        if self.neighbourhood.grid_size == 1:
            return

        # Create random vector from latent space
        z_noise = noise(self.score_sample_size, generators.individuals[0].genome.data_size)

        # Include option to start from random weights
        if self.es_random_init:
            aux_weights = np.random.rand(len(weights_generators))
            aux_weights /= np.sum(aux_weights)
            weights_generators = OrderedDict(zip(weights_generators.keys(), aux_weights))
            self.neighbourhood.mixture_weights_generators = weights_generators

        dataset = MixedGeneratorDataset(generators, weights_generators,
                                          self.score_sample_size, self.mixture_generator_samples_mode, z_noise)

        self.score = self.score_calc.calculate(dataset)[0]
        init_score = self.score

        self._logger.info(
            'Mixture weight mutation - Starting mixture weights optimization ...')
        self._logger.info('Init score: {}\tInit weights: {}.'.format(init_score, weights_generators))

        for g in range(self.es_generations):

            # Mutate mixture weights
            z = np.random.normal(loc=0, scale=self.mixture_sigma, size=len(weights_generators))
            transformed = np.asarray([value for _, value in weights_generators.items()])
            transformed += z

            # Don't allow negative values, normalize to sum of 1.0
            transformed = np.clip(transformed, 0, None)
            transformed /= np.sum(transformed)
            new_mixture_weights = OrderedDict(zip(weights_generators.keys(), transformed))

            # TODO: Testing the idea of not generating the images again
            dataset = MixedGeneratorDataset(generators, new_mixture_weights,
                                              self.score_sample_size,
                                              self.mixture_generator_samples_mode, z_noise)

            if self.score_calc is not None:
                score_after_mutation = self.score_calc.calculate(dataset)[0]
                self._logger.info(
                    'Mixture weight mutation - Generation: {} \tScore of new weights: {}\tNew weights: {}.'.format(g,
                                                                                               score_after_mutation,
                                                                                               new_mixture_weights))

                # For fid the lower the better, for inception_score, the higher the better
                if (score_after_mutation < self.score and self.score_calc.is_reversed) \
                        or (score_after_mutation > self.score and (not self.score_calc.is_reversed)):
                    weights_generators = new_mixture_weights
                    self.score = score_after_mutation
                    self._logger.info(
                        'Mixture weight mutation - Generation: {} \tNew score: {}\tWeights changed to: {}.'.format(g,
                                                                                                    self.score,
                                                                                                    weights_generators))
        self.neighbourhood.mixture_weights_generators = weights_generators

        self._logger.info(
            'Mixture weight mutation - Score before mixture weight optimzation: {}\tScore after mixture weight optimzation: {}.'.format(
                                                                                                        init_score,
                                                                                                        self.score))

    def step(self, original, attacker, defender, input_data, i, loaded, data_iterator, training_epoch=None):
        self.mutate_hyperparams(attacker)
        return self.update_genomes(attacker, defender, input_data, loaded, data_iterator, training_epoch)

    def is_last_batch(self, i):
        return self.dataloader.n_batches != 0 and self.dataloader.n_batches - 1 == i

    def result(self):
        return ((self.concurrent_populations.generator.individuals[0].genome,
                 self.concurrent_populations.generator.individuals[0].fitness),
                (self.concurrent_populations.discriminator.individuals[0].genome,
                 self.concurrent_populations.discriminator.individuals[0].fitness))

    def mutate_hyperparams(self, population):
        loc = -(self._default_adam_learning_rate / 10)
        deltas = np.random.normal(loc=loc, scale=self._default_adam_learning_rate, size=len(population.individuals))
        deltas[np.random.rand(*deltas.shape) < 1 - self._mutation_probability] = 0
        for i, individual in enumerate(population.individuals):
            individual.learning_rate = max(0, individual.learning_rate + deltas[i] * self._alpha)

    def update_genomes(self, population_attacker, population_defender, input_var, loaded, data_iterator, training_epoch):

        # TODO Currently picking random opponent, introduce parameter for this
        defender = random.choice(population_defender.individuals).genome

        for individual_attacker in population_attacker.individuals:
            attacker = individual_attacker.genome
            optimizer = torch.optim.Adam(attacker.net.parameters(),
                                         lr=individual_attacker.learning_rate,
                                         betas=(0.5, 0.999))

            # Restore previous state dict, if available
            if individual_attacker.optimizer_state is not None:
                optimizer.load_state_dict(individual_attacker.optimizer_state)
            loss = attacker.compute_loss_against(defender, input_var, training_epoch)[0]

            attacker.net.zero_grad()
            defender.net.zero_grad()
            loss.backward()
            optimizer.step()

            individual_attacker.optimizer_state = optimizer.state_dict()

        return input_var

    @staticmethod
    def evaluate_fitness(population_attacker, population_defender, input_var, fitness_mode, split=False):
        # Single direction only: Evaluate fitness of attacker based on defender
        # TODO: Simplify and refactor this function
        def compare_fitness(curr_fitness, fitness, mode):
            # The initial fitness value is -inf before evaluation started, so we
            # directly adopt the curr_fitness when -inf is encountered
            if mode == 'best':
                if curr_fitness < fitness or fitness == float('-inf'):
                    return curr_fitness
            elif mode == 'worse':
                if curr_fitness > fitness or fitness == float('-inf'):
                    return curr_fitness
            elif mode == 'average':
                if fitness == float('-inf'):
                    return curr_fitness
                else:
                    return fitness + curr_fitness

            return fitness

        import logging
        _logger = logging.getLogger(__name__)
        _logger.debug('------------ Evaluating fitness ----------------')


        for individual_attacker in population_attacker.individuals:
            _logger.debug(' Atacker {}'.format(individual_attacker.id))
            individual_attacker.fitness = float(
                '-inf')  # Reinitalize before evaluation started (Needed for average fitness)
            for individual_defender in population_defender.individuals:
                _logger.debug('   - Defender {}'.format(individual_attacker.id))
                #Iterate through input
                if split:
                    input_iterator = iter(input_var)
                    batch_number = 0
                    fitness_attacker_acum = 0
                    max_batches = len(input_var)
                    while batch_number < max_batches: #len(input_var):
                        _input = next(input_iterator)
                        fitness_attacker_acum += float(individual_attacker.genome.compute_loss_against(
                        individual_defender.genome, _input)[0])
                        batch_number += 1
                        _logger.debug('     Batch: {}/{}'.format(batch_number, max_batches))
                    fitness_attacker = fitness_attacker_acum / batch_number
                else:
                    fitness_attacker = float(individual_attacker.genome.compute_loss_against(
                        individual_defender.genome, input_var)[0])

                individual_attacker.fitness = compare_fitness(fitness_attacker, individual_attacker.fitness,
                                                              fitness_mode)
                _logger.debug('     Fitness: {}'.format(individual_attacker.fitness))
            if fitness_mode == 'average':
                individual_attacker.fitness /= len(population_defender.individuals)

        if split:
            del _input
            del input_iterator
        del input_var
        torch.cuda.empty_cache()


    def mutate_mixture_weights_with_score(self, input_data):
        if self.score_calc is not None:
            self._logger.info('Calculating FID/inception score.')
            # Not necessary for single-cell grids, as mixture must always be [1]
            if self.neighbourhood.grid_size == 1:
                best_generators = self.neighbourhood.best_generators

                dataset = MixedGeneratorDataset(best_generators,
                                                self.neighbourhood.mixture_weights_generators,
                                                self.score_sample_size,
                                                self.cc.settings['trainer']['mixture_generator_samples_mode'])
                self.score = self.score_calc.calculate(dataset)[0]
            else:
                # Mutate mixture weights
                z = np.random.normal(loc=0, scale=self.mixture_sigma,
                                     size=len(self.neighbourhood.mixture_weights_generators))
                transformed = np.asarray([value for _, value in self.neighbourhood.mixture_weights_generators.items()])
                transformed += z
                # Don't allow negative values, normalize to sum of 1.0
                transformed = np.clip(transformed, 0, None)
                transformed /= np.sum(transformed)

                new_mixture_weights_generators = OrderedDict(
                    zip(self.neighbourhood.mixture_weights_generators.keys(), transformed))

                best_generators = self.neighbourhood.best_generators

                dataset_before_mutation = MixedGeneratorDataset(best_generators,
                                                                self.neighbourhood.mixture_weights_generators,
                                                                self.score_sample_size,
                                                                self.cc.settings['trainer'][
                                                                    'mixture_generator_samples_mode'])
                score_before_mutation = self.score_calc.calculate(dataset_before_mutation)[0]
                del dataset_before_mutation

                dataset_after_mutation = MixedGeneratorDataset(best_generators,
                                                               new_mixture_weights_generators,
                                                               self.score_sample_size,
                                                               self.cc.settings['trainer'][
                                                                   'mixture_generator_samples_mode'])
                score_after_mutation = self.score_calc.calculate(dataset_after_mutation)[0]
                del dataset_after_mutation

                # For fid the lower the better, for inception_score, the higher the better
                if (score_after_mutation < score_before_mutation and self.score_calc.is_reversed) \
                        or (score_after_mutation > score_before_mutation and (not self.score_calc.is_reversed)):
                    # Adopt the mutated mixture_weights only if the performance after mutation is better
                    self.neighbourhood.mixture_weights_generators = new_mixture_weights_generators
                    self.score = score_after_mutation
                else:
                    # Do not adopt the mutated mixture_weights here
                    self.score = score_before_mutation

    def generate_random_fitness_samples(self, fitness_sample_size):
        """
        Generate random samples for fitness evaluation according to fitness_sample_size

        Abit of hack, use iterator of batch_size to sample data of fitness_sample_size
        TODO Implement another iterator (and dataloader) of fitness_sample_size
        """

        def get_next_batch(iterator, loaded):
            # Handle if the end of iterator is reached
            try:
                return next(iterator)[0], iterator
            except StopIteration:
                # Use a new iterator
                iterator = iter(loaded)
                return next(iterator)[0], iterator

        sampled_data, self.fitness_iterator = get_next_batch(self.fitness_iterator, self.fitness_loaded)
        batch_size = sampled_data.size(0)

        if self.cc.settings['dataloader']['dataset_name'] == 'network_traffic':
            return to_pytorch_variable(generate_random_sequences(fitness_sample_size))

        if fitness_sample_size < batch_size:
            sampled_data = sampled_data[:fitness_sample_size]
        else:
            fitness_sample_size -= batch_size
            while fitness_sample_size >= batch_size:
                # Keep concatenate a full batch of data
                curr_data, self.fitness_iterator = get_next_batch(self.fitness_iterator, self.fitness_loaded)
                sampled_data = torch.cat((sampled_data, curr_data), 0)
                fitness_sample_size -= batch_size

            if fitness_sample_size > 0:
                # Concatenate partial batch of data
                curr_data, self.fitness_iterator = get_next_batch(self.fitness_iterator, self.fitness_loaded)
                sampled_data = torch.cat((sampled_data, curr_data[:fitness_sample_size]), 0)

        if self.cc.settings['dataloader']['dataset_name'] == 'celeba' \
                or self.cc.settings['dataloader']['dataset_name'] == 'cifar':
            fitness_samples = to_pytorch_variable(sampled_data)
        else:
            fitness_samples = to_pytorch_variable(sampled_data.view(self.fitness_sample_size, -1))

        return fitness_samples
