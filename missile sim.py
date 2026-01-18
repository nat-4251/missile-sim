import numpy as np
import random
import neat
import time
 
#====================== LOAD CONFIG ======================
config = neat.config.Config(
    neat.DefaultGenome,
    neat.DefaultReproduction,
    neat.DefaultSpeciesSet,
    neat.DefaultStagnation,
    "config-feedforward.txt"
)
 
#====================== SIMULATION =======================
def simulate(genome, config, print_steps=False):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
 
    #======= setup ========

 
    missiles = [
        {"pos": np.array([0.0, 0.0]), "alive": True},
        {"pos": np.array([-40.0, 30.0]), "alive": True},
        {"pos": np.array([20.0, -50.0]), "alive": True}
    ]
 
    ship_pos = np.array([100.0, 60.0])
    ship_hit_radius = 5.0
    missile_speed = 5.0
 
    min_intercept_dist = 10.0
    max_intercept_dist = 60.0
    sam_ammo = 7
    sam_hit_prob = 0.60
    cd_time = 0
 
    fitness = 0
    step = 0
    ship_alive = True
 
    if print_steps:
        print("Simulation starting...\n")
 
    #================== SIM LOOP ===================
    while any(m["alive"] for m in missiles) and ship_alive:
        inputs = []
 
        # Build inputs: distance & alive for each missile
        for missile in missiles:
            if missile["alive"]:
                distance = np.linalg.norm(ship_pos - missile["pos"])
                inputs.extend([distance, 1])
            else:
                inputs.extend([0.0, 0])
 
        # Add ship SAM info
        inputs.append(sam_ammo)
        inputs.append(cd_time)
 
        # Get AI output
        output = net.activate(inputs)
        fire_prob = output[0]      # first output: fire or not
        target_probs = output[1:]  # rest: target probabilities
 
        # Move missiles
        for i, missile in enumerate(missiles):
            if missile["alive"]:
                distance = np.linalg.norm(ship_pos - missile["pos"])
                direction = (ship_pos - missile["pos"]) / distance
                missile["pos"] += direction * missile_speed
 
                if distance <= ship_hit_radius:
                    missile["alive"] = False
                    ship_alive = False
                    if print_steps:
                        print(f"Missile {i} hit the ship!")
                    break
 
        # AI decision: fire?
        if fire_prob > 0.5:
            # Only consider alive missiles
            alive_probs = [p if m["alive"] else -1 for p, m in zip(target_probs, missiles)]
            target_index = alive_probs.index(max(alive_probs))
            target_missile = missiles[target_index]
 
            # Check valid launch conditions
            distance_to_target = np.linalg.norm(ship_pos - target_missile["pos"])
            if sam_ammo > 0 and cd_time == 0 and min_intercept_dist < distance_to_target < max_intercept_dist:
                sam_ammo -= 1
                cd_time = 1
                hit = random.random() < sam_hit_prob
                if hit:
                    target_missile["alive"] = False
                    fitness += 100
                    if print_steps:
                        print(f"AI intercepted missile {target_index} at distance {distance_to_target:.2f}")
                else:
                    fitness += 10  # reward for trying
                    if print_steps:
                        print(f"AI fired at missile {target_index} but missed")
            else:
                fitness -= 20  # penalty for invalid launch
 
        # Update cooldown
        if cd_time > 0:
            cd_time -= 1
 
        fitness += 1  # reward for surviving step
        step += 1
 
    #================== End of simulation ===================
    if not ship_alive:
        fitness -= 300  # penalty for getting hit
        if print_steps:
            print("The ship was destroyed!")
    elif all(not m["alive"] for m in missiles):
        fitness += 100  # bonus for clearing all missiles
        if print_steps:
            print("AI has intercepted all missiles!")
 
    return fitness
 
#================== Evaluation Function ===================
def eval_population(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = simulate(genome, config, print_steps=False)
 
#================== MAIN TRAINING ===================
pop = neat.Population(config)
pop.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
pop.add_reporter(stats)
 
winner = pop.run(eval_population, 50)
 
print("\nTesting best genome...\n")
simulate(winner, config, print_steps=True)
