'''
Config for simulated instances
'''

n_services_range = (120, 340)
n_services = [i for i in range(n_services_range[0], n_services_range[1]+1, 40)]
scenarios = ["easy", "normal", "hard"]
seeds = range(10)