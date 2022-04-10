import os

seeds = [131,712,11,776,364,423,111,642,594,835,540,262,344,661,872,406,620,544,822,993,73,538,355,405,987,593,491,823,929,4,657,230,235,49,921,806,802,644,38,741,665,68,816,637,800,69,173,327,633,599]

for arg in seeds:
    print(arg)
    os.system("python main_img_save.py --result_folder 'final_res_more_seed' --seed {} --sampler 'HMC' --epsilon 0.2 --k 50 --step_size 100000 --mh_reject True".format(arg))
