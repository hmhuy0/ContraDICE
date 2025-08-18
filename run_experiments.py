import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
delay_between_starts = 100 # Define the delay in seconds

dicts = {
    'exp_name': ['[more_bad_labeled]'],
    'bad_size_list': [
        25
        ],
    'mixed_size_list': [
        # '-1,-1,1',
        # '-1,-1,-1,1',
        '-1,-1,-1,-1,-1,1',
        # '-1,100',

        ],
    'expert_dataset_size': [
        1
        ],

    'env_name': [
                # 'halfcheetah-expert-v2',
                #  'walker2d-expert-v2',
                #  'hopper-expert-v2',
                #  'ant-expert-v2',
                #  'pen-expert-v1',
                #  'relocate-expert-v1',
                #  'hammer-expert-v1',
                #  'door-expert-v1'
                'kitchen-complete-v0'
                 ],
    'seed': [
        1,2,3,4,5
        ],
}
### -------  all mujoco  -------  ##

# add_scripts =  [[
# 'python', '-u', 'train.py',
# '--max_steps=1000000',
# '--bad_name_list=random',
# '--mixed_name_list=random,expert',
# '--is_good_list=0,1',
# '--is_bad_list=1,0',
# '--use_wandb'
# ]]


# ### -------  kitchen  -------  ##

add_scripts =  [[
'python', '-u', 'train.py',
'--max_steps=1000000',
'--bad_name_list=partial',
'--mixed_name_list=partial,partial,partial,partial,partial,complete',
'--is_good_list=0,0,0,0,0,1',
'--is_bad_list=1,1,1,1,1,0',
'--use_wandb'
]]

# # ### -------  all androit  -------  ##

# add_scripts =  [[
# 'python', '-u', 'train.py',
# '--max_steps=1000000',
# '--bad_name_list=cloned',
# '--mixed_name_list=cloned,expert',
# '--is_good_list=0,1',
# '--is_bad_list=1,0',
# '--use_wandb'
# ]]



for key in dicts.keys():
    new_add_scripts = []
    for x in dicts[key]:
        for script in add_scripts:
            new_add_scripts.append(script + [f'--{key}={x}'])
    add_scripts = new_add_scripts



scripts = add_scripts




def run_manual_script(script):
    print(f"Starting script: {' '.join(script)}")
    subprocess.run(script)
    print(f"Finished script: {' '.join(script)}")



if __name__ == '__main__':
    max_processes = min(5, len(scripts))  # Number of scripts to run concurrently
    # run script every 5 seconds
    print('\n'*5)
    print('start running')    
    
    for i, script in enumerate(scripts):
        if (i % max_processes == 0):
            print()
        print(script)

    print('-'*50)
    print(f"Starting {len(scripts)} scripts with a {delay_between_starts}s delay between each start, using {max_processes} concurrent workers.")

    with ThreadPoolExecutor(max_workers=max_processes) as executor:
        future_to_script = {}
        for i, script in enumerate(scripts):
            print(f"Submitting script {i}: {' '.join(script)}")
            future = executor.submit(run_manual_script, script)
            future_to_script[future] = script
            if i < len(scripts) - 1: # Don't sleep after submitting the last script
                 print(f"Waiting {delay_between_starts}s before submitting next script...")
                 time.sleep(delay_between_starts) # Add delay here

        print("All scripts submitted. Waiting for completion...")
        for future in as_completed(future_to_script):
            script_name = future_to_script[future]
            try:
                # result = future.result() # Optional: Check result/exceptions
                print(f"Script completed: {' '.join(script_name)}")
            except Exception as exc:
                print(f"Script {' '.join(script_name)} generated an exception: {exc}")

    print("All scripts finished.")



