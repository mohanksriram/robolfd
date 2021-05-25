import subprocess
amp_factors = [0]#[1, 2, 5, 10, 20]
batch_sizes = [10, 64, 128, 256]#[10, 100, 200, 500, 1000]
max_episodes = [40]#[2, 10, 20, 40]#[10, 30, 50, 100]
hidden_sizes = [10, 50, 100, 200, 500]
for i in range(len(amp_factors)):
    for j in range(len(batch_sizes)):
        for k in range(len(max_episodes)):
            for l in range(len(hidden_sizes)):
                p = subprocess.Popen(('python', 'examples/offline/run_bc.py',
                                        '--train_iterations=10000', f'--max_episodes={max_episodes[k]}',
                                        '--num_actors=10', '--video_path=/home/mohan/research/experiments/bc/panda_lift/rollouts/',
                                        f'--batch_size={batch_sizes[j]}', f'--amp_factor={amp_factors[i]}', f'--hidden_size={hidden_sizes[l]}'))
                p.wait()