import os
import torch
import re
import sys
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from src.model import E2E0
from src.utils import summary, cycle
from src.loss import bce
from src.dataset import HYBRID
from evaluate import evaluate

now_dir = os.getcwd()
sys.path.append(now_dir)

def find_latest_iteration(logdir):
    if not os.path.exists(logdir):
        return None
    
    model_files = [f for f in os.listdir(logdir) if f.startswith('model_') and f.endswith('.pt')]
    
    iterations = []
    for f in model_files:
        match = re.search(r'model_(\d+)\.pt', f)
        if match:
            iterations.append(int(match.group(1)))
    
    return max(iterations) if iterations else None


def train():
    logdir = '/content/drive/MyDrive/RMPVE'

    hop_length = 160
    optimizer_type = 'adam'
    learning_rate = 5e-4
    batch_size = 16
    validation_interval = 2000
    log_interval = 10
    clip_grad_norm = 3
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    only_latest = False

    train_dataset = HYBRID('/content/drive/MyDrive/dataset', hop_length, ['train'], whole_audio=False, use_aug=True)
    validation_dataset = HYBRID('/content/drive/MyDrive/dataset', hop_length, ['test'], whole_audio=True, use_aug=False)

    data_loader = DataLoader(train_dataset, batch_size, shuffle=True, drop_last=True, pin_memory=True, persistent_workers=True, num_workers=2)
    
    iterations = 200000
    learning_rate_decay_steps = 2000
    warmup_steps = int(len(data_loader) * 3)
    learning_rate_decay_rate = 0.99

    resume_path = None
    if only_latest:
        potential_path = os.path.join(logdir, 'model_latest.pt')
        if os.path.exists(potential_path):
            resume_path = potential_path
    else:
        latest_iter = find_latest_iteration(logdir)
        if latest_iter is not None:
            resume_path = os.path.join(logdir, f'model_{latest_iter}.pt')
        elif os.path.exists(os.path.join(logdir, 'model_latest.pt')):
            resume_path = os.path.join(logdir, 'model_latest.pt')

    if resume_path and os.path.exists(resume_path):
        should_resume = True
    else:
        should_resume = False
        resume_iteration = 0 

    os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(logdir)
    
    model = E2E0(1, 1, 16).to(device)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs", flush=True)
        model = nn.DataParallel(model)

    if optimizer_type == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=learning_rate_decay_steps, gamma=learning_rate_decay_rate)

    best_rpa = 0.0

    if should_resume:
        print(f"Resuming from {resume_path}", flush=True)
        ckpt = torch.load(resume_path, map_location=torch.device(device), weights_only=False)
        
        state_dict = ckpt['model']
        if isinstance(model, nn.DataParallel):
             if list(state_dict.keys())[0].startswith('module.'):
                 model.load_state_dict(state_dict)
             else:
                 new_state_dict = {'module.'+k: v for k, v in state_dict.items()}
                 model.load_state_dict(new_state_dict)
        else:
            if list(state_dict.keys())[0].startswith('module.'):
                new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                model.load_state_dict(new_state_dict)
            else:
                model.load_state_dict(state_dict)
        
        if 'optimizer' in ckpt:
            try:
                optimizer.load_state_dict(ckpt['optimizer'])
            except:
                pass
        if 'scheduler' in ckpt:
            try:
                scheduler.load_state_dict(ckpt['scheduler'])
            except:
                pass
        
        resume_iteration = ckpt.get('iteration', 0)
        best_rpa = ckpt.get('best_rpa', 0.0) 

    if not isinstance(model, nn.DataParallel):
        summary(model)

    RPA, RCA, OA, VFA, VR = 0, 0, 0, 0, 0

    for i, data in zip(range(resume_iteration + 1, iterations + 1), cycle(data_loader)):
        if i <= warmup_steps:
            warmup_factor = float(i) / float(warmup_steps)
            warmup_factor = max(warmup_factor, 1e-6)
            
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate * warmup_factor
        
        mel = data['mel'].to(device)
        pitch_label = data['pitch'].to(device)
        pitch_pred = model(mel)
        loss = bce(pitch_pred, pitch_label)

        optimizer.zero_grad()
        loss.backward()
        if clip_grad_norm:
            clip_grad_norm_(model.parameters(), clip_grad_norm)
        optimizer.step()
        if i > warmup_steps:
            scheduler.step()

        # Запись в TB каждые log_interval итераций
        if i % log_interval == 0:
            writer.add_scalar('loss/loss_pitch', loss.item(), global_step=i)

        # print каждые log_interval итераций
        if i % log_interval == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f"Iter {i}/{iterations} | Loss: {loss.item():.6f} | LR: {lr:.2e}", flush=True)

        if i % validation_interval == 0:
            model.eval()
            with torch.no_grad():
                eval_model = model.module if isinstance(model, nn.DataParallel) else model
                metrics = evaluate(validation_dataset, eval_model, hop_length, device)
                for key, value in metrics.items():
                    writer.add_scalar('stage_pitch/' + key, np.mean(value), global_step=i)
                rpa = np.mean(metrics['RPA'])
                rca = np.mean(metrics['RCA'])
                oa = np.mean(metrics['OA'])
                vr = np.mean(metrics['VR'])
                vfa = np.mean(metrics['VFA'])
                
                RPA, RCA, OA, VR, VFA = rpa, rca, oa, vr, vfa
                
                print(f"=== Validation @ {i} | RPA: {rpa:.4f} | RCA: {rca:.4f} | OA: {oa:.4f} ===", flush=True)
                
                with open(os.path.join(logdir, 'result.txt'), 'a') as f:
                    f.write(str(i) + '\t')
                    f.write(str(RPA) + '\t')
                    f.write(str(RCA) + '\t')
                    f.write(str(OA) + '\t')
                    f.write(str(VR) + '\t')
                    f.write(str(VFA) + '\n')
                
                is_best = False
                if rpa >= best_rpa:
                    best_rpa = rpa
                    is_best = True
                    print(f'New best model at {i}!', flush=True)

                model_to_save = model.module if isinstance(model, nn.DataParallel) else model
                checkpoint_dict = {
                    'iteration': i,
                    'model': model_to_save.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'best_rpa': best_rpa
                }

                if is_best:
                    torch.save(checkpoint_dict, os.path.join(logdir, 'model_best.pt'))

                model_filename = 'model_latest.pt' if only_latest else f'model_{i}.pt'
                torch.save(checkpoint_dict, os.path.join(logdir, model_filename))
                
            model.train()

    print("Training finished.", flush=True)
    writer.close()

if __name__ == '__main__':
    try:
        train()
    except Exception as e:
        print(e)
