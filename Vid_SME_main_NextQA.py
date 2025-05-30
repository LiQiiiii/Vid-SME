from longva.model.builder import load_pretrained_model
from longva.mm_utils import tokenizer_image_token
from longva.constants import IMAGE_TOKEN_INDEX
from decord import VideoReader, cpu
import torch
import numpy as np
import json
from tqdm import tqdm
import os
import random
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset
from sklearn.metrics import auc, roc_curve
import cv2
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
cuda_available_devices = [0, 1, 2, 3, 4, 5, 6, 7] 
selected_gpu = cuda_available_devices[3] 
torch.cuda.set_device(selected_gpu) 
device = torch.device(f"cuda:{selected_gpu}")

def sweep(score, x):
    fpr, tpr, _ = roc_curve(x, -score)
    acc = np.max(1-(fpr+(1-tpr))/2)
    return fpr, tpr, auc(fpr, tpr), acc

def fpr_tpr(all_output, output_dir):
    method_metrics = defaultdict(lambda: defaultdict(list))
    for ex in all_output:
        label = ex["label"]
        for method, preds in ex["pred_mia"].items():
            for metric, prediction in preds.items():
                if ("raw" in metric) and ("clf" not in metric):
                    continue
                method_metrics[method][metric].append((prediction, label))
    for method, metrics in method_metrics.items():
        method_output_dir = f"{output_dir}/{method}"
        os.makedirs(method_output_dir, exist_ok=True)
        save_dict = {}
        with open(f"{method_output_dir}/auc.txt", "w") as f:
            for metric, data in metrics.items():
                predictions, labels = zip(*data)
                arr = np.array(predictions)
                finite_mask = np.isfinite(arr) 
                finite_vals = arr[finite_mask]
                val_max = finite_vals.max()
                val_min = finite_vals.min()
                arr[np.isposinf(arr)] = val_max
                arr[np.isneginf(arr)] = val_min
                fpr, tpr, auc, acc = sweep(arr, np.array(labels, dtype=bool))
                low = tpr[np.where(fpr<.05)[0][-1]]
                print('Attack %s   AUC %.4f, Accuracy %.4f, TPR@5%%FPR of %.4f\n'%(metric, auc, acc, low))
                    
                f.write(f'{metric}   AUC {auc:.4f}, Accuracy {acc:.4f}, TPR@5% FPR of {low:.4f}\n')
                save_dict[f"{metric}_fpr"] = fpr
                save_dict[f"{metric}_tpr"] = tpr
                save_dict[f"{metric}_auc"] = np.array([auc])
                save_dict[f"{metric}_acc"] = np.array([acc])
                save_dict[f"{metric}_low5"] = np.array([low])
            np.savez(f"{method_output_dir}/roc_data.npz", **save_dict)

def get_img_metric(
    ppl, all_prob, p1_likelihood, entropies, mod_entropy, max_p, org_prob, 
    gap_p, renyi_05, renyi_2, log_probs, mod_renyi_05, mod_renyi_2, sequence_vetp, vetp, sequence_vetp_inverse, vetp_inverse
):
    pred = {}

    # Perplexity
    pred["ppl"] = ppl

    # Min-K% Probability Scores
    for ratio in [0, 0.05, 0.1, 0.3, 0.6, 0.9]:
        k_length = max(1, int(len(all_prob) * ratio))
        topk_prob = np.sort(all_prob)[:k_length]
        pred[f"Min_{ratio*100}% Prob"] = -np.mean(topk_prob).item()

    # Modified entropy metrics
    pred["Modified_entropy"] = np.nanmean(mod_entropy).item()
    pred["Modified_renyi_05"] = np.nanmean(mod_renyi_05).item()
    pred["Modified_renyi_2"] = np.nanmean(mod_renyi_2).item()

    # Probability gap metric
    pred["Max_Prob_Gap"] = -np.mean(gap_p).item()

    # Max-K% Renyi Entropy Values
    for ratio in [0, 0.05, 0.1, 0.3, 0.6, 0.9, 1]:
        k_length = max(1, int(len(renyi_05) * ratio))

        pred[f"Max_{ratio*100}% renyi_05"] = np.mean(np.sort(renyi_05)[-k_length:]).item()
        pred[f"Max_{ratio*100}% renyi_1"] = np.mean(np.sort(entropies)[-k_length:]).item()
        pred[f"Max_{ratio*100}% renyi_2"] = np.mean(np.sort(renyi_2)[-k_length:]).item()
        pred[f"Max_{ratio*100}% renyi_inf"] = np.mean(np.sort(-np.array(max_p))[-k_length:]).item()

    # Min-K% Renyi Entropy Values
    for ratio in [0, 0.05, 0.1, 0.3, 0.6, 0.9, 1]:
        k_length = max(1, int(len(renyi_05) * ratio))

        pred[f"Min_{ratio*100}% renyi_05"] = np.mean(np.sort(renyi_05)[:k_length]).item()
        pred[f"Min_{ratio*100}% renyi_1"] = np.mean(np.sort(entropies)[:k_length]).item()
        pred[f"Min_{ratio*100}% renyi_2"] = np.mean(np.sort(renyi_2)[:k_length]).item()
        pred[f"Min_{ratio*100}% renyi_inf"] = np.mean(np.sort(-np.array(max_p))[:k_length]).item()

    # Sharma–Mittal Entropy
    vetp_diff = [x - y for x, y in zip(vetp, vetp_inverse)]
    pred["Max_vetp"] = np.max(vetp_diff).item()
    pred["Min_vetp"] = np.min(vetp_diff).item()
    pred["Mean_vetp"] = np.mean(vetp_diff).item()
    for ratio in [0, 0.05, 0.1, 0.3, 0.6, 0.9, 1]:
        k_length = max(1, int(len(vetp) * ratio))
        pred[f"Max_{ratio*100}% vetp"] = np.mean(np.sort(vetp_diff)[-k_length:]).item()
        pred[f"Min_{ratio*100}% vetp"] = np.mean(np.sort(vetp_diff)[:k_length]).item()
    pred["sequence_vetp"] = sequence_vetp - sequence_vetp_inverse
    return pred


def compute_sharma_mittal_entropy(prob_dist, q, r, epsilon=1e-10):
    prob_dist = torch.clamp(prob_dist, min=epsilon, max=1 - epsilon)
    if abs(q - r) < epsilon:
        sum_q = torch.sum(prob_dist ** q)
        sm_entropy = (1 / (1 - q + epsilon)) * (sum_q - 1)
    elif abs(r - 1) < epsilon:
        sum_q = torch.sum(prob_dist ** q)
        sum_q = torch.clamp(sum_q, min=epsilon)
        sm_entropy = (1 / (1 - q + epsilon)) * torch.log(sum_q)
    elif abs(q - 1) < epsilon and abs(r - 1) < epsilon:
        sm_entropy = -torch.sum(prob_dist * torch.log(prob_dist))
    else:
        sum_q = torch.sum(prob_dist ** q)
        sum_q = torch.clamp(sum_q, min=epsilon)
        exponent = (1 - r) / (1 - q + epsilon)
        if abs(exponent) > 100:
            exponent = torch.tensor(exponent, dtype=torch.float32)
            exponent = torch.clamp(exponent, -100, 100)
        sm_entropy = (1 / (1 - r + epsilon)) * (sum_q ** exponent - 1)

    return sm_entropy

def get_overall_sharma_mittal_entropy(probabilities, q, r):
    avg_probs = torch.mean(probabilities, dim=0)      
    avg_probs = avg_probs / avg_probs.sum()       
    sm_entropy_all = compute_sharma_mittal_entropy(avg_probs, q, r)

    return sm_entropy_all.item()

def get_meta_metrics(input_ids, probabilities, log_probabilities, q, r, prefix):
    entropies = []
    all_prob = []
    modified_entropies = []
    max_prob = []
    gap_prob = []
    renyi_05 = []
    renyi_2 = []
    losses = []
    modified_entropies_alpha05 = []
    modified_entropies_alpha2 = []
    sharma_mittal_entropies = []
    epsilon = 1e-10

    input_ids_processed = input_ids[1:] 
    for i, token_id in enumerate(input_ids_processed):
        token_probs = probabilities[i, :]  
        token_probs = token_probs.clone().detach().to(dtype=torch.float64)
        token_log_probs = log_probabilities[i, :] 
        token_log_probs = token_log_probs.clone().detach().to(dtype=torch.float64)
        
        entropy = -(token_probs * token_log_probs).sum().item() 
        entropies.append(entropy)

        token_probs_safe = torch.clamp(token_probs, min=epsilon, max=1-epsilon)

        alpha = 0.5
        renyi_05_ = (1 / (1 - alpha)) * torch.log(torch.sum(torch.pow(token_probs_safe, alpha))).item()
        renyi_05.append(renyi_05_)
        alpha = 2
        renyi_2_ = (1 / (1 - alpha)) * torch.log(torch.sum(torch.pow(token_probs_safe, alpha))).item()
        renyi_2.append(renyi_2_)

        max_p = token_log_probs.max().item()
        vals = token_log_probs[token_log_probs != token_log_probs.max()]  

        if vals.numel() == 0:
            second_p = max_p
        else:
            second_p = token_log_probs[token_log_probs != token_log_probs.max()].max().item()

        gap_p = max_p - second_p
        gap_prob.append(gap_p)
        max_prob.append(max_p)

        mink_p = token_log_probs[token_id].item()
        all_prob.append(mink_p)

        cross_entropy_loss = -mink_p
        losses.append(cross_entropy_loss)

        p_y = token_probs_safe[token_id].item()
        modified_entropy = -(1 - p_y) * torch.log(torch.tensor(p_y)) - (token_probs * torch.log(1 - token_probs_safe)).sum().item() + p_y * torch.log(torch.tensor(1 - p_y)).item()
        modified_entropies.append(modified_entropy)

        token_probs_remaining = torch.cat((token_probs_safe[:token_id], token_probs_safe[token_id+1:]))
        
        for alpha in [0.5,2]:
            entropy = - (1 / abs(1 - alpha)) * (
                (1-p_y)* p_y**(abs(1-alpha))\
                    - (1-p_y)
                    + torch.sum(token_probs_remaining * torch.pow(1-token_probs_remaining, abs(1-alpha))) \
                    - torch.sum(token_probs_remaining)
                    ).item() 
            if alpha==0.5:
                modified_entropies_alpha05.append(entropy)
            if alpha==2:
                modified_entropies_alpha2.append(entropy)
        epsilon = 1e-10 
        if abs(q - r) < epsilon: 
            sum_q = torch.sum(torch.pow(torch.clamp(token_probs_safe, min=epsilon), q))
            sm_entropy = (1 / (1 - q + epsilon)) * (sum_q - 1)
        elif abs(r - 1) < epsilon:  
            sum_q = torch.sum(torch.pow(torch.clamp(token_probs_safe, min=epsilon), q))
            sum_q = torch.clamp(sum_q, min=epsilon)  
            sm_entropy = (1 / (1 - q + epsilon)) * torch.log(sum_q)
        elif abs(q - 1) < epsilon and abs(r - 1) < epsilon: 
            probs_safe = torch.clamp(token_probs_safe, min=epsilon)
            sm_entropy = -torch.sum(probs_safe * torch.log(probs_safe))
        else:  
            sum_q = torch.sum(torch.pow(torch.clamp(token_probs_safe, min=epsilon), q))
            sum_q = torch.clamp(sum_q, min=epsilon) 
            exponent = (1 - r) / (1 - q + epsilon)
            if abs(exponent) > 100:  
                exponent = torch.tensor(exponent, dtype=torch.float32)
                exponent = torch.clamp(exponent, -100, 100)
            sm_entropy = (1 / (1 - r + epsilon)) * (sum_q ** exponent - 1)
        
        sharma_mittal_entropies.append(sm_entropy.item())
    overall_sm_entropy = get_overall_sharma_mittal_entropy(probabilities, q, r)
    loss = np.nanmean(losses)

    return {
        "ppl": np.exp(loss),
        "all_prob": all_prob,
        "loss": loss,
        "entropies": entropies,
        "modified_entropies": modified_entropies,
        "max_prob": max_prob,
        "probabilities": probabilities,
        "log_probs" : log_probabilities,
        "gap_prob": gap_prob,
        "renyi_05": renyi_05,
        "renyi_2": renyi_2,
        "mod_renyi_05" : modified_entropies_alpha05,
        "mod_renyi_2" : modified_entropies_alpha2,
        "sequence_sm_entropy": overall_sm_entropy,
        "sm_entropy": sharma_mittal_entropies
    }

def get_prompt2(conv):
    ret = conv.system + conv.sep
    count = 0
    for role, message in conv.messages:
        count += 1
        if count == len(conv.messages):
            ret += role + ": " + message
        else:
            if message:
                ret += role + ": " + message + conv.sep
            else:
                ret += role + ":"
    return ret

class MLVU(Dataset):
    def __init__(self, data_dir, data_list):
        self.data_list = []
        for k, v in data_list.items():
            with open(os.path.join(data_dir, v[0]), 'r') as f:
                json_data = json.load(f)
            for data in json_data:
                self.data_list.append({
                    'task_type': k,
                    'prefix': v[1],
                    'data_type': v[2],
                    'label': 1 if k == "train" else 0,
                    'data': data
                })
        
    
    def __str__(self):
        len_list = {}
        option_list = {}
        for data in self.data_list:
            if data['task_type'] not in len_list:
                len_list[data['task_type']] = 0
            len_list[data['task_type']] += 1
            if data['task_type'] not in option_list:
                option_list[data['task_type']] = 0
            option_list[data['task_type']] += len(data['data']['candidates'])
        
        correct = 0
        total = 0
        res = f"There are {len(self.data_list)} videos as follow:\n"
        for k, v in len_list.items():
            correct += len_list[k]
            total += option_list[k]
            res += f"{v} for {k} ({option_list[k]} options => {len_list[k]/option_list[k]*100:.2f}%)\n"
            correct = correct + 1 / option_list[k]
        res += f"Total random accuracy: {correct/total*100:.2f}%"
        return res.rstrip()
        
    def __len__(self):
        return len(self.data_list)
    
    def get_index(self, bound, fps, max_frame, first_idx=0):
        start, end = -100000, 100000
        start_idx = max(first_idx, round(start * fps))
        end_idx = min(round(end * fps), max_frame)
        seg_size = float(end_idx - start_idx) / self.num_segments
        frame_indices = np.array([
            int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
            for idx in range(self.num_segments)
        ])
        return frame_indices
    

    def qa_template(self, data):
        question = f"Question: {data['question']}\n"
        question += "Options:\n"
        answer = data['answer']
        answer_idx = -1
        for idx, c in enumerate(data['candidates']):
            question += f"({chr(ord('A') + idx)}) {c}\n"
            if c == answer:
                answer_idx = idx
        question = question.rstrip()
        answer = f"({chr(ord('A') + answer_idx)}) {answer}"
        return question, answer

    def __getitem__(self, idx):
        video_path = os.path.join(self.data_list[idx]['prefix'], self.data_list[idx]['data']['video'])
        question, answer = self.qa_template(self.data_list[idx]['data'])
        return {
            'video': video_path, 
            'question': question, 
            'answer': answer,
            'task_type': self.data_list[idx]['task_type'],
            'label': self.data_list[idx]['label']
        }



def check_ans(pred, gt):
    flag = False

    index=gt.index("(")
    index2=gt.index(")")
    gt_option=gt[index+1:index2]

    if ")" in pred:
        index3=pred.index(")")
        pred=pred[index3-1:index3]

    print("2222222",pred,gt_option)
    if pred==gt_option:
        print("11111111111111",pred,gt_option)
        flag=True

    return flag




data_list = {
    "train": ("./video_json/nextqa_train_official.json", f"./video_json/videos", "video"),
    "test": ("./video_json/nextqa_test_official.json", f"./video_json/videos", "video")
    }

data_dir = f"./"
frame_list = [16]
beta_1 = 1.0
beta_2 = 0.1
for max_frames_num in frame_list:
    model_type = f"vidsme_nextqa_frm{max_frames_num}"
    save_path = f"./statistics_{model_type}"

    dataset = MLVU(data_dir, data_list)
    torch.manual_seed(0)
    model_path = "./checkpoints/Vid-SME-NExTQA-7B"

    gen_kwargs = {"do_sample": True, "temperature": 0.1, "top_p": None, "num_beams": 1, "use_cache": True, "max_new_tokens": 256}
    tokenizer, model, image_processor, _ = load_pretrained_model(model_path, None, "llava_qwen", device_map="auto")

    correct = 0
    total = 0
    res_list = []
    acc_dict = {}
    all_output = []
    idx = 0
    v_etp_list = []
    def compute_optical_flow_variance(frames):
        prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY)
        flow_variances = []
        for i in range(1, len(frames)):
            next_gray = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            flow_variances.append(np.var(flow))
            prev_gray = next_gray 

        return np.mean(flow_variances) if flow_variances else 0 

    def compute_brightness_variance(frames):
        brightness_values = []
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            brightness_values.append(np.mean(gray))

        return np.std(brightness_values)
    optical_flow_var_list = []
    brightness_var_list = []
    q_file = f"q_values_nextqa_frame_{max_frames_num}.npy"
    r_file = f"r_values_nextqa_frame_{max_frames_num}.npy"
    if os.path.exists(q_file) and os.path.exists(r_file):
        q_values = np.load(q_file)
        r_values = np.load(r_file)
        print("q_values and r_values loaded from files!")
    else:
        print("Files not found! Computing q_values and r_values...")            
        for example in tqdm(dataset):
            video_path = example["video"]
            
            vr = VideoReader(video_path, ctx=cpu(0))
            total_frame_num = len(vr)
            frame_idx = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int).tolist()
            frames = vr.get_batch(frame_idx).asnumpy()
            optical_flow_var = compute_optical_flow_variance(frames)
            optical_flow_var_list.append(optical_flow_var)
            brightness_var = compute_brightness_variance(frames)
            brightness_var_list.append(brightness_var)
        optical_flow_var_array = np.array(optical_flow_var_list)
        brightness_var_array = np.array(brightness_var_list)
        scaler_k1 = StandardScaler()
        scaler_k3 = StandardScaler()
        optical_flow_var_norm = scaler_k1.fit_transform(optical_flow_var_array.reshape(-1, 1)).flatten()
        brightness_var_norm = scaler_k3.fit_transform(brightness_var_array.reshape(-1, 1)).flatten()
        min_k1, max_k1 = np.min(optical_flow_var_norm), np.max(optical_flow_var_norm)
        min_k3, max_k3 = np.min(brightness_var_norm), np.max(brightness_var_norm)
        q_values = 1 + beta_1 * (max_k1 - optical_flow_var_norm) / (max_k1 - min_k1)
        r_values = 1 + beta_2 * (brightness_var_norm - min_k3) / (max_k3 - min_k3)
        np.save(q_file, q_values)
        np.save(r_file, r_values)
        print("Computed and saved q_values and r_values!")

        plt.figure(figsize=(10, 6))
        plt.hist(q_values, bins=5, alpha=0.6, label="q value", color="darkred")
        plt.hist(r_values, bins=5, alpha=0.6, label="r value", color="darkgreen")
        plt.xlabel("Value", fontsize=17) 
        plt.ylabel("Frequency", fontsize=17)   
        plt.xticks(fontsize=14)          
        plt.yticks(fontsize=14)            
        plt.legend(fontsize=14)      
        plt.grid(True)
        plt.savefig(f"{model_type}.png")


    idx_qr_value = 0
    idx_now = 1
    for example in tqdm(dataset):
        q_value = q_values[idx_qr_value]
        r_value = r_values[idx_qr_value]
        idx_qr_value += 1
        task_type = example['task_type']
        if task_type not in acc_dict:
            acc_dict[task_type] = [0, 0] # correct, total
        acc_dict[task_type][1] += 1
        total += 1
        video_path = example["video"]
        
        inp=example["question"] + "\nOnly give the best option."
        #video input
        prompt1="<|im_start|>system\nCarefully watch this video and pay attention to every detail. Based on your observations, select the best option that accurately addresses the question.<|im_end|>\n<|im_start|>user\n<image>\n"
        prompt2=inp
        prompt3="|im_end|>\n<|im_start|>assistant\nBest Option: "
        prompt=prompt1 + prompt2 + prompt3
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)
        prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split("<image>")]
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frame_num = len(vr)
        frame_idx = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int).tolist()
        frames = vr.get_batch(frame_idx).asnumpy()

        frames_inverse = frames[::-1] 
        video_tensor = image_processor.preprocess(frames, return_tensors="pt")["pixel_values"].to(model.device, dtype=torch.float16)
        video_tensor_inverse = image_processor.preprocess(frames_inverse, return_tensors="pt")["pixel_values"].to(model.device, dtype=torch.float16)

        with torch.inference_mode():
            output_ids = model.generate(input_ids, images=[video_tensor], modalities=["video"], **gen_kwargs)
        pred = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        descp_encoding = tokenizer(pred, return_tensors="pt", add_special_tokens=False).to(model.device).input_ids
        with torch.no_grad():
            outputs = model(input_ids=input_ids, images=[video_tensor], modalities=["video"])
            outputs_inverse = model(input_ids=input_ids, images=[video_tensor_inverse], modalities=["video"])

        logits = outputs.logits
        logits_inverse = outputs_inverse.logits
        video_token_id = tokenizer.convert_tokens_to_ids(input_ids[0])
        goal_slice_dict = {
            'img' : slice(len(prompt_chunks[0]),-len(prompt_chunks[-1])+1),
            }
        img_loss_slice = logits[0, goal_slice_dict['img'].start-1:goal_slice_dict['img'].stop-1, :]
        img_target_np = torch.nn.functional.softmax(img_loss_slice, dim=-1).cpu().numpy()
        max_indices = np.argmax(img_target_np, axis=-1)
        img_max_input_id = torch.from_numpy(max_indices).to(model.device)

        img_loss_slice_inverse = logits_inverse[0, goal_slice_dict['img'].start-1:goal_slice_dict['img'].stop-1, :]
        img_target_np_inverse = torch.nn.functional.softmax(img_loss_slice_inverse, dim=-1).cpu().numpy()
        max_indices_inverse = np.argmax(img_target_np_inverse, axis=-1)
        img_max_input_id_inverse = torch.from_numpy(max_indices_inverse).to(model.device)

        tensor_a = torch.tensor(prompt_chunks[0]).to(model.device) if not isinstance(prompt_chunks[0], torch.Tensor) else prompt_chunks[0]
        tensor_b = torch.tensor(prompt_chunks[-1][1:]).to(model.device) if not isinstance(prompt_chunks[-1][1:], torch.Tensor) else prompt_chunks[-1][1:]
        mix_input_ids = torch.cat([tensor_a, img_max_input_id, tensor_b], dim=0)
        mix_input_ids_inverse = torch.cat([tensor_a, img_max_input_id_inverse, tensor_b], dim=0)

        goal_parts = ['img']
        all_pred = {}

        v_etp = []
        for goal in goal_parts:
            target_slice = goal_slice_dict[goal]
            logits_slice = logits[0, target_slice, :]
            input_ids = mix_input_ids[target_slice]
            probabilities = torch.nn.functional.softmax(logits_slice, dim=-1)
            log_probabilities = torch.nn.functional.log_softmax(logits_slice, dim=-1)

            logits_slice_inverse = logits_inverse[0, target_slice, :]
            input_ids_inverse = mix_input_ids_inverse[target_slice]
            probabilities_inverse = torch.nn.functional.softmax(logits_slice_inverse, dim=-1)
            log_probabilities_inverse = torch.nn.functional.log_softmax(logits_slice_inverse, dim=-1)

            prefix = f"{goal}_{idx_now}"
            idx_now += 1
            metrics = get_meta_metrics(input_ids, probabilities, log_probabilities, q_value, r_value, prefix)
            metrics_inverse = get_meta_metrics(input_ids_inverse, probabilities_inverse, log_probabilities_inverse, q_value, r_value, prefix)

            pred_mia = get_img_metric(
            metrics["ppl"], metrics["all_prob"], metrics["loss"], metrics["entropies"], 
            metrics["modified_entropies"], metrics["max_prob"], metrics["probabilities"], 
            metrics["gap_prob"], metrics["renyi_05"], metrics["renyi_2"], metrics["log_probs"],
            metrics["mod_renyi_05"], metrics["mod_renyi_2"], metrics["sequence_sm_entropy"], metrics["sm_entropy"], metrics_inverse["sequence_sm_entropy"], metrics_inverse["sm_entropy"]
            )

            all_pred[goal] = pred_mia
        example["pred_mia"] = all_pred
        all_output.append(example)

        gt = example['answer']
        print("##########")
        print("GT",gt)
        print("Pred",pred)
        print("##########")

        res_list.append({
            'pred': pred,
            'gt': gt,
            'question':example['question'],
            'question_type':example['task_type'],
            'video':example['video']
        })
        if check_ans(pred=pred, gt=gt):
            acc_dict[task_type][0] += 1
            correct += 1
        print(f"Part  Acc: {acc_dict[task_type][0] / acc_dict[task_type][1] * 100 :.2f}%")
        print('-' * 30, task_type, '-' * 30)

    fpr_tpr(all_output, save_path)
    del model
    del tokenizer
    del image_processor
    torch.cuda.empty_cache()  

