import logging
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import numpy as np
from torchvision import transforms
from sklearn.metrics import accuracy_score

from seed import seed_everything
from attacks.Attack import AttackError

from attacks.adaptive.Square import Square
from attacks.adaptive.NESScore import NESScore
from attacks.adaptive.HSJA import HSJA
from attacks.adaptive.QEBA import QEBA
from attacks.adaptive.SurFree import SurFree
from attacks.adaptive.Boundary import Boundary


@torch.no_grad()
def natural_performance(model, loader):
    logging.info("Computing natural accuracy")
    y_true, y_pred = [], []
    pbar = tqdm(range(0, len(loader)), colour="red")
    for i, (x, y, p) in (enumerate(loader)):
        x, y = x.cuda(), y.cuda()
        start = time.time()
        logits, is_cache = model(x)
        end = time.time()
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy().tolist()

        logging.info(
            f"True Label : {y[0]} | Predicted Label : {preds[0]} | is_cache : {is_cache[0]} | latency : {end - start}")

        if model.config["action"] == "rejection":
            preds = [preds[j] if not is_cache[j] else -1 for j in range(len(preds))]
        true = y.detach().cpu().numpy().tolist()
        y_true.extend(true)
        y_pred.extend(preds)
        pbar.update(1)
        pbar.set_description(
            "Running accuracy: {} | hits : {}".format(accuracy_score(y_true, y_pred), model.cache_hits))
    logging.info("FINISHED")
    return accuracy_score(y_true, y_pred)

def benign_loader(model, loader):
    good1x_values=[]
    good1y_values=[]
    good2x_values=[]
    good2y_values=[]

    pbar = tqdm(loader, colour="greeen")
    for i, (x, y, p) in enumerate(pbar):
        x = x.cuda()
        y = y.cuda()

        benignLogits = model.model(x)
        #print ("benignLogits:")
        #print (benignLogits)
        benignLogitsList = torch.nn.functional.softmax(benignLogits, dim=-1)
        sortedBinignList = benignLogitsList[0].tolist()
        sortedBinignList.sort(reverse=True)
        print("benignLogits= ", sortedBinignList)
        print("lable = ", y)
        print("top 2= ", sortedBinignList[0], sortedBinignList[1])

        good1x_values.append(i)
        good1y_values.append(sortedBinignList[0])
        good2x_values.append(i)
        good2y_values.append(sortedBinignList[1])

    logging.info("FINISHED")
    # Create the plot
    plt.figure(figsize=(14, 6))
    plt.plot(good1x_values, good1y_values, marker='o', linestyle='-', markersize=6, color='b', label='Largest Probability Score')
    plt.plot(good2x_values, good2y_values, marker='x', linestyle='-', markersize=6, color='r', label='Second Largest Probability Score')


    # Add labels and title

    plt.xlabel('Input Image Case', fontsize=22)
    plt.ylabel('Probability Score', fontsize=22)
    plt.title('Top-2 Probability Scores Generated from Benign Input', fontsize=22)
    plt.legend(loc='center left', fontsize=15)

    #plt.legend()

    # Show the plot
    #plt.grid(True)
    plt.show()

# @torch.no_grad()
def attack_loader(model, loader, model_config, attack_config):
    # Load attack
    try:
        attacker = globals()[attack_config['attack']](model, model_config, attack_config)
    except KeyError:
        raise NotImplementedError(f'Attack {attack_config["attack"]} not implemented.')

    if attack_config['targeted']:
        target_labels = []
        for _, (_, y, p) in enumerate(loader):
            target_label = y.item()
            while target_label == y.item():
                target_label = np.random.randint(0, len(loader.dataset.targeted_dict))
            target_labels.append(target_label)
    else:
        target_labels = None

    # Run attack and compute adversarial accuracy
    y_true, y_pred = [], []
    x_values = []
    y_values = []

    goodx_values=[]
    goody_values=[]

    x2_values = []
    y2_values = []

    goodx2_values=[]
    goody2_values=[]

    pbar = tqdm(loader, colour="yellow")
    for i, (x, y, p) in enumerate(pbar):
        x = x.cuda()
        y = y.cuda()

        seed_everything()
        try:
            if model.model(x).argmax(dim=1) != y:
                x_adv = x
            elif attack_config['targeted']:
                y_target = target_labels[i]
                x_adv_init = loader.dataset.initialize_targeted(y_target).cuda()
                y_target = torch.tensor([y_target]).cuda()
                x_adv = attacker.attack_targeted(x, y_target, x_adv_init)
            else:
                x_adv = attacker.attack_untargeted(x, y)
        except AttackError as e:
            print(e)
            x_adv = x

        x_adv = x_adv.cuda()
        logits = model.model(x_adv)
        # print("Attackers x_adv is ", x_adv)
        # print("logits:")
        # print (logits)
        logits_list = torch.nn.functional.softmax(logits, dim=-1)
        # print(logits_list)
        sortedList = logits_list[0].tolist()
        sortedList.sort(reverse=True)
        # print(sortedList)
        # print(sortedList[0],sortedList[1])

        x_values.append(i)
        y_values.append(sortedList[0])

        x2_values.append(i)
        y2_values.append(sortedList[1])

        benignLogits = model.model(x)
        # print ("benignLogits:")
        # print (benignLogits)
        benignLogitsList = torch.nn.functional.softmax(benignLogits, dim=-1)
        sortedBinignList = benignLogitsList[0].tolist()
        sortedBinignList.sort(reverse=True)

        # print(sortedBinignList[0],sortedBinignList[1])

        goodx_values.append(i)
        goody_values.append(sortedBinignList[0])
        goodx2_values.append(i)
        goody2_values.append(sortedBinignList[1])

        preds = torch.argmax(logits, dim=1).detach().cpu().numpy().tolist()
        true = y.detach().cpu().numpy().tolist()
        # print("preds:")
        # print (preds)
        # print("true")
        # print(true)

        y_true.extend(true)
        y_pred.extend(preds)

        pbar.set_description("Running Accuracy: {} ".format(accuracy_score(y_true, y_pred)))
        logging.info(
            f"True Label : {true[0]} | Predicted Label : {preds[0]} | Cache Hits / Total Queries : {attacker.get_cache_hits()} / {attacker.get_total_queries()}")
        attacker.reset()
    logging.info("FINISHED")
    # Create the plot
    plt.figure(figsize=(14, 6))
    plt.plot(x_values, y_values, marker='+', markersize=6, linestyle='', color='r', label='Largest Probability Score (Adversary Input)')
    plt.plot(x2_values, y2_values, marker='x', markersize=6, linestyle='', color='r', label='Second Largest Probability Score (Adversary Input)')

    plt.plot(goodx_values, goody_values, marker='+', markersize=6, linestyle='', color='b', label='Largest Probability Score (Benign Input)')
    plt.plot(goodx2_values, goody2_values, marker='x', markersize=6, linestyle='', color='b', label='Second Largest Probability Score (Benign Input)')

    # Add labels and title
    plt.xlabel('Input Image Case', fontsize=22)
    plt.ylabel('Probability Score', fontsize=22)
    plt.title('Top-2 Probability Scores Generated from Benign and Adversary Input', fontsize=22)

    # Move the legend below the plot
    plt.legend(loc='center', bbox_to_anchor=(0.5, -0.3), fontsize=15, ncol=2)

    plt.tight_layout()
    plt.show()

def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits))  # Subtract max for numerical stability
    return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
